from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = _REPO_ROOT / "models"

__all__ = ["predict_all_saved_models"]


@dataclass(frozen=True)
class _LoadedModel:
    model_path: Path
    predictor: Any
    diagnostic: str
    expected_columns: list[str]
    threshold: float


def _normalize_feature_names(feature_names: Sequence[str]) -> list[str]:
    normalized = list(feature_names)
    if not normalized:
        raise ValueError("`feature_names` ne doit pas etre vide.")
    if not all(isinstance(name, str) and name for name in normalized):
        raise ValueError("`feature_names` doit contenir uniquement des noms de colonnes non vides.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("`feature_names` doit contenir des noms de colonnes uniques.")
    return normalized


def _merge_feature_names_in_order(all_feature_names: Sequence[Sequence[str]]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for feature_names in all_feature_names:
        for name in feature_names:
            normalized_name = str(name)
            if normalized_name not in seen:
                seen.add(normalized_name)
                merged.append(normalized_name)
    if not merged:
        raise ValueError("Impossible de construire des `feature_names` par defaut.")
    return merged


def _build_input_row(
    feature_values: Sequence[Any],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    values = list(feature_values)
    names = _normalize_feature_names(feature_names)
    if len(values) != len(names):
        raise ValueError("`feature_values` et `feature_names` doivent avoir la meme longueur.")
    return pd.DataFrame([values], columns=names)


def _discover_model_paths(model_dir: str | Path | None) -> list[Path]:
    target_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    if not target_dir.exists():
        raise FileNotFoundError(f"Dossier de modeles introuvable: {target_dir}")

    model_paths = sorted(target_dir.glob("*.joblib"))
    if not model_paths:
        raise FileNotFoundError(f"Aucun fichier .joblib trouve dans: {target_dir}")
    return model_paths


def _extract_predictor(payload: dict[str, Any], model_path: Path) -> Any:
    predictor = payload.get("pipe_inference")
    if predictor is None:
        predictor = payload.get("pipe_train")
    if predictor is None:
        raise ValueError(
            f"Le modele sauvegarde '{model_path.name}' ne contient ni `pipe_inference` ni `pipe_train`."
        )
    return predictor


def _extract_expected_columns(predictor: Any, model_path: Path) -> list[str]:
    if hasattr(predictor, "feature_names_in_"):
        expected_columns = list(getattr(predictor, "feature_names_in_"))
        if expected_columns:
            return [str(column) for column in expected_columns]

    named_steps = getattr(predictor, "named_steps", None)
    scaler = named_steps.get("scaler") if isinstance(named_steps, dict) else None
    if scaler is not None and hasattr(scaler, "colonnes_numeriques"):
        expected_columns = list(getattr(scaler, "colonnes_numeriques") or [])
        if expected_columns:
            return [str(column) for column in expected_columns]

    raise ValueError(
        f"Impossible de determiner les colonnes attendues pour le modele '{model_path.name}'."
    )


def _load_model_descriptor(model_path: Path) -> _LoadedModel:
    payload = joblib.load(model_path)
    predictor = _extract_predictor(payload, model_path)
    diagnostic = str(payload.get("diagnostic") or model_path.stem)
    expected_columns = _extract_expected_columns(predictor, model_path)
    threshold = float(payload.get("Youden_threshold", 0.5) or 0.5)
    return _LoadedModel(
        model_path=model_path,
        predictor=predictor,
        diagnostic=diagnostic,
        expected_columns=expected_columns,
        threshold=threshold,
    )


def _load_model_descriptors(model_paths: Sequence[Path]) -> list[_LoadedModel]:
    loaded_models: list[_LoadedModel] = []
    seen_diagnostics: set[str] = set()
    for model_path in model_paths:
        loaded_model = _load_model_descriptor(model_path)
        if loaded_model.diagnostic in seen_diagnostics:
            raise ValueError(
                f"Diagnostic duplique detecte parmi les modeles sauvegardes: {loaded_model.diagnostic}."
            )
        seen_diagnostics.add(loaded_model.diagnostic)
        loaded_models.append(loaded_model)
    return loaded_models


def _infer_default_feature_names(loaded_models: Sequence[_LoadedModel]) -> list[str]:
    return _merge_feature_names_in_order(
        loaded_model.expected_columns for loaded_model in loaded_models
    )


def _align_features(
    input_row: pd.DataFrame,
    expected_columns: Sequence[str],
    model_name: str,
) -> pd.DataFrame:
    missing_columns = [column for column in expected_columns if column not in input_row.columns]
    if missing_columns:
        raise ValueError(
            f"Le modele '{model_name}' requiert des colonnes absentes: {', '.join(missing_columns)}."
        )
    return input_row.loc[:, list(expected_columns)].copy()


def _extract_positive_probability(predicted_proba: Any) -> float:
    proba_array = np.asarray(predicted_proba)
    if proba_array.ndim == 2 and proba_array.shape[1] >= 2:
        return float(proba_array[0, -1])
    return float(proba_array.reshape(-1)[0])


def _score_loaded_model(
    loaded_model: _LoadedModel,
    input_row: pd.DataFrame,
) -> dict[str, Any]:
    aligned_input = _align_features(
        input_row,
        loaded_model.expected_columns,
        loaded_model.diagnostic,
    )

    probability: float | None
    if hasattr(loaded_model.predictor, "predict_proba"):
        probability = _extract_positive_probability(
            loaded_model.predictor.predict_proba(aligned_input)
        )
        prediction = int(probability > loaded_model.threshold)
    else:
        probability = None
        prediction_raw = loaded_model.predictor.predict(aligned_input)
        prediction = int(np.asarray(prediction_raw).reshape(-1)[0])

    return {
        "diagnostic": loaded_model.diagnostic,
        "model_path": str(loaded_model.model_path),
        "probability": probability,
        "threshold": loaded_model.threshold,
        "prediction": prediction,
    }


def predict_all_saved_models(
    feature_values: Sequence[Any],
    feature_names: Sequence[str] | None = None,
    model_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Score a single patient row against every saved joblib model.

    Args:
        feature_values: Ordered feature values for one patient.
        feature_names: Optional ordered column names matching the transformed
            `grroh_features` columns. If omitted, they are inferred from the
            saved models.
        model_dir: Optional directory containing `*.joblib` saved models.

    Returns:
        Mapping keyed by diagnostic/model name with probability, threshold and prediction.
    """
    model_paths = _discover_model_paths(model_dir)
    loaded_models = _load_model_descriptors(model_paths)
    active_feature_names = (
        _infer_default_feature_names(loaded_models)
        if feature_names is None
        else _normalize_feature_names(feature_names)
    )
    input_row = _build_input_row(feature_values, active_feature_names)
    return {
        loaded_model.diagnostic: _score_loaded_model(loaded_model, input_row)
        for loaded_model in loaded_models
    }
