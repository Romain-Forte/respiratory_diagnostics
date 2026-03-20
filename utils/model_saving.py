from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

# Typage souple pour accepter aussi bien les pipelines imblearn que sklearn.
PipelineLike = Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = _REPO_ROOT / "saved_models"


def _slugify(value: str) -> str:
    clean = "".join(char if char.isalnum() else "_" for char in (value or ""))
    clean = clean.strip("_")
    return clean or "target"


def _model_path(diagnostic: str, model_dir: Optional[str | Path], create: bool) -> Path:
    base_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    if create:
        base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{_slugify(diagnostic)}.joblib"


def save_model(
    diagnostic: str,
    pipe_train: PipelineLike,
    pipe_inference: Optional[PipelineLike] = None,
    model_dir: Optional[str | Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Path:
    """Sauvegarde un pipeline d'entrainement/inference pour un diagnostic.

    Args:
        diagnostic: Nom lisible du diagnostic (ex: 'All fungus').
        pipe_train: Pipeline complet ayant servi a l'entrainement.
        pipe_inference: Pipeline utilise pour l'inference (par defaut pipe_train).
        model_dir: Dossier racine ou stocker les modeles.
        metadata: Informations additionnelles librement definissables.
        overwrite: Remplace le fichier existant si True.

    Returns:
        Le chemin du fichier cree.
    """
    target_path = _model_path(diagnostic, model_dir, create=True)
    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f"Un modele existe deja pour '{diagnostic}' ({target_path})."
            " Utilisez overwrite=True pour le remplacer."
        )

    payload = {
        "diagnostic": diagnostic,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "pipe_train": pipe_train,
        "pipe_inference": pipe_inference or pipe_train,
        "metadata": metadata or {},
    }
    joblib.dump(payload, target_path)
    return target_path


def load_model(
    diagnostic: str,
    model_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Charge le pipeline precedemment sauvegarde pour un diagnostic donne."""
    target_path = _model_path(diagnostic, model_dir, create=False)
    if not target_path.exists():
        raise FileNotFoundError(
            f"Aucun modele sauvegarde pour '{diagnostic}'."
            f" Chemin attendu : {target_path}"
        )
    payload = joblib.load(target_path)
    if "pipe_inference" not in payload:
        payload["pipe_inference"] = payload.get("pipe_train")
    return payload
