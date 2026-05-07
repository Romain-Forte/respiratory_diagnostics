from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "correspondence_analysis",
    "correspondence_analysis_from_dataframe",
]


def _validate_dataframe(name: str, value: Any) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"`{name}` doit etre un pandas.DataFrame.")
    if value.empty:
        raise ValueError(f"`{name}` ne doit pas etre vide.")
    return value.copy()


def _normalize_column_names(
    column_names: list[str] | tuple[str, ...] | pd.Index,
    argument_name: str,
) -> list[str]:
    if isinstance(column_names, pd.Index):
        normalized = column_names.tolist()
    elif isinstance(column_names, tuple):
        normalized = list(column_names)
    elif isinstance(column_names, list):
        normalized = column_names.copy()
    else:
        raise ValueError(f"`{argument_name}` doit etre une liste de noms de colonnes.")

    if not normalized:
        raise ValueError(f"`{argument_name}` ne doit pas etre vide.")
    if not all(isinstance(column, str) and column for column in normalized):
        raise ValueError(
            f"`{argument_name}` doit contenir uniquement des noms de colonnes non vides."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"`{argument_name}` ne doit pas contenir de doublons.")
    return normalized


def _validate_binary_columns(
    df: pd.DataFrame,
    column_names: list[str],
    argument_name: str,
) -> pd.DataFrame:
    missing_columns = [column for column in column_names if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Colonnes introuvables dans `df` pour `{argument_name}`: "
            + ", ".join(missing_columns)
            + "."
        )

    subset_df = df.loc[:, column_names].copy()
    invalid_columns: list[str] = []

    for column in column_names:
        series = subset_df[column]
        non_null = series.dropna()
        if non_null.empty:
            invalid_columns.append(column)
            continue

        numeric_series = pd.to_numeric(non_null, errors="coerce")
        if numeric_series.isna().any():
            invalid_columns.append(column)
            continue

        unique_values = set(numeric_series.unique().tolist())
        if not unique_values.issubset({0, 1}):
            invalid_columns.append(column)
            continue

        subset_df[column] = pd.to_numeric(subset_df[column], errors="coerce").fillna(0).astype(int)

    if invalid_columns:
        raise ValueError(
            f"Les colonnes de `{argument_name}` doivent etre binaires/boolennes "
            "(valeurs 0/1 ou True/False): "
            + ", ".join(invalid_columns)
            + "."
        )

    return subset_df


def _align_columns_to_rows(
    rows: pd.DataFrame,
    columns: pd.DataFrame,
) -> pd.DataFrame:
    if len(rows) != len(columns):
        raise ValueError(
            "`diagnoses` et `underlying_conditions` doivent avoir le meme nombre de lignes."
        )

    if rows.index.equals(columns.index):
        return columns.copy()

    if rows.index.has_duplicates or columns.index.has_duplicates:
        raise ValueError(
            "Les index de `diagnoses` et `underlying_conditions` doivent etre uniques "
            "pour permettre un realignement."
        )

    if not columns.index.isin(rows.index).all():
        raise ValueError(
            "Les index de `underlying_conditions` ne correspondent pas a ceux de `diagnoses`."
        )

    return columns.loc[rows.index].copy()


def _build_contingency_table(
    diagnoses: pd.DataFrame,
    underlying_conditions: pd.DataFrame,
) -> pd.DataFrame:
    return diagnoses.T.dot(underlying_conditions)


def _pad_coordinates(
    coordinates: np.ndarray,
    prefix: str,
) -> pd.DataFrame:
    if coordinates.shape[1] == 0:
        padded = np.zeros((coordinates.shape[0], 2), dtype=float)
    elif coordinates.shape[1] == 1:
        padded = np.column_stack([coordinates[:, 0], np.zeros(coordinates.shape[0], dtype=float)])
    else:
        padded = coordinates[:, :2]

    return pd.DataFrame(padded, columns=[f"{prefix}1", f"{prefix}2"])


def correspondence_analysis(
    diagnoses: pd.DataFrame,
    underlying_conditions: pd.DataFrame,
    *,
    to_save: bool = False,
    save_dir: str | Path | None = None,
    diagnosis_color: str = "#E41A1C",
    underlying_condition_color: str = "#377EB8",
    figsize: tuple[float, float] = (10, 8),
) -> dict[str, Any]:
    """
    Build and plot a 2D correspondence analysis from two aligned binary tables.

    Args:
        diagnoses: DataFrame binaire des diagnostics.
        underlying_conditions: DataFrame binaire des terrains/sous-jacents.
        to_save: Si True, sauvegarde la figure.
        save_dir: Repertoire de sauvegarde.
        diagnosis_color: Couleur des points de diagnostic.
        underlying_condition_color: Couleur des points de terrain.
        figsize: Taille de la figure matplotlib.

    Returns:
        dict contenant la figure, les coordonnees, la table de contingence
        et la variabilite expliquee des axes.
    """
    diagnoses_df = _validate_dataframe("diagnoses", diagnoses)
    conditions_df = _align_columns_to_rows(
        diagnoses_df,
        _validate_dataframe("underlying_conditions", underlying_conditions),
    )

    contingency_table = _build_contingency_table(diagnoses_df, conditions_df)
    active_rows = contingency_table.sum(axis=1) > 0
    active_columns = contingency_table.sum(axis=0) > 0

    dropped_diagnoses = contingency_table.index[~active_rows].tolist()
    dropped_conditions = contingency_table.columns[~active_columns].tolist()

    if not active_rows.any() or not active_columns.any():
        raise ValueError(
            "La table de contingence est vide. Verifier les colonnes de diagnostic et de terrain."
        )

    contingency_table = contingency_table.loc[active_rows, active_columns]
    max_rank = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
    if max_rank < 1:
        raise ValueError(
            "La correspondence analysis requiert au moins deux diagnostics et deux terrains actifs."
        )

    total = contingency_table.to_numpy(dtype=float).sum()
    if total <= 0:
        raise ValueError("La somme de la table de contingence doit etre strictement positive.")

    relative_frequencies = contingency_table / total
    row_masses = relative_frequencies.sum(axis=1).to_numpy(dtype=float)
    column_masses = relative_frequencies.sum(axis=0).to_numpy(dtype=float)
    expected = np.outer(row_masses, column_masses)

    standardized_residuals = (
        relative_frequencies.to_numpy(dtype=float) - expected
    ) / np.sqrt(expected)

    left_singular_vectors, singular_values, right_singular_vectors_t = np.linalg.svd(
        standardized_residuals,
        full_matrices=False,
    )
    right_singular_vectors = right_singular_vectors_t.T
    explained_variance = singular_values**2
    explained_variance_ratio = explained_variance / explained_variance.sum()

    row_coordinates_raw = (
        np.diag(1.0 / np.sqrt(row_masses)) @ left_singular_vectors @ np.diag(singular_values)
    )
    column_coordinates_raw = (
        np.diag(1.0 / np.sqrt(column_masses)) @ right_singular_vectors @ np.diag(singular_values)
    )

    row_coordinates = _pad_coordinates(row_coordinates_raw, "CA")
    row_coordinates.index = contingency_table.index
    column_coordinates = _pad_coordinates(column_coordinates_raw, "CA")
    column_coordinates.index = contingency_table.columns

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color="lightgray", linewidth=1.0, zorder=1)
    ax.axvline(0, color="lightgray", linewidth=1.0, zorder=1)

    ax.scatter(
        row_coordinates["CA1"],
        row_coordinates["CA2"],
        color=diagnosis_color,
        s=80,
        alpha=0.9,
        label="Diagnosis",
        zorder=3,
    )
    ax.scatter(
        column_coordinates["CA1"],
        column_coordinates["CA2"],
        color=underlying_condition_color,
        s=80,
        alpha=0.9,
        marker="s",
        label="Underlying conditions",
        zorder=3,
    )

    for label, point in row_coordinates.iterrows():
        ax.text(point["CA1"], point["CA2"], f" {label}", color=diagnosis_color, va="center")
    for label, point in column_coordinates.iterrows():
        ax.text(
            point["CA1"],
            point["CA2"],
            f" {label}",
            color=underlying_condition_color,
            va="center",
        )

    explained_variance_ca1 = (
        explained_variance_ratio[0] * 100 if len(explained_variance_ratio) >= 1 else 0.0
    )
    explained_variance_ca2 = (
        explained_variance_ratio[1] * 100 if len(explained_variance_ratio) >= 2 else 0.0
    )
    ax.set_xlabel(f"CA1 ({explained_variance_ca1:.1f}% explained variance)")
    ax.set_ylabel(f"CA2 ({explained_variance_ca2:.1f}% explained variance)")
    ax.set_title("Correspondence analysis: diagnosis vs underlying conditions")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    save_path: Path | None = None
    if to_save:
        target_dir = Path(save_dir) if save_dir is not None else Path.cwd()
        target_dir.mkdir(parents=True, exist_ok=True)
        save_path = target_dir / "correspondence_analysis.png"
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    if "agg" not in plt.get_backend().lower():
        plt.show()

    return {
        "figure": fig,
        "ax": ax,
        "contingency_table": contingency_table,
        "row_coordinates": row_coordinates,
        "column_coordinates": column_coordinates,
        "singular_values": singular_values,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "explained_inertia": explained_variance,
        "explained_inertia_ratio": explained_variance_ratio,
        "dropped_diagnoses": dropped_diagnoses,
        "dropped_underlying_conditions": dropped_conditions,
        "save_path": save_path,
    }


def correspondence_analysis_from_dataframe(
    df: pd.DataFrame,
    diagnosis_columns: list[str] | tuple[str, ...] | pd.Index,
    underlying_condition_columns: list[str] | tuple[str, ...] | pd.Index,
    *,
    to_save: bool = False,
    save_dir: str | Path | None = None,
    diagnosis_color: str = "#E41A1C",
    underlying_condition_color: str = "#377EB8",
    figsize: tuple[float, float] = (10, 8),
) -> dict[str, Any]:
    """
    Wrapper that builds a correspondence analysis from a single dataframe.

    The specified diagnosis and underlying-condition columns must be binary.
    """
    df_validated = _validate_dataframe("df", df)
    normalized_diagnosis_columns = _normalize_column_names(
        diagnosis_columns,
        "diagnosis_columns",
    )
    normalized_condition_columns = _normalize_column_names(
        underlying_condition_columns,
        "underlying_condition_columns",
    )

    overlapping_columns = sorted(
        set(normalized_diagnosis_columns).intersection(normalized_condition_columns)
    )
    if overlapping_columns:
        raise ValueError(
            "Les colonnes de diagnostic et de terrain doivent etre distinctes: "
            + ", ".join(overlapping_columns)
            + "."
        )

    diagnoses_df = _validate_binary_columns(
        df_validated,
        normalized_diagnosis_columns,
        "diagnosis_columns",
    )
    conditions_df = _validate_binary_columns(
        df_validated,
        normalized_condition_columns,
        "underlying_condition_columns",
    )

    return correspondence_analysis(
        diagnoses=diagnoses_df,
        underlying_conditions=conditions_df,
        to_save=to_save,
        save_dir=save_dir,
        diagnosis_color=diagnosis_color,
        underlying_condition_color=underlying_condition_color,
        figsize=figsize,
    )
