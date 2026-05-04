from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

__all__ = ["PCA", "pca_from_dataframe"]


def _validate_dataframe(name: str, value: Any) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"`{name}` doit etre un pandas.DataFrame.")
    if value.empty:
        raise ValueError(f"`{name}` ne doit pas etre vide.")
    return value.copy()


def _align_labels_to_features(
    labels: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    if len(labels) != len(features):
        raise ValueError("`labels` et `features` doivent avoir le meme nombre de lignes.")

    if labels.index.equals(features.index):
        return labels.copy()

    if labels.index.has_duplicates or features.index.has_duplicates:
        raise ValueError(
            "Les index de `labels` et `features` doivent etre uniques pour permettre un realignement."
        )

    if not labels.index.isin(features.index).all():
        raise ValueError(
            "Les index de `labels` ne correspondent pas a ceux de `features`."
        )

    return labels.loc[features.index].copy()


def _sample_points(
    coords: pd.DataFrame,
    max_points: int | None,
    random_state: int,
) -> pd.DataFrame:
    if max_points is None or max_points <= 0 or len(coords) <= max_points:
        return coords
    return coords.sample(n=max_points, random_state=random_state, replace=False)


def _add_sigma_ellipse(
    ax: plt.Axes,
    coords: pd.DataFrame,
    color: Any,
    sigma: float,
) -> None:
    if len(coords) < 3 or sigma <= 0:
        return

    covariance = np.cov(coords[["PC1", "PC2"]].to_numpy(), rowvar=False)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if np.any(eigenvalues <= 0):
        return

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2.0 * sigma * np.sqrt(eigenvalues)
    center = coords[["PC1", "PC2"]].mean().to_numpy()

    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=0.15,
        linewidth=1.5,
        zorder=1.5,
    )
    ax.add_patch(ellipse)


def _get_vivid_colors(n_colors: int) -> list[Any]:
    vivid_palette = [
        "#E41A1C",  # rouge
        "#377EB8",  # bleu
        "#4DAF4A",  # vert
        "#FF7F00",  # orange
        "#984EA3",  # violet
        "#00BFC4",  # cyan
        "#F781BF",  # rose
        "#A65628",  # brun
        "#FFD700",  # jaune
        "#1B9E77",  # vert turquoise
    ]

    if n_colors <= len(vivid_palette):
        return vivid_palette[:n_colors]

    cmap = plt.get_cmap("nipy_spectral")
    return [cmap(i) for i in np.linspace(0.05, 0.95, n_colors)]


def _normalize_color_columns(color_columns: list[str] | tuple[str, ...] | pd.Index) -> list[str]:
    if isinstance(color_columns, pd.Index):
        normalized = color_columns.tolist()
    elif isinstance(color_columns, tuple):
        normalized = list(color_columns)
    elif isinstance(color_columns, list):
        normalized = color_columns.copy()
    else:
        raise ValueError("`color_columns` doit etre une liste de noms de colonnes.")

    if not normalized:
        raise ValueError("`color_columns` ne doit pas etre vide.")
    if not all(isinstance(column, str) and column for column in normalized):
        raise ValueError("`color_columns` doit contenir uniquement des noms de colonnes non vides.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("`color_columns` ne doit pas contenir de doublons.")
    return normalized


def _validate_binary_label_columns(df: pd.DataFrame, color_columns: list[str]) -> pd.DataFrame:
    missing_columns = [column for column in color_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Colonnes de couleur introuvables dans `df`: {', '.join(missing_columns)}."
        )

    labels_df = df.loc[:, color_columns].copy()
    invalid_columns: list[str] = []

    for column in color_columns:
        series = labels_df[column]
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

        labels_df[column] = pd.to_numeric(labels_df[column], errors="coerce")

    if invalid_columns:
        raise ValueError(
            "Les colonnes de couleur doivent etre binaires/boolennes (valeurs 0/1 ou True/False): "
            + ", ".join(invalid_columns)
            + "."
        )

    return labels_df


def pca_from_dataframe(
    df: pd.DataFrame,
    color_columns: list[str],
    *,
    render_mode: str = "density",
    to_save: bool = False,
    save_dir: str | Path | None = None,
    show_ellipse: bool = False,
    max_points_per_label: int = 10,
    sigma: float = 1.0,
    random_state: int = 0,
) -> dict[str, Any]:
    """
    Lance la PCA a partir d'un DataFrame unique.

    Les colonnes choisies pour la coloration sont interpretees comme labels binaires
    et sont exclues du calcul des composantes principales.
    """
    df_validated = _validate_dataframe("df", df)
    normalized_color_columns = _normalize_color_columns(color_columns)
    labels_df = _validate_binary_label_columns(df_validated, normalized_color_columns)
    features_df = df_validated.drop(columns=normalized_color_columns)

    numeric_features = features_df.select_dtypes(include=[np.number])
    if numeric_features.shape[1] < 2:
        raise ValueError(
            "Il faut au moins deux colonnes numeriques hors colonnes de couleur pour calculer la PCA."
        )

    return PCA(
        labels=labels_df,
        features=features_df,
        to_save=to_save,
        save_dir=save_dir,
        render_mode=render_mode,
        max_points_per_label=max_points_per_label,
        show_ellipse=show_ellipse,
        sigma=sigma,
        random_state=random_state,
    )


def PCA(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    to_save: bool = False,
    save_dir: str | Path | None = None,
    render_mode: str = "density",
    max_points_per_label: int = 50,
    show_ellipse: bool = True,
    sigma: float = 1.0,
    random_state: int = 0,
) -> dict[str, Any]:
    """
    Trace une PCA 2D avec superposition des labels positifs.

    Args:
        labels: DataFrame multi-label binaire, aligne ligne a ligne avec `features`.
        features: DataFrame de variables explicatives.
        to_save: Si True, sauvegarde la figure sous `pca_labels.png`.
        save_dir: Dossier de sauvegarde si `to_save=True`. Par defaut, repertoire courant.
        render_mode: `density` pour afficher des densites, `scatter` pour afficher des points.
        max_points_per_label: Nombre maximal de points colorises affiches par label.
        show_ellipse: Si True, trace une ellipse par label.
        sigma: Taille des ellipses par label en nombre d'ecarts-types.
        random_state: Graine de tirage pour l'echantillonnage des points traces.

    Returns:
        dict contenant `figure`, `ax`, `coordinates`, `explained_variance_ratio`
        `save_path` et `sampled_points`.
    """
    features_df = _validate_dataframe("features", features)
    labels_df = _align_labels_to_features(_validate_dataframe("labels", labels), features_df)
    render_mode = render_mode.lower()
    if render_mode not in {"density", "scatter"}:
        raise ValueError("`render_mode` doit etre egal a 'density' ou 'scatter'.")

    numeric_features = features_df.select_dtypes(include=[np.number]).copy()
    if numeric_features.empty:
        raise ValueError("`features` doit contenir au moins une colonne numerique.")
    if numeric_features.shape[1] < 2:
        raise ValueError(
            "La PCA 2D requiert au moins deux colonnes numeriques dans `features`."
        )

    numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
    valid_rows = numeric_features.notna().all(axis=1)
    dropped_rows = int((~valid_rows).sum())
    if dropped_rows > 0:
        print(
            f"PCA: {dropped_rows} ligne(s) supprimee(s) a cause de NaN/Inf dans les features numeriques."
        )

    numeric_features = numeric_features.loc[valid_rows].copy()
    labels_df = labels_df.loc[numeric_features.index].copy()

    if len(numeric_features) < 2:
        raise ValueError(
            "Pas assez de lignes exploitables apres suppression des NaN/Inf pour calculer une PCA."
        )

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_features)

    pca_model = SklearnPCA(n_components=2)
    pca_coordinates = pca_model.fit_transform(scaled_features)
    explained_variance_ratio = pca_model.explained_variance_ratio_

    coordinates_df = pd.DataFrame(
        pca_coordinates,
        index=numeric_features.index,
        columns=["PC1", "PC2"],
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    legend_handles: list[Any] = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="lightgray",
            markerfacecolor="lightgray",
            markeredgecolor="none",
            markersize=8,
            alpha=0.7,
            label="Tous les patients",
        )
    ]

    ax.scatter(
        coordinates_df["PC1"],
        coordinates_df["PC2"],
        color="lightgray",
        alpha=0.18 if render_mode == "density" else 0.5,
        s=18 if render_mode == "density" else 30,
        zorder=1,
    )

    colors = _get_vivid_colors(max(len(labels_df.columns), 1))
    sampled_points: dict[str, pd.Index] = {}

    for idx, column in enumerate(labels_df.columns):
        label_series = pd.to_numeric(labels_df[column], errors="coerce").fillna(0)
        positive_mask = label_series > 0
        if not positive_mask.any():
            continue

        positive_coords = coordinates_df.loc[positive_mask]
        sampled_coords = _sample_points(
            positive_coords,
            max_points=max_points_per_label,
            random_state=random_state,
        )
        sampled_points[column] = sampled_coords.index

        if show_ellipse:
            _add_sigma_ellipse(ax, positive_coords, colors[idx], sigma=sigma)

        if render_mode == "density" and len(positive_coords) >= 5:
            sns.kdeplot(
                data=positive_coords,
                x="PC1",
                y="PC2",
                fill=True,
                levels=4,
                thresh=0.1,
                bw_adjust=1.0,
                alpha=0.22,
                color=colors[idx],
                ax=ax,
                zorder=2,
            )
            sns.kdeplot(
                data=positive_coords,
                x="PC1",
                y="PC2",
                fill=False,
                levels=4,
                thresh=0.1,
                bw_adjust=1.0,
                color=colors[idx],
                linewidths=1.2,
                ax=ax,
                zorder=2.2,
            )
            legend_label = f"{column} (densite, n={len(positive_coords)})"
        else:
            ax.scatter(
                sampled_coords["PC1"],
                sampled_coords["PC2"],
                color=colors[idx],
                alpha=0.85,
                s=55,
                edgecolors="black",
                linewidths=0.3,
                zorder=2.5,
            )
            legend_label = (
                f"{column} (n={len(sampled_coords)})"
                if render_mode == "scatter"
                else f"{column} (scatter fallback, n={len(sampled_coords)})"
            )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=colors[idx],
                marker="o",
                linestyle="-",
                markerfacecolor=colors[idx],
                markeredgecolor=colors[idx],
                markersize=7,
                linewidth=2,
                alpha=0.9,
                label=legend_label,
            )
        )

    ax.set_xlabel(f"PC1 ({explained_variance_ratio[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_variance_ratio[1] * 100:.1f}% variance)")
    title = f"PCA des features avec superposition des labels ({render_mode})"
    if render_mode == "scatter":
        title += f", max {max_points_per_label} points/label"
    if show_ellipse:
        title += f", ellipse {sigma:.1f} sigma"
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(handles=legend_handles, loc="best", frameon=True)
    fig.tight_layout()

    save_path: Path | None = None
    if to_save:
        target_dir = Path(save_dir) if save_dir is not None else Path.cwd()
        target_dir.mkdir(parents=True, exist_ok=True)
        save_path = target_dir / "pca_labels.png"
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    if "agg" not in plt.get_backend().lower():
        plt.show()

    return {
        "figure": fig,
        "ax": ax,
        "coordinates": coordinates_df,
        "explained_variance_ratio": explained_variance_ratio,
        "save_path": save_path,
        "sampled_points": sampled_points,
        "render_mode": render_mode,
    }
