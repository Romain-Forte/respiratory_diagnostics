from __future__ import annotations

from typing import Any

import pandas as pd

from utils.pca import pca_from_dataframe

try:
    import ipywidgets as widgets
except ImportError:  # pragma: no cover - depends on notebook environment
    widgets = None

__all__ = ["show_pca_widget"]


def show_pca_widget(df: pd.DataFrame) -> widgets.Widget | None:
    """
    Affiche une IHM notebook pour parametrer et lancer une PCA.
    """
    if widgets is None:
        print("ipywidgets n'est pas installe. Installez-le pour utiliser l'IHM PCA notebook.")
        return None

    if not isinstance(df, pd.DataFrame):
        raise ValueError("`df` doit etre un pandas.DataFrame.")
    if df.empty:
        raise ValueError("`df` ne doit pas etre vide.")

    color_columns = widgets.SelectMultiple(
        options=tuple(df.columns.tolist()),
        value=(),
        description="Couleurs",
        rows=min(max(len(df.columns), 8), 18),
        layout=widgets.Layout(width="100%", height="260px"),
    )
    render_mode = widgets.Dropdown(
        options=("density", "scatter"),
        value="density",
        description="Mode",
        layout=widgets.Layout(width="320px"),
    )
    to_save = widgets.Checkbox(
        value=False,
        description="Sauvegarder",
        indent=False,
        layout=widgets.Layout(width="160px"),
    )
    save_dir = widgets.Text(
        value="",
        description="save_dir",
        placeholder="Laisser vide pour le repertoire courant",
        disabled=True,
        layout=widgets.Layout(width="100%"),
    )
    show_ellipse = widgets.Checkbox(
        value=False,
        description="Ellipse",
        indent=False,
        layout=widgets.Layout(width="160px"),
    )
    max_points_per_label = widgets.BoundedIntText(
        value=10,
        min=1,
        max=max(len(df), 10),
        step=1,
        description="Max points",
        layout=widgets.Layout(width="220px"),
    )
    run_button = widgets.Button(
        description="Afficher la PCA",
        button_style="primary",
        icon="line-chart",
    )
    output = widgets.Output()
    help_text = widgets.HTML(
        value=(
            "<b>PCA notebook</b><br>"
            "Selectionnez une ou plusieurs colonnes binaires pour la couleur. "
            "Ces colonnes seront exclues du calcul de la PCA."
        )
    )

    def _toggle_save_dir(change: dict[str, Any]) -> None:
        save_dir.disabled = not bool(change["new"])

    to_save.observe(_toggle_save_dir, names="value")

    def _run_pca(_: widgets.Button) -> None:
        with output:
            output.clear_output(wait=True)
            selected_columns = list(color_columns.value)

            if not selected_columns:
                print("Selectionnez au moins une colonne de couleur.")
                return

            effective_save_dir = save_dir.value.strip() or None
            if not to_save.value:
                effective_save_dir = None

            try:
                pca_from_dataframe(
                    df=df,
                    color_columns=selected_columns,
                    render_mode=render_mode.value,
                    to_save=to_save.value,
                    save_dir=effective_save_dir,
                    show_ellipse=show_ellipse.value,
                    max_points_per_label=max_points_per_label.value,
                )
            except Exception as exc:
                print(f"Erreur PCA: {exc}")

    run_button.on_click(_run_pca)

    controls = widgets.VBox(
        [
            help_text,
            color_columns,
            widgets.HBox([render_mode, max_points_per_label]),
            widgets.HBox([to_save, show_ellipse]),
            save_dir,
            run_button,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    container = widgets.VBox([controls, output], layout=widgets.Layout(width="100%", gap="12px"))

    return container
