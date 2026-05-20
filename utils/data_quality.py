import math
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

TRUE_LIKE = {"1", "true", "vrai", "yes", "oui", "y", "t"}
FALSE_LIKE = {"0", "false", "faux", "no", "non", "n", "f"}


def _normalize_binary_value(value):
    if value is None:
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        if value in (0, 1):
            return int(value)
        return value
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return np.nan
        if value in (0.0, 1.0):
            return int(value)
        return value
    text = str(value).strip()
    if not text:
        return np.nan
    lowered = text.lower()
    if lowered in TRUE_LIKE:
        return 1
    if lowered in FALSE_LIKE:
        return 0
    return text


def _print_binary_occurrence_diff(
    col_name,
    serie_left,
    serie_right,
    total_left,
    total_right,
    min_diff_pct: float,
):
    left_clean = serie_left.apply(_normalize_binary_value).dropna()
    right_clean = serie_right.apply(_normalize_binary_value).dropna()
    combined = pd.concat([left_clean, right_clean], ignore_index=True).dropna()
    if combined.empty:
        return False, False, []
    unique_vals = pd.unique(combined)
    if len(unique_vals) > 2:
        return False, False, []

    stats = []
    should_print = False
    for val in sorted(unique_vals, key=lambda v: str(v)):
        left_count = left_clean.eq(val).sum()
        right_count = right_clean.eq(val).sum()
        left_pct = (left_count / total_left * 100) if total_left else 0.0
        right_pct = (right_count / total_right * 100) if total_right else 0.0
        diff = left_pct - right_pct
        stats.append((val, left_count, right_count, left_pct, right_pct, diff))
        if abs(diff) >= min_diff_pct:
            should_print = True

    if should_print:
        print(f"[{col_name}] difference d'occurrence >= {min_diff_pct} pts")
        for val, left_count, right_count, left_pct, right_pct, diff in stats:
            highest_pct = max(left_pct, right_pct)
            variation_pct = (abs(diff) / highest_pct * 100) if highest_pct > 0 else 0.0
            print(
                f"    valeur={val!r}: {left_count}/{total_left} ({left_pct:.2f}%) "
                f"vs {right_count}/{total_right} ({right_pct:.2f}%) -> diff {diff:+.2f} pts "
                f"({variation_pct:.1f}% de la valeur max {highest_pct:.2f}%)"
            )

    return True, should_print, stats

def analyser_nan(df: pd.DataFrame, top_n: int = 10, plot: bool = True) -> dict:
    """
    Analyse les valeurs manquantes (NaN) dans un DataFrame pandas.

    Parameters
    ----------
    df : pd.DataFrame
        Le DataFrame à analyser.
    top_n : int, optional
        Nombre d'éléments à afficher dans les tableaux triés (par défaut 10).
    plot : bool, optional
        Si True, affiche les graphiques de répartition des NaN (par défaut True).

    Returns
    -------
    dict
        Dictionnaire contenant :
        - 'colonnes' : DataFrame des NaN par colonne (nombre + %)
        - 'lignes'   : DataFrame des NaN par ligne (nombre + %)
        - 'resume'   : résumé global (nb total, % global)
    """

    # 🔹 Colonnes : nombre et pourcentage de NaN
    nan_colonnes = df.isna().sum().sort_values(ascending=False)
    pct_colonnes = (df.isna().mean() * 100).sort_values(ascending=False)
    stats_colonnes = pd.DataFrame({
        "nb_nan": nan_colonnes,
        "pct_nan": pct_colonnes
    })

    # 🔹 Lignes : nombre et pourcentage de NaN
    nan_lignes = df.isna().sum(axis=1)
    pct_lignes = df.isna().mean(axis=1) * 100
    stats_lignes = pd.DataFrame({
        "nb_nan_ligne": nan_lignes,
        "pct_nan_ligne": pct_lignes
    }).sort_values("nb_nan_ligne", ascending=False)

    # 🔹 Résumé global
    resume = {
        "total_nan": int(df.isna().sum().sum()),
        "pct_total_nan": round(df.isna().mean().mean() * 100, 2)
    }

    # 🔹 Graphiques
    if plot:
        plt.figure(figsize=(8, 6))
        stats_colonnes["pct_nan"].head(top_n).plot(kind="barh", title=f"Top {top_n} colonnes avec le plus de NaN (%)")
        plt.xlabel("% de NaN")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 4))
        stats_lignes["pct_nan_ligne"].head(top_n).plot(kind="bar", title=f"Top {top_n} lignes avec le plus de NaN (%)")
        plt.ylabel("% de NaN par ligne")
        plt.tight_layout()
        plt.show()

    return {
        "colonnes": stats_colonnes,
        "lignes": stats_lignes,
        "resume": resume
    }
def nettoyer_lignes_vides(df):
    """
    Remplace tous les NaN par 0, supprime les lignes ne contenant que des 0.
    Renvoie le DataFrame nettoyé et la liste des index des lignes supprimées.
    """
    # Remplacer les NaN par 0
    df_rempli = df.fillna(0)

    # Identifier les lignes contenant uniquement des 0
    index_a_drop = df_rempli.index[(df_rempli == 0).all(axis=1)].tolist()

    # Supprimer ces lignes
    df_nettoye = df_rempli.drop(index=index_a_drop)
    nb_vides = len(index_a_drop) 
    if nb_vides > 0:
        print(f"Nombre de lignes totalement vides : {nb_vides}")

    return df_nettoye, index_a_drop


def extraire_contenu_central(chaine):
    parties = str(chaine).split('_')
    if len(parties) < 3:
        return chaine  # garde le nom original si le format ne correspond pas
    return ' '.join(parties[1:-1])  # partie centrale entre les underscores

def nettoyer_colonnes(df):
    """
    Nettoie les noms de colonnes d'un DataFrame en extrayant
    la partie centrale des chaînes de type :
    Etiology_Cardiogenic pulmonary oedema_Definitive diagnosis
    → devient : Cardiogenic pulmonary oedema
    """

    # Application à toutes les colonnes
    df = df.copy()
    df.columns = [extraire_contenu_central(col) for col in df.columns]
    df.columns = [col.replace("(specify below)", "") for col in df.columns]

    # Harmonize specific label wording
    rename_map = {
        "Other causes ": "Other non infectious causes",
        "Drug related": "Drug toxicity related",
    }
    df = df.rename(columns=rename_map)
    return df


def _imputer_nan_par_mice(
    df: pd.DataFrame,
    target_columns=None,
    predictor_columns=None,
    random_state: int = 0,
    max_iter: int = 10,
) -> pd.DataFrame:
    """
    Impute les NaN de colonnes numeriques avec IterativeImputer (MICE).

    `target_columns` definit les colonnes a mettre a jour apres imputation.
    `predictor_columns` definit les colonnes utilisees comme predicteurs.
    Les colonnes non numeriques ou entierement vides sont ignorees.
    """

    df = df.copy()
    target_columns = list(target_columns) if target_columns is not None else list(df.columns)
    predictor_columns = (
        list(predictor_columns) if predictor_columns is not None else list(df.columns)
    )

    missing_targets = [col for col in target_columns if col not in df.columns]
    for col in missing_targets:
        print(f"Colonne '{col}' absente du DataFrame pour MICE, ignoree.")
    target_columns = [col for col in target_columns if col in df.columns]

    missing_predictors = [col for col in predictor_columns if col not in df.columns]
    if missing_predictors:
        raise ValueError(
            f"Colonnes predictrices absentes pour MICE: {missing_predictors}"
        )

    numeric_predictors = {}
    ignored_predictors = []
    for col in predictor_columns:
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        if numeric_series.notna().sum() == 0:
            ignored_predictors.append(col)
            continue
        numeric_predictors[col] = numeric_series

    if not numeric_predictors:
        print("Aucune colonne numerique exploitable pour l'imputation MICE.")
        return df

    df_numeric = pd.DataFrame(numeric_predictors, index=df.index)
    eligible_targets = [col for col in target_columns if col in df_numeric.columns]

    if not eligible_targets:
        print("Aucune colonne cible exploitable pour l'imputation MICE.")
        return df

    if ignored_predictors:
        print(
            "Colonnes ignorees pour MICE (non numeriques ou entierement vides): "
            + ", ".join(map(str, ignored_predictors))
        )

    nb_nan_avant = int(df_numeric[eligible_targets].isna().sum().sum())
    if nb_nan_avant == 0:
        print("Aucun NaN a imputer sur les colonnes cible MICE.")
        return df

    imputer = IterativeImputer(
        random_state=random_state,
        max_iter=max_iter,
        sample_posterior=False,
    )
    imputed_array = imputer.fit_transform(df_numeric)
    df_imputed = pd.DataFrame(imputed_array, columns=df_numeric.columns, index=df.index)
    df.loc[:, eligible_targets] = df_imputed[eligible_targets]

    print(
        f"MICE applique sur {len(eligible_targets)} colonne(s), "
        f"{nb_nan_avant} NaN imputes."
    )
    return df


def nettoyer_nan_par_colonne(
    df,
    strategies=None,
    use_mice: bool = False,
    mice_columns=None,
    mice_random_state: int = 42,
    mice_max_iter: int = 10,
):
    """
    Remplace les NaN colonne par colonne selon la stratégie choisie.

    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame à nettoyer.
    strategies : dict, optional
        Dictionnaire {colonne: méthode} où la méthode peut être :
          - "median" : remplace par la médiane de la colonne
          - "zero"   : remplace par 0
          - "mice"   : impute la colonne avec MICE en s'appuyant sur les
            autres colonnes numériques du DataFrame
          - une valeur numérique ou textuelle spécifique (ex: "inconnu", 99, etc.)
    use_mice : bool, optional
        Si True, applique MICE sur tout le DataFrame numérique, ou sur
        `mice_columns` si fourni, avant les stratégies colonne par colonne.
    mice_columns : list, optional
        Sous-ensemble de colonnes à mettre à jour par MICE.
    mice_random_state : int, optional
        Graine passée à IterativeImputer.
    mice_max_iter : int, optional
        Nombre maximal d'itérations pour MICE.

    Retour
    ------
    df_nettoye : pd.DataFrame
        Le DataFrame après nettoyage.
    """

    df = df.copy()
    strategies = dict(strategies or {})

    mice_strategy_columns = [
        col for col, methode in strategies.items() if methode == "mice"
    ]
    if use_mice or mice_columns is not None or mice_strategy_columns:
        mice_target_columns = []
        if use_mice and mice_columns is None:
            mice_target_columns.extend(df.columns.tolist())
        if mice_columns is not None:
            mice_target_columns.extend(list(mice_columns))
        mice_target_columns.extend(mice_strategy_columns)

        seen = set()
        mice_target_columns = [
            col for col in mice_target_columns if not (col in seen or seen.add(col))
        ]

        df = _imputer_nan_par_mice(
            df,
            target_columns=mice_target_columns,
            predictor_columns=df.columns.tolist(),
            random_state=mice_random_state,
            max_iter=mice_max_iter,
        )

    for col, methode in strategies.items():
        if col not in df.columns:
            print(f"[WARN] Colonne '{col}' absente du DataFrame, ignoree.")
            continue
        # 🔹 Tenter conversion numérique si c’est une stratégie numérique
        if methode == "mice":
            continue
        if methode in ["median", "zero"] or isinstance(methode, (int, float)):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if methode == "median":
            if pd.api.types.is_numeric_dtype(df[col]):
                mediane = df[col].median()
                df[col] = df[col].fillna(mediane)
                print(f"[INFO] {col} -> NaN remplaces par la mediane ({mediane})")
            else:
                print(f"[WARN] {col} n'est pas numerique -> mediane impossible, non modifiee.")

        elif methode == "zero":
            df[col] = df[col].fillna(0)
            # print(f"[INFO] {col} -> NaN remplaces par 0")
        
        elif methode == "str":
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = df[col].fillna(methode)
            print(f"[INFO] {col} -> NaN remplaces par '{methode}'")

    nb_nan_restants = df.isna().sum().sum()
    print(f"\n[OK] Nettoyage termine. NaN restants : {nb_nan_restants}")
    return df

def fusionner_labels(df: pd.DataFrame, mapping: dict, mode: str = "max") -> pd.DataFrame:
    """
    Fusionne les colonnes de labels selon un mapping donné.

    Paramètres
    ----------
    df : DataFrame contenant les labels (0/1 ou numériques)
    mapping : dict {nouvelle_colonne: [anciennes_colonnes]}
    mode : "max" (par défaut), "sum" ou "mean" pour la méthode de fusion

    Retour
    ------
    df_fusionné : nouveau DataFrame avec les colonnes fusionnées
    """
    df = df.copy()
    df_new = pd.DataFrame(index=df.index)

    for new_col, old_cols in mapping.items():
        if mode == "max":
            df_new[new_col] = df[old_cols].max(axis=1)
        elif mode == "sum":
            df_new[new_col] = df[old_cols].sum(axis=1)
        elif mode == "mean":
            df_new[new_col] = df[old_cols].mean(axis=1)
        else:
            raise ValueError("Mode inconnu : choisir 'max', 'sum' ou 'mean'")

    return df_new


def convert_types(df):
    df = df.copy()
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                if df[col].isin(['True', 'False', 'true', 'false']).all():
                    df[col] = df[col].map(lambda x: str(x).lower() == 'true')
    return df

def compare_columns(df1, df2):
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    only_in_df1 = cols1 - cols2
    only_in_df2 = cols2 - cols1

    print("Colonnes uniquement dans df1 :")
    print(only_in_df1)

    print("\nColonnes uniquement dans df2 :")
    print(only_in_df2)


def plot_column_histograms(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    columns=None,
    bins: int = 30,
    cols_per_row: int = 3,
    figsize_per_col=(4, 3),
    density: bool = False,
    label_left: str = "dataset_1",
    label_right: str = "dataset_2",
    alpha: float = 0.6,
    suptitle: Optional[str] = None,
    min_diff_pct: float = 5.0,
):
    """
    Compare la distribution de chaque colonne numerique via histogrammes superposes.

    Parameters
    ----------
    df_left : pd.DataFrame
        Premier jeu de donnees.
    df_right : pd.DataFrame
        Second jeu de donnees (meme schema attendu).
    columns : list, optional
        Sous-ensemble de colonnes a visualiser. Par defaut toutes les colonnes communes.
    bins : int, optional
        Nombre de bacs de l'histogramme (defaut 30).
    cols_per_row : int, optional
        Nombre de sous-graphiques par ligne (defaut 3).
    figsize_per_col : tuple, optional
        Taille (largeur, hauteur) d'un sous-graphe en pouces (defaut (4, 3)).
    density : bool, optional
        Si False (defaut), affiche les pourcentages de lignes par classe d'histogramme.
        Si True, trace les densites (aire = 1) via matplotlib.
    label_left / label_right : str, optional
        Etiquettes de legende pour les deux jeux de donnees.
    alpha : float, optional
        Transparence des histogrammes (defaut 0.6).
    suptitle : str, optional
        Titre global de la figure.
    min_diff_pct : float, optional
        Seuil minimal (en points de pourcentage) pour afficher les differences
        d'occurrence des variables binaires dans la console.

    Returns
    -------
    (fig, axes) : tuple
        Figure matplotlib et matrice d'axes generes.

    Notes
    -----
    - Les hauteurs des histogrammes (mode par defaut) representent le pourcentage
      de lignes du DataFrame complet.
    - Les colonnes contenant au plus deux valeurs distinctes (apres normalisation
      des booleens usuels) sont exclues du graphique. Les differences d'occurrence
      au-dessus de `min_diff_pct` sont detaillees dans la console; les deltas plus
      faibles declenchent un court rappel signalant que la colonne a ete ignoree.
    """
    df_left = df_left.copy()
    df_right = df_right.copy()
    if columns is None:
        columns = list(df_left.columns)
    else:
        columns = list(columns)

    if not columns:
        raise ValueError("Aucune colonne disponible pour la comparaison.")

    missing_left = [col for col in columns if col not in df_left.columns]
    missing_right = [col for col in columns if col not in df_right.columns]
    if missing_left or missing_right:
        raise ValueError(
            f"Colonnes absentes. df_left manques: {missing_left}, df_right manques: {missing_right}"
        )

    total_rows_left = len(df_left)
    total_rows_right = len(df_right)

    numeric_columns = []
    for col in columns:
        is_binary, printed, _ = _print_binary_occurrence_diff(
            col,
            df_left[col],
            df_right[col],
            total_rows_left,
            total_rows_right,
            min_diff_pct=min_diff_pct,
        )
        if "time" in col.lower():
            df_right[col] =  np.maximum(np.minimum(1 / (1 - df_right[col]),50),0)
            df_left[col] =  np.maximum( np.minimum(1 / (1 - df_left[col] ),50),0)
        if 'age' in col.lower():
            df_right[col] = 94 *  df_right[col]
            df_left[col] =  94 * df_left[col]
        if is_binary:
            if not printed:
                print(f"[{col}] variable binaire ignoree (|Delta| < {min_diff_pct:.1f} pts)")
            continue
        numeric_columns.append(col)

    if not numeric_columns:
        raise ValueError("Toutes les colonnes selectionnees sont binaires; rien a tracer.")

    n_cols = len(numeric_columns)
    rows = math.ceil(n_cols / cols_per_row)
    fig_width = max(2, cols_per_row * figsize_per_col[0])
    fig_height = max(2, rows * figsize_per_col[1])
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.ravel()

    for idx, col in enumerate(numeric_columns):
        ax = axes_flat[idx]

        serie_left = pd.to_numeric(df_left[col], errors="coerce").dropna().astype(float)
        serie_right = pd.to_numeric(df_right[col], errors="coerce").dropna().astype(float)

        plotted = False
        if not serie_left.empty:
            weights_left = None
            if not density and total_rows_left > 0:
                weights_left = np.ones(len(serie_left), dtype=float) / total_rows_left * 100
            ax.hist(
                serie_left,
                bins=bins,
                alpha=alpha,
                density=density,
                weights=weights_left,
                label=label_left,
                color="tab:blue",
            )
            plotted = True
        if not serie_right.empty:
            weights_right = None
            if not density and total_rows_right > 0:
                weights_right = np.ones(len(serie_right), dtype=float) / total_rows_right * 100
            ax.hist(
                serie_right,
                bins=bins,
                alpha=alpha,
                density=density,
                weights=weights_right,
                label=label_right,
                color="tab:orange",
            )
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, "Pas de donnees\nnumeriques", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_title(str(col))
        if plotted:
            if not density:
                ax.set_ylabel("% des lignes")
            ax.legend(loc="best")

    for ax in axes_flat[n_cols:]:
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()

    return fig, axes
