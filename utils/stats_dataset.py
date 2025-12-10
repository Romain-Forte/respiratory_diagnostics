import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from itertools import combinations
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, Optional, Sequence, Tuple


def analyser_variables_binaires(df, visualisation=True, print_results=True):
    """
    Analyse un DataFrame contenant des variables binaires (0/1).
    """
    if not ((df.isin([0, 1]) | df.isna()).all().all()):
        print("Attention : certaines colonnes contiennent des valeurs autres que 0 ou 1.")

    nb_plusieurs_uns = (df.sum(axis=1) > 1).sum()
    distribution = df.sum().sort_values(ascending=False)
    distribution_pct = (distribution / len(df)) * 100

    if print_results:
        print(f"Nombre de lignes contenant plus d'un '1' : {nb_plusieurs_uns}")
        print("\nDistribution des valeurs positives (1) par colonne (% du total) :")
        print(distribution_pct.round(2))

    if visualisation:
        plt.figure(figsize=(15, 8))
        plt.bar(distribution_pct.index, distribution_pct.values)
        plt.title("Proportion de chaque diagnostic dans la BDD")
        plt.xlabel("Nom du diagnostic")
        plt.ylabel("Pourcentage des diagnostics (%)")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.show()

    return distribution_pct, nb_plusieurs_uns


def analyser_associations_binaires(df, top_n=10):
    """
    Analyse les associations entre variables binaires (0/1).
    """
    df = df.copy()
    n_total = len(df)

    mask_non_bin = df.apply(lambda col: ~col.isin([0, 1]) & col.notna()).any(axis=1)
    if mask_non_bin.any():
        print("Lignes contenant des valeurs non binaires détectées :")
        print(df[mask_non_bin])
        df = df[~mask_non_bin]
        print(f"\n{mask_non_bin.sum()} ligne(s) supprimée(s).\n")

    lignes_multiples = df[df.sum(axis=1) > 1]
    print(f"Nombre de lignes contenant plus d'un '1' : {len(lignes_multiples)}")

    compteur = Counter()
    for _, ligne in lignes_multiples.iterrows():
        colonnes_actives = ligne[ligne == 1].index.tolist()
        for r in range(2, len(colonnes_actives) + 1):
            for combo in combinations(sorted(colonnes_actives), r):
                compteur[combo] += 1

    if not compteur:
        print("Aucune association détectée.")
        return None, lignes_multiples

    assoc_df = pd.DataFrame(
        compteur.most_common(top_n),
        columns=["Association", "Frequence"],
    )
    assoc_df["Pourcentage"] = (assoc_df["Frequence"] / n_total) * 100

    print("\nAssociations les plus fréquentes (% des lignes totales) :")
    print(assoc_df.round(2))

    plt.figure(figsize=(10, 5))
    plt.barh(
        [" + ".join(a) for a in assoc_df["Association"]],
        assoc_df["Pourcentage"],
    )
    plt.xlabel("Pourcentage des lignes (%)")
    plt.ylabel("Association")
    plt.title(f"Top {top_n} associations de variables binaires")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return assoc_df, lignes_multiples


def analyze_collinearity(df, corr_thresh=0.7, figsize=(12, 8)):
    """
    Analyse la colinéarité : heatmap de corrélations + VIF.
    """
    num_df = df.select_dtypes(include=[float, int])
    if num_df.shape[1] < 2:
        raise ValueError("Pas assez de features numériques pour calculer une corrélation ou un VIF.")

    plt.figure(figsize=figsize)
    sns.heatmap(num_df.corr(), annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title("Heatmap de corrélation")
    plt.show()

    corr_matrix = num_df.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    strong_corr = corr_matrix[corr_matrix > corr_thresh].stack().sort_values(ascending=False)
    print("\nPaires de features fortement corrélées (>|{}|) :".format(corr_thresh))
    if len(strong_corr) == 0:
        print("Aucune corrélation forte détectée.")
    else:
        print(strong_corr)

    if df.isna().any().any():
        return None

    vif_df = pd.DataFrame()
    vif_df["feature"] = num_df.columns
    vif_df["VIF"] = [variance_inflation_factor(num_df.values, i) for i in range(num_df.shape[1])]
    vif_df = vif_df.sort_values(by="VIF", ascending=False)

    print("\nVIF des features :")
    print(vif_df)

    return vif_df


def compare_dg1_dataframes(
    df_reference: pd.DataFrame,
    df_candidate: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    output_path: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[str, int]], pd.DataFrame, pd.DataFrame]:
    """
    Compare deux jeux de colonnes Etiology_* identiques et construit une matrice de confusion.

    Args:
        df_reference: DataFrame contenant l'encodage historique (colonnes binaires).
        df_candidate: DataFrame contenant le nouvel encodage (ex: issu de DG1).
        columns: liste personnalisée de colonnes à comparer (défaut = ETIOLOGY_COLUMNS).
        output_path: chemin optionnel pour sauvegarder les lignes divergentes.

    Returns:
        summary: dict {colonne -> {'overlap','old_only','new_only','none'}}.
        mismatches: DataFrame des lignes ayant au moins une différence.
        confusion: matrice de confusion (index = ancien label, colonnes = nouveau label).
    """
    from utils.feature_transformer import ETIOLOGY_COLUMNS

    cols = columns or ETIOLOGY_COLUMNS
    missing_old = [col for col in cols if col not in df_reference.columns]
    missing_new = [col for col in cols if col not in df_candidate.columns]
    if missing_old:
        raise ValueError(f"Colonnes manquantes dans df_reference: {missing_old}")
    if missing_new:
        raise ValueError(f"Colonnes manquantes dans df_candidate: {missing_new}")

    old = df_reference[cols].fillna(0).astype(int)
    new = df_candidate[cols].fillna(0).astype(int)

    summary: Dict[str, Dict[str, int]] = {}
    mismatch_mask = pd.Series(False, index=old.index)
    for column in cols:
        old_true = old[column] == 1
        new_true = new[column] == 1
        summary[column] = {
            "overlap": int((old_true & new_true).sum()),
            "old_only": int((old_true & ~new_true).sum()),
            "new_only": int((~old_true & new_true).sum()),
            "none": int((~old_true & ~new_true).sum()),
        }
        mismatch_mask |= old_true != new_true

    # Confusion matrix: toutes les combinaisons ancien -> nouveau.
    labels = cols + ["<aucun>"]
    confusion = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for idx in old.index:
        old_active = [col for col in cols if old.at[idx, col] == 1] or ["<aucun>"]
        new_active = [col for col in cols if new.at[idx, col] == 1] or ["<aucun>"]
        for o in old_active:
            for n in new_active:
                confusion.loc[o, n] += 1

    mismatches = pd.concat(
        [old.loc[mismatch_mask].add_suffix("_old"), new.loc[mismatch_mask].add_suffix("_new")],
        axis=1,
    )
    if output_path is not None:
        mismatches.to_csv(output_path, index=False)

    return summary, mismatches, confusion
