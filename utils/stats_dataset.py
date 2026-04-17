import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from itertools import combinations
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, Optional, Sequence, Tuple


def descript_data(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    paper_format: bool = False,
    decimals: int = 2,
    non_binary_only: bool = False,
) -> pd.DataFrame:
    """
    Affiche un tableau descriptif par colonne avec la médiane,
    Q1, Q3, le pourcentage de valeurs manquantes et les effectifs.

    Args:
        df: DataFrame à résumer.
        save_path: chemin optionnel de sauvegarde (.csv, .xlsx ou .tex).
        paper_format: si True, retourne un tableau prêt pour un papier avec
            une colonne "Mediane [Q1-Q3]".
        decimals: nombre de décimales pour l'affichage et l'export.
        non_binary_only: si True, conserve uniquement les colonnes numériques
            non binaires, même en présence de valeurs manquantes.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("`df` doit être un pandas.DataFrame.")

    def _format_number(value):
        if pd.isna(value):
            return "NA"
        return f"{value:.{decimals}f}"

    def _is_binary(series: pd.Series) -> bool:
        unique_values = pd.unique(series.dropna())
        return len(unique_values) <= 2

    def _is_zero_one_binary(series: pd.Series) -> bool:
        unique_values = set(pd.unique(series.dropna()))
        return len(unique_values) > 0 and unique_values.issubset({0, 1, 0.0, 1.0})

    def _inverse_time_transform(series: pd.Series) -> pd.Series:
        numeric_series = pd.to_numeric(series, errors="coerce")
        restored = pd.Series(np.nan, index=series.index, dtype="float64")
        restored.loc[numeric_series == 0] = 0.0

        valid_mask = numeric_series.notna() & (numeric_series != 0) & (numeric_series < 1)
        restored.loc[valid_mask] = 1.0 / (1.0 - numeric_series.loc[valid_mask])
        return restored

    def _restore_original_scale(data: pd.DataFrame) -> pd.DataFrame:
        restored_df = data.copy()

        if "SOFA_scaled" in restored_df.columns:
            restored_df["SOFA_scaled"] = pd.to_numeric(
                restored_df["SOFA_scaled"], errors="coerce"
            ) * 24
            restored_df = restored_df.rename(columns={"SOFA_scaled": "SOFA_score"})

        if "Age_scaled" in restored_df.columns:
            restored_df["Age_scaled"] = pd.to_numeric(
                restored_df["Age_scaled"], errors="coerce"
            ) * 92
            restored_df = restored_df.rename(columns={"Age_scaled": "Age"})

        for col in ["Time H-ICU", "TIME SYMPTOMES-ICU", "Time  DG-ICU"]:
            if col in restored_df.columns:
                restored_df[col] = _inverse_time_transform(restored_df[col])

        return restored_df

    selected_df = df.copy()
    if non_binary_only:
        numeric_df = df.select_dtypes(include=[np.number])
        kept_columns = [
            col
            for col in numeric_df.columns
            if not _is_binary(numeric_df[col])
        ]
        selected_df = numeric_df[kept_columns]

    for col in selected_df.columns:
        series = selected_df[col]
        if pd.api.types.is_numeric_dtype(series) and _is_binary(series):
            selected_df[col] = series.astype("Float64")

    if "Hem_mal" in selected_df.columns:
        selected_df = selected_df.drop(columns="Hem_mal")
    if "Leukocytes" in selected_df.columns:
        selected_df = selected_df.drop(columns="Leukocytes")
    if "PaO2/FiO2 VALUE VALUE" in selected_df.columns:

        selected_df = selected_df.rename(columns={"PaO2/FiO2 VALUE VALUE": "PaO2/FiO2"})

    selected_df = _restore_original_scale(selected_df)

    numeric_df = selected_df.select_dtypes(include=[np.number])
    medians = numeric_df.median()
    q1 = numeric_df.quantile(0.25)
    q3 = numeric_df.quantile(0.75)
    missing_pct = selected_df.isna().mean().mul(100).reindex(selected_df.columns)
    non_missing_count = selected_df.notna().sum().reindex(selected_df.columns)
    total_patients = len(selected_df)
    positive_rate = pd.Series(np.nan, index=selected_df.columns, dtype="float64")
    positive_count = pd.Series(np.nan, index=selected_df.columns, dtype="float64")
    for col in selected_df.columns:
        series = selected_df[col]
        if pd.api.types.is_numeric_dtype(series) and _is_zero_one_binary(series):
            numeric_series = pd.to_numeric(series, errors="coerce")
            positive_rate.loc[col] = numeric_series.eq(1).mean() * 100
            positive_count.loc[col] = numeric_series.eq(1).sum()

    summary = pd.DataFrame({"colonne": selected_df.columns})
    summary["nombre_patients"] = total_patients
    summary["effectif_sans_nan"] = non_missing_count.values
    summary["mediane"] = summary["colonne"].map(medians).round(decimals)
    summary["q1"] = summary["colonne"].map(q1).round(decimals)
    summary["q3"] = summary["colonne"].map(q3).round(decimals)
    summary["pourcentage_donnees_manquantes"] = missing_pct.round(decimals).values
    summary["effectif_positif"] = summary["colonne"].map(positive_count).round(decimals)
    summary["positive_rate"] = summary["colonne"].map(positive_rate).round(decimals)
    
    output_df = summary
    if paper_format:
        output_df = pd.DataFrame(
            {
                "Variable": selected_df.columns,
                "N": [total_patients for _ in selected_df.columns],
                # "Effectif sans NA": [
                #     int(non_missing_count.get(col, 0))
                #     for col in selected_df.columns
                # ],
                "Mediane [Q1-Q3]": [
                    f"{_format_number(medians.get(col))} "
                    f"[{_format_number(q1.get(col))}-{_format_number(q3.get(col))}]"
                    for col in selected_df.columns
                ],
                "Donnees manquantes (%)": [
                    _format_number(missing_pct.get(col))
                    for col in selected_df.columns
                ],
                "Positive rate (%)": [
                    _format_number(positive_rate.get(col))
                    for col in selected_df.columns
                ],
                "Effectif positif": [
                    _format_number(positive_count.get(col))
                    for col in selected_df.columns
                ],
            }
        )
        output_df = output_df.sort_values(by = 'Variable')
    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix == ".csv":
            output_df.to_csv(save_path, index=False)
        elif save_path.suffix == ".xlsx":
            output_df.to_excel(save_path, index=False)
        elif save_path.suffix == ".tex":
            output_df.to_latex(save_path, index=False, escape=False)
        else:
            raise ValueError("Extension non supportée. Utiliser .csv, .xlsx ou .tex.")

    print(output_df)
    return output_df


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
        plt.title("Proportion of each diagnostic in the database")
        plt.xlabel("diagnostic name")
        plt.ylabel("Percentage of diagnostics (%)")
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


def plot_confusion_heatmap(
    confusion: pd.DataFrame,
    *,
    normalize: Optional[str] = "row",
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "Blues",
    annot: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Visualise la matrice de confusion produite par compare_dg1_dataframes sous forme de heatmap.

    Args:
        confusion: DataFrame carrÇ¸ (index = ancien encodage, colonnes = nouveau encodage).
        normalize: None, "row", "column" ou "all" pour normaliser les comptes.
        figsize: taille de la figure si `ax` n'est pas fourni.
        cmap: palette de couleurs matplotlib/seaborn.
        annot: affiche ou non la valeur dans chaque case.
        ax: axes matplotlib existant (optionnel).

    Returns:
        matplotlib.axes.Axes: axes contenant la heatmap.
    """
    if confusion.empty:
        raise ValueError("La matrice de confusion est vide.")

    valid_norm = {None, "row", "column", "all"}
    if normalize not in valid_norm:
        raise ValueError(f"Option normalize invalide: {normalize}. Choisir parmi {valid_norm}.")

    matrix = confusion.copy()
    cbar_label = "Nombre de patients"
    fmt = "d"

    if normalize is not None:
        matrix = matrix.astype(float)
        if normalize == "row":
            matrix = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0)
            norm_label = " (normalisation par ligne)"
        elif normalize == "column":
            matrix = matrix.div(matrix.sum(axis=0).replace(0, np.nan), axis=1)
            norm_label = " (normalisation par colonne)"
        else:  # "all"
            total = matrix.values.sum()
            matrix = matrix / total if total else matrix
            norm_label = " (normalisation globale)"
        matrix = matrix.fillna(0)
        fmt = ".0%"
        cbar_label = "Part des diagnostics"
    else:
        norm_label = ""

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        matrix,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_xlabel("Diagnostique trouvé par DG1")
    ax.set_ylabel("Diagnostique référence")
    ax.set_title(f"Matrice de confusion DG1{norm_label}")
    plt.tight_layout()
    return ax



class ScoreAliceModel:
    """
    Implémentation du score clinique "Alice".

    L'instance est appelable sur une ligne (Series) et expose également les méthodes
    `predict` et `predict_proba` pour être compatible avec les pipelines classiques.
    """

    _SCORE_MAX = 12.0

    def __call__(self, row: pd.Series):
        return self._score_row(row)

    def _score_row(self, row: pd.Series):
        score = 0
        immuno = "others"
        # 1) Immunosuppression
        if row["HSCT_BMT_Allograft"] == 1:
            immuno = "allogenic_stem_cell_transplant"
            score += 3

        if row["Hem_mal_AML"] == 1 or row["Hem_mal_ALL"] == 1:
            immuno = "acute_leukemia"
            score += 1

        # 3) Solid tumors
        if row["Solid_tumor"] == 1:
            immuno = "solid_tumors"
            score -= 2

        # 4) Other hematological malignancies
        if (
            row["Hem_mal_myeloma"] == 1
            or row["Hem_mal_CLL"] == 1
            or row["Hem_mal_CML"] == 1
        ):
            immuno = "other_hematological_malignancies"
            score += 1

        # 2) Corticostéroïdes
        corticosteroids = row["Steroids_YN"] == 1
        if corticosteroids:
            score += 1

        # 3) Symptômes > 7 jours
        symptoms_gt_7_days = int(row["TIME SYMPTOMES-ICU"] >= 6/7)
        if symptoms_gt_7_days:
            score += 1

        # 4) Neutropénie (< 0.5 G/L)
        neutropenia = row["Neutropenie"] == 1
        if neutropenia:
            score += 1

        # 5) Focal alveolar pattern
        focal_alveolar_pattern = int(
            (row["Alveolar"] == 1)
            and row["Quad_no"] == 1
        )

        if focal_alveolar_pattern:
            score += 1

        # 6) Hemoptysis
        hemoptysis = row["Hemoptysis"] == 1
        if hemoptysis:
            score += 1
        if row["Viral infection"] == 1:
            
            score +=1
        predicted_IPA = score >= 4  # ou strict ???

        return (
            {
                "immunosuppression_category": immuno,
                "corticosteroids": corticosteroids,
                "symptoms_gt_7_days": symptoms_gt_7_days,
                "neutropenia": neutropenia,
                "hemoptysis": hemoptysis,
                "focal_alveolar_pattern": focal_alveolar_pattern,
            },
            score,
            predicted_IPA,
        )

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcule les prédictions booléennes pour l'ensemble des lignes d'un DataFrame.
        """
        df = self._ensure_dataframe(df)
        preds = df.apply(lambda row: int(bool(self(row)[2])), axis=1)
        return pd.Series(preds, index=df.index, name="Score_alice")

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retourne une probabilité simplifiée: score / 12 (bornée entre 0 et 1).
        """
        df = self._ensure_dataframe(df)

        def _proba(row):
            _, score, _ = self(row)
            proba = float(score) / self._SCORE_MAX
            proba = max(0.0, min(1.0, proba))
            return proba

        proba_positive = df.apply(_proba, axis=1)
        proba_df = pd.DataFrame(
            {
                "prob_negative": 1.0 - proba_positive,
                "prob_positive": proba_positive,
            },
            index=df.index,
        )
        return proba_df

    @staticmethod
    def _ensure_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
                raise ValueError("`df` doit être un pandas.DataFrame.")
        return df


Score_alice = ScoreAliceModel()


class Scorepneumo:
    """
    Implémentation du score clinique "Pneumocystose" du papier suivant :
    Azoulay, Roux, Vincent, et al.: A Prediction Score for Pneumocystis Pneumonia
    """

    _SCORE_MAX = 11.5

    def __call__(self, row: pd.Series):
        return self._score_row(row)

    def _score_row(self, row: pd.Series):
        score = 0
        immuno = "others"
        # 1) Age
        if row["Age"] >= 50:
            score -= 1.5
            if  row["Age"] > 70:
                score -= 1
        
        # 2) lymphoproliferatif

        if row["Hem_mal_CLL"] == 1 or row["Hem_mal_ALL"] == 1 or row["Hem_mal_Non_hodgkin_lymphoma"] == 1:
            immuno = "lymphoproliferative"
            score += 2
        

        # 3) Prophylaxie non prise
        if row["Indication_prophy_pneumocystose"] == 1 and row["Prophylaxis_pneumocystis"] == 0 :
            
            score += 1 

        # 4) SYmptomes-ICU
        if row["TIME SYMPTOMES-ICU"] >= 2/3 : 
            
            score += 3

        # 5) Shock sceptique
        if row["Vasopressors"] == 1:
            score -= 1.5

        # 6) chest xray 

        if   row[ "Alveolar_xray"] == 0 :
            score += 2.5
        # 7) Pleural_eff 

        if    row[ "Pleural_eff"] == 1 :
            score -= 2
        
        predicted_Pneumo = score >= 3  # ou strict ???

        return (
            {
                "immunosuppression_category": immuno,
            },
            score,
            predicted_Pneumo,
        )

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcule les prédictions booléennes pour l'ensemble des lignes d'un DataFrame.
        """
        df = self._ensure_dataframe(df)
        preds = df.apply(lambda row: int(bool(self(row)[2])), axis=1)
        return pd.Series(preds, index=df.index, name="Score_Pneumocystose")

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retourne une probabilité simplifiée: score / 12 (bornée entre 0 et 1).
        """
        df = self._ensure_dataframe(df)

        def _proba(row):
            _, score, _ = self(row)
            proba = float(score) / self._SCORE_MAX
            proba = max(0.0, min(1.0, proba))
            return proba

        proba_positive = df.apply(_proba, axis=1)
        proba_df = pd.DataFrame(
            {
                "prob_negative": 1.0 - proba_positive,
                "prob_positive": proba_positive,
            },
            index=df.index,
        )
        return proba_df

    @staticmethod
    def _ensure_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("`df` doit être un pandas.DataFrame.")
        return df


Score_pneumo = Scorepneumo()
