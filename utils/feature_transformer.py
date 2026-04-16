"""
Feature transformation utilities for the user's specified clinical dataframe.

This module exposes a single entry point:
    transform_features(df: pandas.DataFrame) -> pandas.DataFrame

It applies the following assumptions (written in FR to match the user's notes):
- Sex                         -> encodage binaire (M/H=1, F=0, sinon NaN)
- Age                         -> min-max scaling vers [0,1] + terme quadratique (Age_scaled_sq)
- Time H-ICU                  ->
- TIME SYMPTOMES-ICU          ->
- Location_before_ICU         -> mapping dictionnaire commun (domicile/etablissement/urgence/autre)
- Hem_mal                     -> mapping dictionnaire (oui/non/catégories simples)
- Dis_status HEM              -> mapping dichotomique (actif/rémission/inconnu)
- HSCT_BMT                    -> encode allo=2, auto=1, none/No=0
- GvHD                        -> combine avec allogreft pour faire la variable rejet_allograft
- Sys_dis, Solid_tumor, Organ_transpl,
  Drug_induced, Chemotherapy,
  Ibr_Flu_Met, Immuno_drugs,
  Tar_ther, Immunotherapy,
  Carttcells, Steroids_YN,
  Prophylaxis_* ,Vaccins      -> nettoyage guillemets + encodage binaire souple (oui/non)
- SOFA_score                  -> numeric, clip [0,24], scale [0,1] en SOFA_scaled
- Resp_rate (+ Intubation/SpO2 si dispo)
                              -> score de sévérité 0..3 (voir fonction _resp_severity)
- Sp02                        -> lineariser
- Temp                        -> coercition °C ; si >45 et <120, on suppose Fahrenheit -> conversion, clip [30,43]
- Neutrophils                 -> combine avec Leukocytes pour une variable binaire neutropénie <500
- Leukocytes                  -> combine avec neutrophile pour une variable binaire neutropénie <1k
Le module est robuste aux colonnes manquantes : elles sont ignorées.
"""

import re
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

ETIOLOGY_COLUMNS = [
    "Etiology_Bacterial infection_Definitive diagnosis",
    "Etiology_Viral infection_Definitive diagnosis",
    "Etiology_Invasive pulmonary aspergillosis_Definitive diagnosis",
    "Etiology_Pneumocystis jirovecii infection_Definitive diagnosis",
    "Etiology_Mucorales_Definitive diagnosis",
    "Etiology_Other fungal (specify below)_Definitive diagnosis",
    "Etiology_Other infection (specify below)_Definitive diagnosis",
    "Etiology_Cardiogenic pulmonary oedema_Definitive diagnosis",
    "Etiology_Drug related_Definitive diagnosis",
    "Etiology_Disease-related infiltrates_Definitive diagnosis",
    "Etiology_Transfusion-related acute lung injury_Definitive diagnosis",
    "Etiology_Other causes (specify below)_Definitive diagnosis",
    "Etiology_Undetermined cause_Definitive diagnosis",
]

ETIOLOGY_MAPPING: Dict[str, set] = {
    "Etiology_Bacterial infection_Definitive diagnosis": {
        "BACTERIAL",
        "LEGIONELLA",
        "CHLAMYDIA",
        "ACTINOMYCES",
        "NOCARDIA",
        "TUBERCULOSIS",
    },
    "Etiology_Viral infection_Definitive diagnosis": {
        "COVID",
        "FLU",
        "VIRAL",
        "RHINOVIRUS",
        "VRS",
        "PIV",
        "CMV",
        "METAPNEUMOVIRUS",
        "VZV",
        "HSV",
    },
    "Etiology_Invasive pulmonary aspergillosis_Definitive diagnosis": {"IPA"},
    "Etiology_Pneumocystis jirovecii infection_Definitive diagnosis": {"PJP"},
    "Etiology_Mucorales_Definitive diagnosis": {"MUCORALES"},
    "Etiology_Other fungal (specify below)_Definitive diagnosis": {
        "CANDIDEMIA",
        "IFI",
        "FUSARIUM",
        "GEOTRICHUM",
        "CRYPTOCOCCUS",
        "TRICHOSPORON",
        "HISTOPLASMOSIS",
        "COCCIDIOIDOSE",
        "TOXOPLASMOSIS",
    },
    "Etiology_Other infection (specify below)_Definitive diagnosis": {
        "OTHER",
        "ASPIRATION",
    },
    "Etiology_Cardiogenic pulmonary oedema_Definitive diagnosis": {"CPO"},
    "Etiology_Drug related_Definitive diagnosis": {"DRPT"},
    "Etiology_Disease-related infiltrates_Definitive diagnosis": {"DISEASE"},
    "Etiology_Transfusion-related acute lung injury_Definitive diagnosis": set(),
    "Etiology_Other causes (specify below)_Definitive diagnosis": set(),
    "Etiology_Undetermined cause_Definitive diagnosis": {"UNDETERMINED", "EMPTY EMPTY"},
}

def contains_keywords(text, keywords):
    if pd.isna(text):
        return pd.NA
    text = str(text).lower()
    return any(keyword in text for keyword in keywords)


def _comparison_nullable(series: pd.Series, value: Any, op) -> pd.Series:
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    mask = series.notna()
    if mask.any():
        result.loc[mask] = op(series.loc[mask], value)
    return result


def _eq_nullable(series: pd.Series, value: Any) -> pd.Series:
    return _comparison_nullable(series, value, lambda s, v: s == v)


def _ge_nullable(series: pd.Series, value: Any) -> pd.Series:
    return _comparison_nullable(series, value, lambda s, v: s >= v)


def _gt_nullable(series: pd.Series, value: Any) -> pd.Series:
    return _comparison_nullable(series, value, lambda s, v: s > v)


def _lt_nullable(series: pd.Series, value: Any) -> pd.Series:
    return _comparison_nullable(series, value, lambda s, v: s < v)


def _nullable_or(*series_list: pd.Series) -> pd.Series:
    result = series_list[0].astype("boolean")
    for series in series_list[1:]:
        result = result | series.astype("boolean")
    return result


def _nullable_and(*series_list: pd.Series) -> pd.Series:
    result = series_list[0].astype("boolean")
    for series in series_list[1:]:
        result = result & series.astype("boolean")
    return result


def _bool_to_float(series: pd.Series) -> pd.Series:
    series = series.astype("boolean")
    result = pd.Series(pd.NA, index=series.index, dtype="Float64")
    mask = series.notna()
    if mask.any():
        result.loc[mask] = series.loc[mask].map({True: 1.0, False: 0.0}).to_numpy()
    return result


def _str_contains_nullable(series: pd.Series, pattern: str) -> pd.Series:
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    mask = series.notna()
    if mask.any():
        result.loc[mask] = series.loc[mask].astype(str).str.contains(pattern, case=False, na=False)
    return result


def _coerce_numeric_frame(df: pd.DataFrame, columns) -> pd.DataFrame:
    coerced = df.copy()
    for col in columns:
        if col in coerced.columns:
            coerced[col] = pd.to_numeric(coerced[col], errors="coerce")
    return coerced


def _rowwise_max_numeric(df: pd.DataFrame, columns) -> pd.Series:
    numeric_subset = _coerce_numeric_frame(df[columns], columns)
    return numeric_subset.max(axis=1)

def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cols_time = [ "Time H-ICU",
    "TIME SYMPTOMES-ICU",
    "Time  DG-ICU"]
    for col in cols_time:
        if col in df.columns:
            time_values = pd.to_numeric(df[col], errors="coerce")
            df[col] = np.where(
                    time_values == 0,
                    0.0,
                    1 - 1 / time_values
                )
    #Clip GvHD 
    if "GvHD" in df.columns:
        df["GvHD"] = pd.to_numeric(df["GvHD"], errors="coerce")
        df.loc[df["GvHD"] >= 1,"GvHD"] = 1 
    
    # Age -> scale [0,1] + square
    if "Age" in df.columns:
        age_scaled, vmin, vmax = _scale_minmax(df["Age"])
        df["Age_scaled"] = age_scaled
        df = df.drop(columns=["Age"])
        # df["Age_scaled_sq"] = age_scaled ** 2
    if "Charlson_index" in df.columns:
        df["Charlson_index"] = pd.to_numeric(df["Charlson_index"], errors="coerce").clip(upper=22)

    if "Location_before_ICU" in df.columns:
        # Mapping numérique → catégorie textuelle

        mapping = {
            0: "ED",
            1: "Ward",
            2: "Other ICU",
            3: "Other hospital",
            4: "Other"
        }

        df = apply_mapping(df, "Location_before_ICU", mapping)

    if "Hem_mal" in df.columns:
        mapping = {
            1: "AML",
            2: "ALL",
            3: "Non hodgkin lymphoma",
            4: "myeloma",
            5: "hodgkin lymphoma",
            6: "CLL",
            7 : "CML",
            8: "MDS",
            9:"other"

        }

        df = apply_mapping(df, "Hem_mal", mapping)

    if "Dis_status HEM" in df.columns:

        # mapping = {
        #     1: "minus 1 month",
        #     2: "first line",
        #     3: "more than 1 line",
        #     4: "remission",
        #     5: "uncontrolled",
        #     6: "palliative"

        # }
        # df = apply_mapping(df, "Dis_status HEM", mapping)
        dis_status = pd.to_numeric(df["Dis_status HEM"], errors="coerce")
        df["Disease_status_inaugural"] = _bool_to_float(
            _nullable_or(_eq_nullable(dis_status, 1), _eq_nullable(dis_status, 2))
        )
        df["Disease_status_remission"] = _bool_to_float(_eq_nullable(dis_status, 4))
        df["Disease_status_evolutive"] = _bool_to_float(
            _nullable_or(_eq_nullable(dis_status, 3), _ge_nullable(dis_status, 5))
        )
        df = df.drop(columns = ["Dis_status HEM"])
        
    if "Alveolar_xray" in df.columns:
        mapping = {
            1: "Focal",
            2: "Diffuse",
        }
        df = apply_mapping(df, "Alveolar_xray", mapping)
    if "Interst_xray" in df.columns:
        mapping = {
            1: "Focal",
            2: "Diffuse",
        }
        df = apply_mapping(df, "Interst_xray", mapping)
    if "Alveolar_cons" in df.columns:
        mapping = {
            1: "Focal",
            2: "Diffuse",
        }
        df = apply_mapping(df, "Alveolar_cons", mapping)
    if "Ground_glass_op" in df.columns:
        mapping = {
            1: "Focal",
            2: "Diffuse",
        }
        df = apply_mapping(df, "Ground_glass_op", mapping)


    if "HSCT_BMT" in df.columns:
        mapping = {
            1: "Autograft",
            2: "Allograft"
        }

        # if "GvHD" in df.columns:
        #     df["GvHD"] = pd.to_numeric(df["GvHD"], errors="coerce")
        #     df["rejet_allograft"] = (df["GvHD"] == 1) & (df["HSCT_BMT"] == 2)
        #     df = df.drop(columns=["GvHD"])
        df = apply_mapping(df, "HSCT_BMT", mapping)

    cols_nodules = [
    "CT_nodules#Centrolobular",
    "CT_nodules#Peribronchovascular",
    "CT_nodules#Pleural",
    "Nodules"]

    if  all(c in df.columns for c in cols_nodules):
        df["Nodules_any"] = _rowwise_max_numeric(df, cols_nodules)
        df = df.drop(columns=cols_nodules)

    cols_opacity = [
    "Ground_glass_op_Focal",
    "Ground_glass_op_Diffuse",
    "Crazy_paving",
    'Interst_xray_Focal', 
    'Interst_xray_Diffuse']

    if  all(c in df.columns for c in cols_opacity):
        df["GGO"] = _rowwise_max_numeric(df, cols_opacity)
        df = df.drop(columns=cols_opacity)
    
    if "Quad_no" in df.columns:
        df["Quad_no"] = pd.to_numeric(df["Quad_no"], errors="coerce").clip(upper=4.0)
    # SOFA
    if "SOFA_score" in df.columns:
        sofa = pd.to_numeric(df["SOFA_score"], errors="coerce")
        sofa = sofa.clip(lower=0, upper=24)
        df["SOFA_score"] = sofa
        df = df.drop(columns=["SOFA_score"])
        df["SOFA_scaled"] = sofa / 24.0

    # Resp rate severity
    if "SpO2" in df.columns:
       df["SaO2"] = sao2_hill(pd.to_numeric(df["SpO2"], errors="coerce"))
       df = df.drop(columns=["SpO2"])

    if "Resp_rate" in df.columns:

        #ancienne méthode
        df["Resp_severity"] = _resp_severity(df)
        df = df.drop(columns=["Resp_rate"])

    # Temp in °C
    if "Temp" in df.columns:
        df = _temp_to_cat(df, "Temp")

    if "Hem_mal_AML" in df.columns and "Leukocytes" in df.columns:
        hem_mal_aml = pd.to_numeric(df["Hem_mal_AML"], errors="coerce")
        leukocytes = pd.to_numeric(df["Leukocytes"], errors="coerce")
        df["Leukostase"] = _bool_to_float(
            _nullable_and(_eq_nullable(hem_mal_aml, 1), _gt_nullable(leukocytes, 50))
        )

    
    # Neutrophils: clean to numeric and category
    if "Neutrophils" in df.columns and "Leukocytes" in df.columns:
        # Extract numbers like "1.2", "1,2", "1.2 x10^9/L"
        s = df["Neutrophils"].astype(str).str.extract(r"([+-]?\d+(?:[.,]\d+)?)", expand=False)
        val = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
        s_leuko = df["Leukocytes"].astype(str).str.extract(r"([+-]?\d+(?:[.,]\d+)?)", expand=False)
        val_leuko = pd.to_numeric(s_leuko.str.replace(",", ".", regex=False), errors="coerce")
        df["Neutropenie"] = _bool_to_float(
            _nullable_or(_lt_nullable(val, 0.5), _lt_nullable(val_leuko, 1))
        )
        
        df = df.drop(columns=["Leukocytes","Neutrophils"])
        # Ancienne méthode - catégoriser
        # neutrophiles_scaled, vmin, vmax = _scale_minmax(df["Neutrophils_num"])
        # df["Neutrophils_scaled"] = neutrophiles_scaled
        # df["Neutrophils_cat"] = _neutro_category(val)
        # df = df.drop(columns=["Neutrophils_num","Neutrophils_scaled"])
        # Verfier les données absurdes

    # merging of same_signification columns 
    if "Vasopressors" in df.columns and "Septic_shock" in df.columns:
        septic_shock = pd.to_numeric(df["Septic_shock"], errors="coerce")
        vasopressors = pd.to_numeric(df["Vasopressors"], errors="coerce")
        df["Hypotension"] = _bool_to_float(
            _nullable_or(_eq_nullable(septic_shock, 1), _eq_nullable(vasopressors, 1))
        )
        df = df.drop(columns=["Septic_shock","Vasopressors"])
        
    if "Drug_induced" in df.columns and "Immuno_drugs" in df.columns:
        drug_induced = pd.to_numeric(df["Drug_induced"], errors="coerce")
        immuno_drugs = pd.to_numeric(df["Immuno_drugs"], errors="coerce")
        df["Immuno_drugs"] = _bool_to_float(
            _nullable_or(_ge_nullable(drug_induced, 1), _ge_nullable(immuno_drugs, 1))
        )
        df = df.drop(columns=["Drug_induced"])

    if "CT_Pleural_eff" in df.columns and "Pleural_eff" in df.columns:
        pleural_eff = pd.to_numeric(df["Pleural_eff"], errors="coerce")
        ct_pleural_eff = pd.to_numeric(df["CT_Pleural_eff"], errors="coerce")
        df["Pleural_eff"] = _bool_to_float(
            _nullable_or(_eq_nullable(pleural_eff, 1), _eq_nullable(ct_pleural_eff, 1))
        )
        df = df.drop(columns=["CT_Pleural_eff"])

    if "CT_Excavation" in df.columns and "Excavation" in df.columns:
        ct_excavation = pd.to_numeric(df["CT_Excavation"], errors="coerce")
        excavation = pd.to_numeric(df["Excavation"], errors="coerce")
        df["Excavation"] = _bool_to_float(
            _nullable_or(_eq_nullable(ct_excavation, 1), _eq_nullable(excavation, 1))
        )
        df = df.drop(columns=["CT_Excavation"])
    cols_alveolar = [
        "Alveolar_cons_Focal",
        "Alveolar_cons_Diffuse",
        "Alveolar_xray_Focal",
        'Alveolar_xray_Diffuse']

    if  all(c in df.columns for c in cols_alveolar):
        df["Alveolar"] = _rowwise_max_numeric(df, cols_alveolar)
        df = df.drop(columns=cols_alveolar)

    col_proph_anti_fongique = [
        "Hem_mal_AML",
        "HSCT_BMT_Allograft"
    ]
    if all(c in df.columns for c in col_proph_anti_fongique):
        df["Indication_prophy_anti_fun"] = _rowwise_max_numeric(df, col_proph_anti_fongique)
        # df["Indication_prophy_fungal_taken"] = (
        #                                     (df["Indication_prophy_anti_fun"] == 1) &
        #                                     (df["Prophylaxis_antifungal"] == 1)
        #                                 )
        # df["Indication_prophy_fungal_not_taken"] = (
        #                                     (df["Indication_prophy_anti_fun"] == 1) &
        #                                     (df["Prophylaxis_antifungal"] == 0)
        #                                 )
        # df = df.drop(columns = ["Indication_prophy_anti_fun","Prophylaxis_antifungal"])
    col_indic_pneumocystose = [
        "Hem_mal_ALL",
        "HSCT_BMT_Allograft",
        "HSCT_BMT_Autograft",
        "Hem_mal_Non_hodgkin_lymphoma",
        "Ibr_Flu_Met",
        #sklerodermi, granulomatosis, rheumatoid arthritis, wegener ou  artitisreumatoide ou good pasture syndrome
        "Organ_transpl",
        "Steroids_YN"

    ]
    if "Sys_dis_spec" in df.columns:
        diseases = [
            "sklerodermi",
            "granulomatosis",
            "rheumatoid arthritis",
            "wegener",
            "artitisreumatoide",
            "good pasture syndrome"
        ]
        target_disease = df["Sys_dis_spec"].apply(lambda x: contains_keywords(x, diseases))
        df["has_target_disease"] = _bool_to_float(pd.Series(target_disease, index=df.index, dtype="boolean"))
        df = df.drop(columns = ["Sys_dis_spec"])
    if all(c in df.columns for c in col_indic_pneumocystose):
        df["Indication_prophy_pneumocystose"] = _rowwise_max_numeric(df, col_indic_pneumocystose)
        if "has_target_disease" in df.columns:
            df["Indication_prophy_pneumocystose"] = _rowwise_max_numeric(
                df,
                ["Indication_prophy_pneumocystose", "has_target_disease"],
            )
            df = df.drop(columns = ["has_target_disease"])
        indication_pneumo = pd.to_numeric(df["Indication_prophy_pneumocystose"], errors="coerce")
        prophylaxis_pneumo = pd.to_numeric(df["Prophylaxis_pneumocystis"], errors="coerce")
        df["Indication_prophy_pneumocystose_taken"] = _bool_to_float(
            _nullable_and(_eq_nullable(indication_pneumo, 1), _eq_nullable(prophylaxis_pneumo, 1))
        )
        df["Indication_prophy_pneumocystose_not_taken"] = _bool_to_float(
            _nullable_and(_eq_nullable(indication_pneumo, 1), _eq_nullable(prophylaxis_pneumo, 0))
        )
        df = df.drop(columns = ["Indication_prophy_pneumocystose","Prophylaxis_pneumocystis"])
        
    bacterial_columns = ["BACTERIAL", "DG1","DG2"]
    # Ces colonnes sont à rajouter pour les bacterials
    # les pneumonie cliniquement documentée sont les "bacteria"=1 mais il y a une erreur. 
    # Quand tu regardes les colonnes roses à gauche des jaune et appelés DG1,Dg2....
    # certaines ont écrit "legionella" ces pneumonie là sont documentee et auraient dû être en  "bacteria"=2
    # Donc pour faire l'analyse sur les pneumonie microbiologiquement documentée, 
    # il faut prendre les bacteria == 2 +les legionelles de la colonne dg1ou dg2.
    if all(c in df.columns for c in bacterial_columns):
            bacterial = pd.to_numeric(df["BACTERIAL"], errors="coerce")
            legionella_dg1 = _str_contains_nullable(df["DG1"], "legionella")
            legionella_dg2 = _str_contains_nullable(df["DG2"], "legionella")
            pneumonia_microbio = _nullable_or(
                _eq_nullable(bacterial, 2.0),
                legionella_dg1,
                legionella_dg2,
            )
            df["Pneumonia_microbio"] = _bool_to_float(pneumonia_microbio)
            df["Pneumonia_clinic"] = _bool_to_float(
                _nullable_and(_eq_nullable(bacterial, 1.0), ~pneumonia_microbio)
            )

            df = df.drop(columns = bacterial_columns)
    return df


def apply_mapping(df: pd.DataFrame, col: str, mapping: dict, prefix: str = "") -> pd.DataFrame:
    """
    Applique un mapping de valeurs numériques → labels textuels,
    puis crée des colonnes binaires (0/1) pour chaque catégorie.

    Args:
        df (pd.DataFrame): le DataFrame original
        col (str): le nom de la colonne à transformer
        mapping (dict): dictionnaire {valeur_originale: "label"}
        prefix (str, optional): préfixe pour les colonnes binaires.
                                Si None, utilise le nom de la colonne.

    Returns:
        pd.DataFrame: DataFrame enrichi avec la colonne catégorielle et les colonnes binaires
    """
    if col not in df.columns:
        raise KeyError(f"La colonne '{col}' n'existe pas dans le DataFrame")

    if prefix == "":
        prefix = col

    source_values = pd.to_numeric(df[col], errors="coerce")

    # 1️⃣  Créer la colonne catégorielle textuelle
    df[f"{prefix}_cat"] = source_values.map(mapping)

    source_na = source_values.isna()

    # 2️⃣  Créer les colonnes binaires pour chaque catégorie
    for label in mapping.values():
        safe_label = str(label).replace(" ", "_")
        new_col = f"{prefix}_{safe_label}"
        encoded = pd.Series(pd.NA, index=df.index, dtype="Float64")
        valid_rows = ~source_na
        if valid_rows.any():
            encoded.loc[valid_rows] = (
                df.loc[valid_rows, f"{prefix}_cat"] == label
            ).astype(float).to_numpy()
        df[new_col] = encoded
    df = df.drop(columns=[col,f"{prefix}_cat"])
    return df


def sao2_hill(po2, p50=26.8, n=2.7):
    """
    po2 : pression partielle d'O2 en mmHg (scalaire ou array)
    retourne SaO2 entre 0 et 1
    """
    po2 = np.asarray(po2, dtype=float)
    return (po2**n) / (po2**n + p50**n)

def _scale_minmax(series: pd.Series) -> Tuple[pd.Series, float, float]:
    """Retourne la série mise à l'échelle dans [0,1] + (vmin, vmax)."""
    # 1) Série -> numérique (NaN si erreur)
    x_num = pd.to_numeric(series, errors="coerce")

    # 2) Pandas -> NumPy ndarray float64 (type sûr pour np.nanmin/np.nanmax)
    x_np: np.ndarray = x_num.to_numpy(dtype=np.float64, copy=False)

    # 3) Bornes
    if np.isfinite(x_np).any():
        vmin: float = float(np.nanmin(x_np))
        vmax: float = float(np.nanmax(x_np))
    else:
        vmin, vmax = 0.0, 1.0

    # 4) Cas dégénéré (vmax == vmin ou bornes non finies)
    if not np.isfinite(vmax - vmin) or vmax == vmin:
        zeros = pd.Series(np.zeros(x_np.shape, dtype=np.float64), index=series.index, dtype="float64")
        zeros.loc[x_num.isna()] = np.nan
        return zeros, vmin, vmax

    # 5) Scaling
    xs_np = (x_np - vmin) / (vmax - vmin)
    xs = pd.Series(xs_np, index=series.index, dtype="float64")
    return xs, vmin, vmax




def _temp_to_cat(
    df: pd.DataFrame,
    col: str,
    *,
    prefix: str = "",
    drop_first: bool = False
) -> pd.DataFrame:
    """
    - Convertit df[col] en float.
    - Crée une colonne numérique df[f"{col}_gravité"] (score de gravité entre 0 et 1).

    Scores :
        Hypothermie  -> 0.2
        Normale      -> 0.0
        Fièvre       -> 0.6
        Hyperthermie -> 1.0
    """

    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")

    # ➤ Binning en catégories
    bins = [0, 36, 37.5, 39, 100]
    scores = [0.2, 0.0, 0.6, 1.0]  # scores correspondants à chaque intervalle

    # Création d’une colonne catégorielle temporaire
    cat = pd.cut(df[col], bins=bins, labels=False, right=False)

    # Attribution des scores correspondants
    severity = pd.Series(pd.NA, index=df.index, dtype="Float64")
    mask = cat.notna()
    if mask.any():
        severity.loc[mask] = cat.loc[mask].map(lambda x: float(scores[int(x)])).to_numpy()
    df[f"{col}_gravité"] = severity

    # ➤ Supprimer la colonne d'origine
    df = df.drop(columns=[col])

    return df



def _resp_severity(df: pd.DataFrame) -> pd.Series:
    """
    0 = normal, 1 = léger.
    Règles:
      - Sinon, utiliser SpO2 si dispo: <88 -> 3, 88-91 -> 2, 92-94 -> 1, >=95 -> score selon RR
      - Sinon, catégoriser par RR seul: <12 -> 2, 12-20 -> 0, 21-29 -> 1, >=30 -> 2
    """
    rr = pd.to_numeric(df.get("Resp_rate", pd.Series(index=df.index, dtype=float)), errors="coerce")
    sev = pd.Series(pd.NA, index=df.index, dtype="Float64")
    sev.loc[rr.notna()] = 0.0
    sev.loc[rr >= 30] = 1.0
    df = df.drop(columns=["Resp_rate"])
    return sev


def _encode_dg1_etiology(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map DG1 free text codes to the binary etiology columns expected downstream.
    Unknown codes fall back to "Other causes" so that no patient is silently dropped.
    """
    if "DG1" not in df.columns:
        return df

    df = df.copy()
    normalized = df["DG1"].astype(str).str.strip().str.upper()
    coverage = pd.Series(False, index=df.index)

    for column in ETIOLOGY_COLUMNS:
        match = normalized.isin(ETIOLOGY_MAPPING.get(column, set()))
        df.loc[:, column] = match.astype("int8")
        if column != "Etiology_Other causes (specify below)_Definitive diagnosis":
            coverage = coverage | match

    other_mask = ~coverage
    df.loc[other_mask, "Etiology_Other causes (specify below)_Definitive diagnosis"] = 1
    return df


def _convert_all_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit toutes les colonnes d'un DataFrame en valeurs numériques.

    Args:
        df (pd.DataFrame): Le DataFrame à convertir.

    Returns:
        pd.DataFrame: Un nouveau DataFrame avec toutes les colonnes converties en numériques.
    """
    df = df.copy()

    # convertir toutes les colonnes en numérique (float64). Les valeurs non convertibles deviennent NaN.
    df = df.apply(lambda col: pd.to_numeric(col, errors="coerce")).astype("float64")

    return df

