import pandas as pd

def construire_mapping_renommage(colonnes_df):
    """
    colonnes_df : liste des colonnes du df (ex: df.columns.tolist())
    
    Renvoie 1 dataframe de la forme col cible/col source/type/present 
    col cible -> colonne cible dont l'ontologie est celle d'efraim
    col source -> colonne dont l'ontologie est celle du grroh
    
    "type" peut prendre plusieurs valeurs : 
      - rename :     des colonnes a renommer 
      - aggregate : des colonnes a aggréger
      - regex : des colonnes a extraire du texte de la colonne  
      - missing : colonnes non trouver

    present -> Booléen si la colonne source est présente dans les colonnes de df
    """

    # mapping "cible" <- "source" & "type"
    mapping = [
        # Démographie
        ("Sex", "SEXE", "rename"),
        ("Age", "age", "rename"),

        # Temps / délais
        ("Time H-ICU", "Time H-ICU", "rename"),
        ("TIME SYMPTOMES-ICU", "delai_symptome_rea", "rename"),
        ("Time  DG-ICU", "delai_diag_patho", "rename"),

        # Hémato / oncologie / greffes
        ("Hem_mal", "Hemopathie", "regex"), #travail a faire sur nom_hemop et TyPHEMO 8 et 9 a merge 
        ("HSCT_BMT", "ALLOGREFFE", "rename"),
        ("Sys_dis", "Maladie_syst", "rename"), # a retravailler selon type MS
        ("Solid_tumor", "TUMEURSOLIDE", "rename"),
        ("Organ_transpl", "greffe_organe_solide", "rename"),
        ("Chemotherapy", [ "LCHIM", "NCUR"], "aggregate"), #je sais pas ce que c'est 
        ("Immuno_drugs", "IMMUNSUP", "rename"),
        ("Steroids_YN", ["CORTICO", "CORTREA"], "aggregate"),#je sais pas ce que c'est 

        # Prophylaxies
        ("Prophylaxis_pneumocystis", "prphy_pcp", "rename"),
        ("Prophylaxis_antifungal", "prophy_antifung", "rename"),
        ("Prophylaxis_viral", "prophy_virus", "rename"),

        # Scores / clinique
        ("SOFA_Nervous", "GCSSOFA_J1", "rename"),
        ("SOFA_score", "SOFA", "rename"),
        ("Hemoptysis", "HEMOPTYSIE", "rename"),

        # Resp / bio
        ("Resp_rate", "FRMAX", "rename"),
        ("SpO2", "SAO2MIN", "rename"),
        ("Temp", "TEMPMAX", "rename"),
        ("Leukocytes", "LEUCO", "rename"),
        ("Neutrophils", "NEUTROPENIE", "rename"),

        # Scan il manque plein de variables la 
        ("Quad_no", "NBQUADR", "rename"),
        
        # Variables cibles non identifiables directement dans tes colonnes
        ("Dis_status HEM", ["REM", "NCUR", "LCHIM"], "aggregate"), #pas sur, valeur de base a retravailler
        ("GvHD", "remarques_TTT", "regex"),#introuvable mais 2 lignes de remarques_TTT avec GvHD
        ("Drug_induced", None, "missing"), # a eclaircir, pas trouvé 
        ("Ibr_Flu_Met", "remarques_TTT", "regex"),# dans remarques_TTT ou type_IS fludarabine METOTREXATE Ibrutinib
        ("Tar_ther", None, "missing"),# pas trouvé
        ("Immunotherapy", None, "missing"),# pas trouvé
        ("Carttcells", None, "missing"),# pas trouvé
        ("Prophylaxis_bacterial", None, "missing"),# pas trouvé
        ("Vaccins#Flu", None, "missing"), #pas trouvé
        ("Vaccins#COVID", None, "missing"), #pas trouvé
        ("Vaccins#Other", None, "missing"), #pas trouvé

        # PaO2/FiO2 : pas un rename 1->1 (tu as 2 proxys)
        ("PaO2/FiO2 VALUE VALUE", ["SPAO2FIO2", "PAO2FIO2_meca"], "aggregate"),# avec un mean 
    ]

    def est_present(src):
        if src is None:
            return False
        if isinstance(src, list):
            return any(s in colonnes_df for s in src)
        return src in colonnes_df

    mapping_df = pd.DataFrame(mapping, columns=["target", "source", "type"])
    mapping_df["present_in_df"] = mapping_df["source"].apply(est_present)

    # rename_dict uniquement pour les cas 1→1 présents
    rename_df = mapping_df[
        (mapping_df["type"] == "rename") &
        (mapping_df["present_in_df"]) &
        (mapping_df["source"].apply(lambda x: isinstance(x, str)))
    ]
    rename_dict = dict(zip(rename_df["source"], rename_df["target"]))

    return mapping_df


def renommer_df(df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomme les colonnes directes puis synthétise les entrées `aggregate`.
    Les NaN sont remplacées par 0 lorsque nécessaire.
    """
    def _coerce_numeric(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series.fillna(0)
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            return numeric.fillna(0)
        return series.notna().astype(int)

    df = df.copy()
    required_cols = {"target", "source", "type", "present_in_df"}
    if not required_cols.issubset(mapping_df.columns):
        raise ValueError(f"mapping_df doit contenir les colonnes {required_cols}")

    # 1) Renommage simple 1->1
    rename_rows = mapping_df[
        (mapping_df["type"] == "rename")
        & mapping_df["present_in_df"]
        & mapping_df["source"].apply(lambda x: isinstance(x, str))
    ]
    rename_dict = dict(zip(rename_rows["source"], rename_rows["target"]))
    df.rename(columns=rename_dict, inplace=True)

    # 2) Synthèse des colonnes aggregate
    derive_rows = mapping_df[
        (mapping_df["type"] == "aggregate") & mapping_df["present_in_df"]
    ]
    

    for _, row in derive_rows.iterrows():
        target = row["target"]
        sources = row["source"]
        if not isinstance(sources, list):
            continue

        available = [col for col in sources if col in df.columns]
        if not available:
            continue

        if target == "PaO2/FiO2 VALUE VALUE":
            combined = None
            for col in available:
                values = _coerce_numeric(df[col]).astype(float)
                if combined is None:
                    combined = values
                else:
                    combined = combined.where(combined.notna(), values)
            if combined is not None:
                df[target] = combined.fillna(0)
            continue

        combined = None
        for col in available:
            values = _coerce_numeric(df[col])
            values = (values > 0).astype(float)
            combined = values if combined is None else (combined + values)

        if combined is not None:
            df[target] = (combined > 0).astype(float)

    return df
