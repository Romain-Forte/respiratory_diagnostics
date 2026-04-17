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
      - missing : colonnes non trouvés
      - special : transformation spéciale requise

    present -> Booléen si la colonne source est présente dans les colonnes de df
    """

    # mapping "cible" <- "source" & "type"
    mapping = [
        # Démographie
        ("Sex", "SEXE", "special"), #anticonvention 
        ("Age", "age", "rename"), # il faudra le scale apres

        # Temps / délais
        ("Time H-ICU", "delai_hop_rea", "rename"),
        ("TIME SYMPTOMES-ICU", "delai_symptome_rea", "rename"),
        ("Time  DG-ICU", "delai_diag_patho", "rename"),

        # Hémato / oncologie / greffes
        ("Hem_mal", "TYPHEMO", "special"), #travail a faire sur nom_hemop et TyPHEMO 8 et 9 a merge 
        ("HSCT_BMT_Allograft", "ALLOGREFFE", "rename"),
        ("HSCT_BMT_Autograft", "AUTOGREFFE", "rename"),
        ("Sys_dis", "Maladie_syst", "rename"), # a retravailler selon type MS, pas dans les codes
        ("Solid_tumor", "TUMEURSOLIDE", "rename"),
        ("Organ_transpl", "greffe_organe_solide", "rename"),
        ("Chemotherapy", "LCHIM", "speciale"), #LCHIM > 1 speciale
        ("Immuno_drugs", "IMMUNSUP", "rename"),
        ("Steroids_YN", "CORTICO", "rename"),

        # Prophylaxies
        ("Prophylaxis_pneumocystis", "prphy_pcp", "rename"),
        ("Prophylaxis_antifungal", "prophy_antifung", "rename"),
        ("Prophylaxis_viral", "prophy_virus", "rename"),

        # Scores / clinique
        ("SOFA_Nervous", "GLASGOW", "special"),
        ("Charlson_index", "Charlson", "rename"),
        ("SOFA_score", "SOFA", "rename"),
        ("Hemoptysis", "HEMOPTYSIE", "rename"),

        # Resp / bio
        ("Resp_rate", "FRMAX", "rename"),
        ("SaO2", "SAO2MIN", "rename"),
        ("Temp", "TEMPMAX", "rename"),
        ("Leukocytes", "LEUCO", "rename"), #Inutile
        ("Neutropenie", "NEUTROPENIE", "rename"),
        ("PaO2/FiO2 VALUE VALUE", ["SPAO2FIO2", "PAO2FIO2_meca"], "special"),# aggregate avec un mean, a reverifier 

        # Scanner et radio dans la TDM il y a plein de variables supplémentaires
        ("Quad_no", "NBQUADR", "rename"),
        ("Pleural_eff", ["RADIOTHOR_CHOICE_6", "RTDMTHOR_CHOICE_12"], "aggregate"), 
        ("Excavation",["RADIOTHOR_CHOICE_7", "RTDMTHOR_CHOICE_11"], "aggregate"),
        ("Septal_line", "RTDMTHOR_CHOICE_9", "rename"),
        ("Halo_sign","RTDMTHOR_CHOICE_10", "rename"),
        ("Lymph_bulky", ["RTDMTHOR_CHOICE_18","RTDMTHOR_CHOICE_19","RTDMTHOR_CHOICE_17"], "aggregate"),
        ("GGO",["RADIOTHOR_CHOICE_4", "RTDMTHOR_CHOICE_1", "RTDMTHOR_CHOICE_2"], "aggregate"),
        ("Nodules_any",["RADIOTHOR_CHOICE_5", "RTDMTHOR_CHOICE_3", "RTDMTHOR_CHOICE_4", "RTDMTHOR_CHOICE_5", "RTDMTHOR_CHOICE_6"], "aggregate"), 
        ("Alveolar",["RADIOTHOR_CHOICE_1","RADIOTHOR_CHOICE_2","RADIOTHOR_CHOICE_3", "RTDMTHOR_CHOICE_7", "RTDMTHOR_CHOICE_8","RTDMTHOR_CHOICE_13",], "aggregate"), 

        # Variables cibles non identifiables directement dans tes colonnes
        ("Disease_status_remission", "REM", "rename"), #REM = remission, les autres pas possibles
        ("GvHD", "remarques_TTT", "regex"),#introuvable mais 2 lignes de remarques_TTT avec GvHD
        ("Drug_induced", None, "missing"), # a eclaircir, pas trouvé 
        ("Ibr_Flu_Met", "remarques_TTT", "regex"),# dans remarques_TTT ou type_IS fludarabine METOTREXATE Ibrutinib
        ("Ibr_Flu_Met", "Type_IS", "regex"),
        ("Tar_ther", None, "missing"),# pas trouvé
        ("Immunotherapy", None, "missing"),# pas trouvé
        ("Carttcells", None, "missing"),# pas trouvé
        ("Hypotension", "REAAMIN", "rename"),
        ("Prophylaxis_bacterial", None, "missing"),# pas trouvé
        ("Vaccins#Flu", None, "missing"), #pas trouvé
        ("Vaccins#COVID", None, "missing"), #pas trouvé
        ("Vaccins#Other", None, "missing"), #pas trouvé
        ("Disease_status_inaugural", None, "missing"), #pas trouvé
        ("Disease_status_evolutive", None, "missing"), #pas trouvé
        #Diagnostiques sur DIAGPRINCIPAL_final
        ('Diagnostique', "DIAGPRINCIPAL_final.recod", "special"),
    ]

    def est_present(src):
        if src is None:
            return False
        if isinstance(src, list):
            return any(col in colonnes_df for col in src)
        return src in colonnes_df


    
    mapping_df = pd.DataFrame(mapping, columns=["target", "source", "type"])
    # affiche les sources non présentes
    mapping_df["source_presente"] = mapping_df["source"].apply(est_present)

    # df_absent = mapping_df[~mapping_df["source_presente"]]

    # for _, row in df_absent.iterrows():
    #     print("Impossible to construct", row["target"], "a partir de", row["source"])


    return mapping_df

def contains_keywords(text, keywords):
        text = text.lower()
        return any(keyword in text for keyword in keywords)



def format_to_efraim(df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    
    Les NaN sont remplacées par 0 lorsque nécessaire.
    Format to efraim column names
    """
    df = df.copy()
    required_cols = {"target", "source", "type", "source_presente"}
    if not required_cols.issubset(mapping_df.columns):
        raise ValueError(f"mapping_df doit contenir les colonnes {required_cols}")
    

    # 1) Renommage simple 1->1
    rename_rows = mapping_df[
        (mapping_df["type"] == "rename")
        & mapping_df["source_presente"]
        & mapping_df["source"].apply(lambda x: isinstance(x, str))
    ]

    rename_dict = dict(zip(rename_rows["source"], rename_rows["target"]))
    # print(rename_dict.keys(),df.keys())
    df_exit = df[rename_rows["source"]].rename(columns=rename_dict)

    # 2) Synthèse des colonnes aggregate
    derive_rows = mapping_df[
            (mapping_df["type"] == "aggregate") & mapping_df["source_presente"]
        ]
    

    for _, row in derive_rows.iterrows():
        target = row["target"]
        sources = row["source"]
        for source in sources:
            df = df.assign(**{
                    source: pd.to_numeric(df[source], errors="coerce")
                })

        df_exit[target] = df[sources].max(axis = 1)

    # 3 Regex 
    df["remarques_TTT"] = df["remarques_TTT"].astype(str)

    df["Type_IS"] = df["Type_IS"].astype(str)
    rejet_greffe = [
            "gvhd"
        ]
    df_exit["GvHD"] = df["remarques_TTT"].apply(
                    lambda x: contains_keywords(x, rejet_greffe)
                )
    Ibr_Flu_Met = [
            "fludarabine","metotrexate","ibrutinib"
            ]
    
    df_exit["Ibr_Flu_Met1"] = df["remarques_TTT"].apply(
                    lambda x: contains_keywords(x, Ibr_Flu_Met)
                )
    df_exit["Ibr_Flu_Met2"] = df["Type_IS"].apply(
                    lambda x: contains_keywords(x, Ibr_Flu_Met)
                )
    df_exit["Ibr_Flu_Met"] = df_exit[["Ibr_Flu_Met1","Ibr_Flu_Met2"]].max(axis = 1)
    df_exit = df_exit.drop(columns = ["Ibr_Flu_Met1","Ibr_Flu_Met2"])

    # 4 Special 

    # Sexe
    df_exit["Sex"] = (df["SEXE"] == 0)
    
    # Hem_mal
        
    mapping = {
         0: 0,
           1:1,
           2:2,
           3 : 4,
           4:5,
           5:8,
           6:7,
           7:6,
           8:9,
           9:9,
           10:3
        }
    df_exit["Hem_mal"] = df[ "TYPHEMO"].map(mapping)

    # Chimiotherapy
    df_exit["Chemotherapy"] = df["LCHIM"] >= 1

    #PaO2/FiO2 pas fou 
    df_exit["PaO2/FiO2 VALUE VALUE"] = df["PAO2FIO2_meca"]  
     
    # Sofa nervous
    df_exit["SOFA_Nervous"] = 0
    df["GLASGOW"] = pd.to_numeric(df["GLASGOW"], errors="coerce")
    df_exit.loc[df["GLASGOW"] < 6,"SOFA_Nervous"] += 1
    df_exit.loc[df["GLASGOW"] < 10,"SOFA_Nervous"] += 1
    df_exit.loc[df["GLASGOW"] < 13,"SOFA_Nervous"] += 1
    df_exit.loc[df["GLASGOW"] < 15,"SOFA_Nervous"] += 1

    # SaO2 pour avoir en % 
    df_exit["SaO2"] = pd.to_numeric(df_exit["SaO2"], errors="coerce")
    df_exit["SaO2"] = df_exit["SaO2"]/ 100

    #Diagnostiques sur DIAGPRINCIPAL_final
    raw_to_category = {
        "bact": "Bactérien",
        "bact docu": "Bactérien microbiologiquement documenté",
        "extra": "ARDS",
        "infiltratif": "Spécifique",
        "bact cli": "Bactérien cliniquement documenté",
        "influenza": "Virus",
        "pcp": "Pneumocystis",
        "aspiration": "Aspiration",
        "virus": "Virus",
        "candidemie": "IFI autre",
        "airways": "Autre",
        "pleura": "Autre",
        "tox": "Toxicité (DRPT)",
        "ipa": "Aspergillus, IFI",
        "copd": "Autre",
        "autre": "Autre",
        "api": "Aspergillus, IFI",
        "vrs": "Virus",
        "ep": "Embolism",
        "pe": "Embolism",
        "drpt": "Toxicité (DRPT)",
        "rhinovirus": "Virus",
        "cmv": "Virus",
        "metapneumovirus": "Virus",
        "coronavirus": "Virus",
        "piv3": "Virus",
        "piv3-metapneumo": "Virus",
        "parasite": "Infectieux autre",
        "adenovirus": "Virus",
        "bk": "Mycoactérie ou infectieux autre",
        "trichosporon": "Infectieux autre",
        "bact docu + cand": "Bactérien + IFI autre",
        "enterovirus": "Virus",
        "mucor": "mucor",
        "neut recov": "ARDS",
        "hsv": "Virus",
        # Diag2
        "fusariose":"IFI autre",
        "pcp/ipa" : "Pneumocystis",
        "ifi conclusion" : "IFI autre",
        "candid√©mie" : "IFI autre",
        "metapneumo" : "Virus",
        "samr" : "Bactérien",
        "metapneumo" : "Virus",
        "vrs-meta" : "Virus",
        

    }
    
    # 2) mapping catégorie -> code numérique
    category_to_code = {
        "Bactérien": 2,
        "Bactérien microbiologiquement documenté": 2,
        "ARDS": 1,
        "Spécifique": 7,  
        "Bactérien cliniquement documenté":2,
        "Virus": 3,
        "Pneumocystis": 4,
        "Aspiration": 13,
        "IFI autre": 6,
        "Autre": 13,
        "Toxicité (DRPT)": 10,
        "Aspergillus, IFI": 14,
        "Embolism": 13,
        "Infectieux autre": 11,
        "Mycoactérie ou infectieux autre": 11,
        "Bactérien + IFI autre": 11,
        "mucor" : 17
    }
    
    # création de la catégorie puis du code
    df["diag1"] = df["DIAGPRINCIPAL_final.recod"].str.lower().map(raw_to_category)
    df["diag2"] = df["DIAG2"].str.lower().map(raw_to_category)
    df["diag1_code"] = df["diag1"].map(category_to_code).fillna(0).astype(int)
    df["diag2_code"] = df["diag2"].map(category_to_code).fillna(0).astype(int)
    # df_exit["diag1_code"] = df["diag1"].map(category_to_code).fillna(0).astype(int)
    # df_exit["diag2_code"] = df["diag2"].map(category_to_code).fillna(0).astype(int)
    L_col_diag = ["diag1_code","diag2_code","DIAGPRINCIPAL_final.recod"]
    df_exit['Bacterial infection'] = (df[L_col_diag]  == 2).any(axis=1) | (df[L_col_diag]  == 9).any(axis=1)
    df_exit['Viral infection'] = (df[L_col_diag]  == 3).any(axis=1)
    df_exit['Invasive pulmonary aspergillosis'] = (df[L_col_diag]  == 14).any(axis=1)
    df_exit['All fungus'] = (df[L_col_diag]  == 5).any(axis=1) | (df[L_col_diag]  == 6).any(axis=1)
    df_exit['Other fungal'] =  (df[L_col_diag]  == 6).any(axis=1)
    df_exit['Mucorales'] =  (df[L_col_diag]  == 17).any(axis=1)
    df_exit['Pneumocystis jirovecii infection'] = (df[L_col_diag]  == 4).any(axis=1)
    df_exit['Cardiogenic pulmonary oedema'] = (df[L_col_diag]  == 1).any(axis=1)
    df_exit['Disease-related infiltrates'] = (df[L_col_diag]  == 7).any(axis=1)
    df_exit['Drug toxicity related'] = (df[L_col_diag]  == 10).any(axis=1)
    df_exit['Other infection'] = (df[L_col_diag]  == 11).any(axis=1)
    # df_exit['Undetermined cause'] = (df[L_col_diag]  == 12).any(axis=1)
    df_exit['Other non infectious causes'] = (df[L_col_diag]  == 13).any(axis=1) | (df[L_col_diag]  == 15).any(axis=1) | (df[L_col_diag]  == 16).any(axis=1)

    # 5) None values
    none_rows = mapping_df[mapping_df["type"] == 'missing']

    for _, row in none_rows.iterrows():
        df_exit[row["target"]] = 0
    # bacterial 2 ou 9
    # ('Viral infection', "3", "rename"),
    # ("Invasive pulmonary aspergillosis", "14", "rename"),
    # ("All fungus", "5,6", "rename"),
    # ("Other fungal", "6", "rename"),
    # ("Mucorales", "?", "rename"),
    # ("Pneumocystis jirovecii infection", "4", "rename"),
    # ("Cardiogenic pulmonary oedema", "1", "rename"),
    # ("Disease-related infiltrates", "7", "rename"),
    # ("Drug toxicity related", "10", "rename"),
    # ("Other infection", "11", "rename"),
    # ("Other non infectious causes", "13 15 16", "rename"),

    
    return df_exit


def ensure_float_clip(df: pd.DataFrame) -> pd.DataFrame:
    clip_max_dict = {
        "Age" : 100,
        "GGO" : 1 ,
        "Time H-ICU": 365,
        "TIME SYMPTOMES-ICU": 365,
        "Time  DG-ICU": 365,
        "Prophylaxis_antifungal" : 1,
        "Charlson_index": 22,
        "SOFA_score" : 24,
        "Temp" : 43,
        "Quad_no" : 4 ,
        "Excavation": 1,
        "Nodules_any" : 1,
        "PaO2/FiO2 VALUE VALUE" : 500,


    }

    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if col in clip_max_dict.keys():
            mask_a_clip = (df[col] > clip_max_dict[col])
            df.loc[mask_a_clip,col] = clip_max_dict[col]
    return df

