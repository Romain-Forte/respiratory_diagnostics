import pandas as pd
import matplotlib.pyplot as plt

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
    return df

def nettoyer_nan_par_colonne(df, strategies):
    """
    Remplace les NaN colonne par colonne selon la stratégie choisie.

    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame à nettoyer.
    strategies : dict
        Dictionnaire {colonne: méthode} où la méthode peut être :
          - "median" : remplace par la médiane de la colonne
          - "zero"   : remplace par 0
          - une valeur numérique ou textuelle spécifique (ex: "inconnu", 99, etc.)

    Retour
    ------
    df_nettoye : pd.DataFrame
        Le DataFrame après nettoyage.
    """

    df = df.copy()

    for col, methode in strategies.items():
        if col not in df.columns:
            print(f"⚠️ Colonne '{col}' absente du DataFrame, ignorée.")
            continue
        # 🔹 Tenter conversion numérique si c’est une stratégie numérique
        if methode in ["median", "zero"] or isinstance(methode, (int, float)):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if methode == "median":
            if pd.api.types.is_numeric_dtype(df[col]):
                mediane = df[col].median()
                df[col] = df[col].fillna(mediane)
                print(f"🔹 {col} → NaN remplacés par la médiane ({mediane})")
            else:
                print(f"⚠️ {col} n’est pas numérique → médiane impossible, non modifiée.")

        elif methode == "zero":
            df[col] = df[col].fillna(0)
            # print(f"🔹 {col} → NaN remplacés par 0")

        else:
            df[col] = df[col].fillna(methode)
            print(f"🔹 {col} → NaN remplacés par '{methode}'")

    nb_nan_restants = df.isna().sum().sum()
    print(f"\n✅ Nettoyage terminé. NaN restants : {nb_nan_restants}")
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
