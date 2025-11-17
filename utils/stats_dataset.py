import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
def analyser_variables_binaires(df,visualisation = True,print_results = True):
    """
    Analyse un DataFrame contenant des variables binaires (0/1).

    - Compte le nombre de lignes ayant plus d'un '1'.
    - Affiche le nombre total de lignes concernées.
    - Visualise la distribution des valeurs positives (1) par colonne en % du total.
    """
    # Vérification basique
    if not ((df.isin([0, 1]) | df.isna()).all().all()):
        print("⚠️ Attention : certaines colonnes contiennent des valeurs autres que 0 ou 1.")

    # Calcul des lignes avec plus d’un ‘1’
    nb_plusieurs_uns = (df.sum(axis=1) > 1).sum()

    # Distribution des '1' par colonne
    distribution = df.sum().sort_values(ascending=False)

    # Conversion en pourcentage
    distribution_pct = (distribution / len(df)) * 100
    if print_results:

        print(f"Nombre de lignes contenant plus d’un '1' : {nb_plusieurs_uns}")
        print("\nDistribution des valeurs positives (1) par colonne (% des données totales) :")
        print(distribution_pct.round(2))
        
    if visualisation:
        # Visualisation
        plt.figure(figsize=(15, 8))
        plt.bar(distribution_pct.index, distribution_pct.values)
        plt.title("Proportion de chaque diagnostic dans la BDD")
        plt.xlabel("Nom du diagnostic")
        plt.ylabel("Pourcentage des diagnostics (%)")
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        plt.show()

    return distribution_pct, nb_plusieurs_uns



def analyser_associations_binaires(df, top_n=10):
    """
    Analyse les associations entre variables binaires (0/1).

    - Supprime les lignes contenant des valeurs non binaires.
    - Compte les lignes avec plus d’un '1'.
    - Identifie et affiche les associations les plus fréquentes.
    - Affiche la fréquence des associations en % du total.
    """

    df = df.copy()
    n_total = len(df)  # total de lignes, pour le calcul du %

    # --- 1️⃣ Identifier les lignes contenant des valeurs non binaires
    mask_non_bin = df.apply(lambda col: ~col.isin([0, 1]) & col.notna()).any(axis=1)

    if mask_non_bin.any():
        print("⚠️ Lignes contenant des valeurs non binaires détectées :")
        print(df[mask_non_bin])
        df = df[~mask_non_bin]
        print(f"\n➡️ {mask_non_bin.sum()} ligne(s) supprimée(s) du jeu de données.\n")

    # --- 2️⃣ Lignes avec plusieurs '1'
    lignes_multiples = df[df.sum(axis=1) > 1]
    print(f"Nombre de lignes contenant plus d’un '1' : {len(lignes_multiples)}")

    # --- 3️⃣ Comptage des associations
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
        columns=["Association", "Fréquence"]
    )

    # --- 3️⃣ Convertir en pourcentage du total
    assoc_df["Pourcentage"] = (assoc_df["Fréquence"] / n_total) * 100

    print("\nAssociations les plus fréquentes (% des lignes totales) :")
    print(assoc_df.round(2))

    # --- 4️⃣ Visualisation en %
    plt.figure(figsize=(10, 5))
    plt.barh(
        [' + '.join(a) for a in assoc_df["Association"]],
        assoc_df["Pourcentage"]
    )
    plt.xlabel("Pourcentage des lignes (%)")
    plt.ylabel("Association")
    plt.title(f"Top {top_n} associations de variables binaires")
    plt.gca().invert_yaxis()  # la plus fréquente en haut
    plt.tight_layout()
    plt.show()

    return assoc_df, lignes_multiples
