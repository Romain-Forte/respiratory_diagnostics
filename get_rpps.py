"""Utilitaires pour vérifier la présence d'un numéro RPPS dans la base PS."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd


# Emplacement du fichier contenant les RPPS et colonnes utiles à extraire.
DATA_FILE_PATH = Path(r"C:\Users\romai\Desktop\travail\ps-libreacces-savoirfaire.txt")
FILE_ENCODING = "utf-8"
RPPS_COLUMNS = [
    "Identifiant PP",
    "Identification nationale PP",
    "Nom d'exercice",
    "Prénom d'exercice",
    "Code profession",
    "Libellé profession",
]
IDENTIFIANT_COLUMN = "Identifiant PP"


@lru_cache(maxsize=1)
def _load_rpps_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Charge le fichier RPPS (et met en cache la DataFrame résultante)."""

    resolved_path = file_path or DATA_FILE_PATH
    

    return pd.read_csv(
        resolved_path,
        sep="|",
        usecols=RPPS_COLUMNS,
        dtype=str,
        encoding=FILE_ENCODING,
    )


def verifier_rpps(numero_rpps: str) -> pd.DataFrame:
    """Retourne les lignes correspondant au numéro RPPS fourni (DataFrame vide sinon)."""

    numero_normalise = str(numero_rpps).strip()
    if not numero_normalise:
        raise ValueError("Le numéro RPPS fourni est vide.")

    rpps_df = _load_rpps_data()
    resultats = rpps_df[rpps_df[IDENTIFIANT_COLUMN] == numero_normalise]
    resultats = resultats.drop_duplicates()
    return resultats


if __name__ == "__main__":
    rpps_cible = "10101621273"
    
    resultats = verifier_rpps(rpps_cible)

    if resultats.empty:
        print(f"Le numéro RPPS {rpps_cible} est absent de la base.")
    else:
        print(f"RPPS {rpps_cible} trouvé :")
        print(resultats.to_string(index=False))
