# =========================
# 0) Imports
# =========================
import pandas as pd
import numpy as np

from typing import List, Tuple, Dict, Union, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, f1_score, hamming_loss, jaccard_score, roc_auc_score, roc_curve, auc
)


# =========================
# 2) Séparer X / y + split train/test
# =========================
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split

def preparer_jeu_xy(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    align: str = "inner",          # "inner" (intersection), "left" (conserver X), "right" (conserver y)
    ensure_binary: bool = True,    # force les labels en {0,1}
    fill_labels_nan: int = 0       # valeur pour remplir les NaN des labels
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Prépare X/y pour un problème multi-label à partir de deux DataFrames séparés.

    Étapes :
      1) Aligne X_df et y_df sur l'index (inner/left/right).
      2) Nettoie y (NaN -> fill_labels_nan ; optionnellement force binaire).
      3) Split train/test ; stratification simple par nombre d'étiquettes actives.

    Retourne :
      X_train, X_test, y_train, y_test, liste_labels
    """

    # 1) Alignement sur index
    if align == "inner":
        X_al, y_al = X_df.align(y_df, join="inner", axis=0)
    elif align == "left":
        # garder toutes les lignes de X ; aligner y
        X_al, y_al = X_df.copy(), y_df.reindex(X_df.index)
    elif align == "right":
        # garder toutes les lignes de y ; aligner X
        X_al, y_al = X_df.reindex(y_df.index), y_df.copy()
    else:
        raise ValueError("align doit être 'inner', 'left' ou 'right'.")
    # Sanity check
    if len(X_al) != len(y_al):
        raise ValueError("Après alignement, X et y n'ont pas la même longueur.")
    if not X_al.index.equals(y_al.index):
        # pas bloquant, mais on explicite
        X_al, y_al = X_al.sort_index(), y_al.sort_index()
        if not X_al.index.equals(y_al.index):
            raise ValueError("Impossible d'aligner correctement les index de X et y.")

    labels = list(y_al.columns)

    # 2) Nettoyage des labels
    y_clean = y_al.copy()
    y_clean = y_clean.fillna(fill_labels_nan)

    if ensure_binary:
        # Si des valeurs non binaires existent, on les force en {0,1} (strict >0 -> 1)
        # et on prévient si on a changé quelque chose.
        mask_non_bin = ~y_clean.isin([0, 1]).all(axis=1)
        if mask_non_bin.any():
            # conversion prudente : tout >0 devient 1, le reste 0
            y_conv = y_clean.apply(pd.to_numeric, errors="coerce").fillna(0)
            y_conv = (y_conv > 0).astype(int)
            y_clean = y_conv
            print(f"⚠️ {mask_non_bin.sum()} ligne(s) contenaient des labels non binaires — conversion en 0/1 appliquée.")

        # assurer le type
        y_clean = y_clean.astype(int)

    # 3) Stratification simple : par nb d’étiquettes actives (bornée à 3+)
    sums = y_clean.sum(axis=1)
    strat = None
    if sums.nunique() > 1:
        # catégories 0,1,2,3+ pour stabiliser la stratification
        strat = np.clip(sums, 0, 3)

    X_train, X_test, y_train, y_test = train_test_split(
        X_al, y_clean, test_size=test_size, random_state=random_state, stratify=strat
    )

    return X_train, X_test, y_train, y_test, labels

# =========================
# 3) Mise à l’échelle (scaling)
# =========================
def normaliser_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    colonnes_numeriques: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, List[str]]:
    """
    Applique un StandardScaler sur les colonnes numériques.
    - Si 'colonnes_numeriques' est None, on les détecte automatiquement.
    Retourne: X_train_scaled, X_test_scaled, scaler, colonnes_numeriques
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    if colonnes_numeriques is None:
        colonnes_numeriques = X_train.select_dtypes(include="number").columns.tolist()

    scaler = StandardScaler()
    if colonnes_numeriques:
        X_train[colonnes_numeriques] = scaler.fit_transform(X_train[colonnes_numeriques])
        X_test[colonnes_numeriques] = scaler.transform(X_test[colonnes_numeriques])

    return X_train, X_test, scaler, colonnes_numeriques
# =========================
# 4) Entraînement du modèle multi-label
# =========================
def entrainer_modele_multilabel(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    type_modele: str = "random_forest"
):
    """
    Entraîne un modèle multi-label selon 'type_modele':
      - "random_forest" : MultiOutputClassifier(RandomForestClassifier)
      - "logreg_ovr"    : OneVsRest(LogisticRegression)
      - "mlp"           : MLPClassifier (sorties sigmoid via 'logistic')
    Retourne le modèle entraîné.
    """
    if type_modele == "random_forest":
        base = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        model = MultiOutputClassifier(base)
    elif type_modele == "logreg_ovr":
        base = LogisticRegression(max_iter=1000, n_jobs=-1)
        model = OneVsRestClassifier(base)
    elif type_modele == "mlp":
        # Pour multi-label, sklearn MLPClassifier accepte y multi-output binaire
        model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                              learning_rate_init=1e-3, batch_size=64, max_iter=200,
                              random_state=42)
    else:
        raise ValueError("type_modele inconnu. Utiliser: 'random_forest', 'logreg_ovr', ou 'mlp'.")
    model.fit(X_train, y_train)



    return model
# =========================
# 5) Évaluation du modèle multi-label
# =========================
def evaluer_modele_multilabel(
    model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    seuil: float = 0.5
) -> dict:
    """
    Évalue un modèle multi-label avec plusieurs métriques.
    - Gère à la fois les modèles qui retournent des probabilités (predict_proba)
      et ceux qui retournent directement des 0/1 via predict.

    Retourne un dict de métriques + imprime un rapport par classe.
    """
    # Tentative de prédire des proba, sinon classes
    y_pred = None

    # Cas 1: multioutput avec predict_proba par sortie

    # Certains wrappers renvoient une liste de arrays [n_labels] de shape (n_samples, 2) ou (n_samples,)
    probas = []
    for est in getattr(model, "estimators_", []):  # MultiOutputClassifier
        if hasattr(est, "predict_proba"):
            p = est.predict_proba(X_test)
            # Si 2 colonnes (classe 0/1), on prend la proba de la classe 1
            if isinstance(p, list):  # certains modèles renvoient list par label (rare ici)
                p = p[1]
            if p.ndim == 2 and p.shape[1] == 2:
                probas.append(p[:, 1])
            else:
                # Probabilité directement pour la classe positive
                probas.append(p.ravel())
        else:
            probas = None
            break

    if probas:
        y_pred = (np.vstack(probas).T >= seuil).astype(int)

    # Cas 2: OneVsRest/MLP ou fallback -> predict direct
    if y_pred is None:
        y_pred = model.predict(X_test)
    # Calcul des métriques
    metrics = {
        "hamming_loss": hamming_loss(y_test, y_pred),
        "f1_micro": f1_score(y_test, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "jaccard_micro": jaccard_score(y_test, y_pred, average="micro", zero_division=0),
        "jaccard_macro": jaccard_score(y_test, y_pred, average="macro", zero_division=0),
    }

    print("=== Rapport par étiquette ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Métrique partial match score ")
    print(at_least_one_correct(np.array(y_test), y_pred))
    print(np.array(y_test).shape, np.array(probas).T.shape)
    res = multilabel_roc(
        np.array(y_test), y_pred,
        plot=True,
        label_names=y_test.columns.to_list(),
        per_label_max=8,   # ne tracer que les 8 meilleures courbes par label
        show_micro=True,
        show_macro=True
        )
    score = reject_n_lowest_correct(np.array(y_test), np.array(probas).T, n=3)
    print(f"Score 'au moins 4 labels négatifs bien rejetés' = {score:.2f}")
    print("=== Métriques globales ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics
def metrics_of_predictions(y_test,y_pred):
    metrics = {
        "hamming_loss": hamming_loss(y_test, y_pred),
        "f1_micro": f1_score(y_test, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "jaccard_micro": jaccard_score(y_test, y_pred, average="micro", zero_division=0),
        "jaccard_macro": jaccard_score(y_test, y_pred, average="macro", zero_division=0),
    }

    print("=== Rapport par étiquette ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Métrique partial match score ")
    print(at_least_one_correct(np.array(y_test), y_pred))
    score = reject_n_lowest_correct(np.array(y_test), np.array(y_pred), n=3)
    print(f"Score 'au moins 4 labels négatifs bien rejetés' = {score:.2f}")
    print("=== Métriques globales ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics
def at_least_one_correct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Score multi-label : 1 si au moins un label correct est prédit, 0 sinon.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples, n_labels)
        Matrice de vérité (0/1).
    y_pred : np.ndarray of shape (n_samples, n_labels)
        Matrice de prédiction (0/1).

    Returns
    -------
    score : float
        Moyenne du taux de succès (entre 0 et 1).
    """
    # Vérification de forme
    assert y_true.shape == y_pred.shape, "Dimensions incompatibles entre y_true et y_pred."

    # Pour chaque ligne : intersection non vide ?
    success = (y_true * y_pred).sum(axis=1) > 0
    score = success.mean()
    return float(score)

def reject_n_lowest_correct(y_true: np.ndarray, y_pred: np.ndarray, n: int = 3) -> float:
    """
    Score = 1 si les `n` plus petites probabilités correspondent toutes
    à des labels absents (y_true == 0).

    Parameters
    ----------
    y_true : np.ndarray (n_samples, n_labels)
        Matrice de vérité (0/1)
    y_pred : np.ndarray (n_samples, n_labels)
        Probabilités prédites (entre 0 et 1)
    n : int
        Nombre de labels les moins probables à vérifier

    Returns
    -------
    score : float
        Moyenne du taux de réussite (entre 0 et 1)
    """
    assert y_true.shape == y_pred.shape, "y_true et y_pred doivent avoir la même forme"

    success = []
    for yt, yp in zip(y_true, y_pred):
        # Indices des n plus petites probabilités
        lowest_idx = np.argsort(yp)[:n]
        # Vérifier que ces labels sont bien absents
        if np.all(yt[lowest_idx] == 0):
            success.append(1)
        else:
            success.append(0)

    return float(np.mean(success))

def multilabel_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    average: str = "macro",
    plot: bool = True,
    label_names=None,
    per_label_max: int = 10,  # nombre max de courbes par label à afficher
    show_micro: bool = True,
    show_macro: bool = True,
    return_fig_ax: bool = False,
):
    """
    Calcule AUC globaux (micro/macro/weighted), AUC par label et courbes ROC.
    Optionnellement, trace les courbes ROC (par label + micro + macro).

    Paramètres
    ----------
    y_true : array (n_samples, n_labels), binaire
    y_score: array (n_samples, n_labels), scores/probas
    average: str, ignoré ici pour les retours; présent pour compat API
    plot : bool, si True génère une figure matplotlib
    label_names : list[str] ou None, noms de colonnes; sinon indices 0..L-1
    per_label_max : int, nombre max de courbes de labels à dessiner
    show_micro : bool, trace la courbe micro-moyennée si disponible
    show_macro : bool, trace la courbe macro (moyenne des interp. par label valides)
    return_fig_ax : bool, si True retourne aussi (fig, ax)

    Retour
    ------
    dict avec:
      - auc_micro, auc_macro, auc_weighted (floats ou NaN)
      - auc_per_label : array (n_labels) avec NaN pour labels invalides
      - roc_per_label : dict[label] -> (fpr, tpr) ou None
      - roc_micro : (fpr, tpr, auc) ou None
      - roc_macro : (fpr, tpr, auc) ou None
      - plotted_labels : liste des labels effectivement dessinés
    (+ éventuellement fig, ax si return_fig_ax=True)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    assert y_true.shape == y_score.shape and y_true.ndim == 2, "Shapes attendues: (n_samples, n_labels)"

    n_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f"Label {i}" for i in range(n_labels)]

    out = {}

    # AUC globaux
    for avg in ["micro", "macro", "weighted"]:
        try:
            out[f"auc_{avg}"] = roc_auc_score(y_true, y_score, average=avg)
        except ValueError:
            out[f"auc_{avg}"] = np.nan

    # Par label
    per_label_auc = np.full(n_labels, np.nan, dtype=float)
    roc_per_label = {}

    valid_mask = np.zeros(n_labels, dtype=bool)
    for k in range(n_labels):
        yk = y_true[:, k]
        sk = y_score[:, k]
        # besoin d'au moins un positif et un négatif
        if yk.max() == 0 or yk.min() == 1:
            roc_per_label[k] = None
            continue
        fpr_k, tpr_k, _ = roc_curve(yk, sk)
        roc_per_label[k] = (fpr_k, tpr_k)
        per_label_auc[k] = auc(fpr_k, tpr_k)
        valid_mask[k] = True




    out["auc_per_label"] = per_label_auc
    out["roc_per_label"] = roc_per_label

    # Micro
    if valid_mask.any():
        yt = y_true[:, valid_mask].ravel()
        ys = y_score[:, valid_mask].ravel()
        fpr_m, tpr_m, _ = roc_curve(yt, ys)
        out["roc_micro"] = (fpr_m, tpr_m, auc(fpr_m, tpr_m))
    else:
        out["roc_micro"] = None

    # Macro: interpolation sur une grille commune, puis moyenne
    if valid_mask.any():
        # grille FPR commune
        all_fpr = np.unique(np.concatenate([roc_per_label[k][0] for k in np.where(valid_mask)[0]]))
        mean_tpr = np.zeros_like(all_fpr)
        for k in np.where(valid_mask)[0]:
            fpr_k, tpr_k = roc_per_label[k]
            mean_tpr += np.interp(all_fpr, fpr_k, tpr_k)
        mean_tpr /= valid_mask.sum()
        auc_macro_curve = auc(all_fpr, mean_tpr)
        out["roc_macro"] = (all_fpr, mean_tpr, auc_macro_curve)
    else:
        out["roc_macro"] = None

    plotted_labels = []
    fig = ax = None

    if plot:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        # micro
        if show_micro and out["roc_micro"] is not None:
            fpr_m, tpr_m, auc_m = out["roc_micro"]
            ax.plot(fpr_m, tpr_m, label=f"micro-average (AUC={auc_m:.3f})", linewidth=2)

        # macro
        if show_macro and out["roc_macro"] is not None:
            fpr_g, tpr_g, auc_g = out["roc_macro"]
            ax.plot(fpr_g, tpr_g, label=f"macro-average (AUC={auc_g:.3f})", linewidth=2, linestyle="--")

        # choisir jusqu'à per_label_max meilleurs AUC pour lisibilité
        valid_indices = np.where(valid_mask)[0]
        if valid_indices.size:
            # trier par AUC décroissante
            order = valid_indices[np.argsort(per_label_auc[valid_indices])[::-1]]
            to_plot = order[:per_label_max]
            for k in to_plot:
                fpr_k, tpr_k = roc_per_label[k]
                ax.plot(fpr_k, tpr_k, label=f"{label_names[k]} (AUC={per_label_auc[k]:.3f})", alpha=0.85)
            plotted_labels = [label_names[k] for k in to_plot]

        # diagonale aléatoire
        ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1)

        ax.set_xlabel("FPR (False Positive Rate)")
        ax.set_ylabel("TPR (True Positive Rate)")
        ax.set_title("ROC multi-label")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=":", linewidth=0.7)

        plt.tight_layout()

    out["plotted_labels"] = plotted_labels

    if return_fig_ax:
        return out, (fig, ax)
    return out
