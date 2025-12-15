# =========================
# 0) Imports
# =========================
import pandas as pd
import numpy as np

from typing import List, Tuple, Dict, Union, Optional, Callable, Sequence

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from utils.models_and_metrics import get_metric
import seaborn as sns
from typing import List, Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

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
def normaliser_features(X_train, X_test):
    # Sélection des colonnes numériques
    cols_num = X_train.select_dtypes(include=["int64", "float64"]).columns

    # Initialisation du scaler
    scaler = StandardScaler()

    # Ajustement sur le train + transformation
    X_train_sc = X_train.copy()
    X_train_sc[cols_num] = scaler.fit_transform(X_train[cols_num])

    # Transformation du test
    X_test_sc = X_test.copy()
    X_test_sc[cols_num] = scaler.transform(X_test[cols_num])

    return X_train_sc, X_test_sc, scaler, cols_num

class AutoStandardScaler(BaseEstimator, TransformerMixin):
    """
    StandardScaler capable d'être utilisé dans un pipeline sklearn.
    - Détecte automatiquement les colonnes numériques si none.
    - Conserve le DataFrame (retourne DataFrame et non numpy array).
    """

    def __init__(self, colonnes_numeriques: Optional[List[str]] = None):
        self.colonnes_numeriques = colonnes_numeriques
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        # Détection automatique des colonnes numériques
        if self.colonnes_numeriques is None:
            self.colonnes_numeriques = X.select_dtypes(include="number").columns.tolist()

        # Entraîner le StandardScaler uniquement sur ces colonnes
        if self.colonnes_numeriques:
            self.scaler.fit(X[self.colonnes_numeriques])

        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        if self.colonnes_numeriques:
            X[self.colonnes_numeriques] = self.scaler.transform(X[self.colonnes_numeriques])

        return X

# =========================
# 4) Entraînement du modèle multi-label
# =========================
def entrainer_modele_multilabel(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model
):
    """
    Entraîne un modèle multi-label à partir d'un modèle sklearn simple.

    Parameters
    ----------
    X_train : pd.DataFrame
        Données d'entrée
    y_train : pd.DataFrame
        Labels multi-label (plusieurs colonnes binaires)
    model : sklearn model
        Modèle sklearn simple (estimator), ex: RandomForestClassifier()

    Returns
    -------
    multi_label_model : MultiOutputClassifier
        Modèle multi-label entraîné
    """

    # Vérification du format multi-label
    if not isinstance(y_train, pd.DataFrame):
        raise ValueError("y_train doit être un DataFrame multi-label (plusieurs colonnes).")

    # Emballer le modèle dans un MultiOutputClassifier
    multi_label_model = MultiOutputClassifier(model)

    print("🔄 Entraînement du modèle multi-label...")
    multi_label_model.fit(X_train, y_train)

    print("✅ Modèle multi-label entraîné avec succès.")
    return multi_label_model
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


def calculate_score(
    df_features: pd.DataFrame,
    y_true: Union[pd.Series, pd.DataFrame, Sequence[int]],
    score_fn,
    metrics: Optional[Sequence[Union[str, Callable[[pd.Series, pd.Series], float]]]] = None,
    *,
    align: str = "inner",
    confusion_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Union[pd.Series, Dict[str, float], np.ndarray]]:
    """
    Évalue un score "manuel" (ex: Score_alice) sur un jeu de données sans entraînement.

    Args:
        df_features: DataFrame de features.
        y_true: Series/DataFrame cible binaire (une seule colonne) ou séquence alignée sur df_features.
        score_fn: objet exposant une méthode `predict(df)` (ex: Score_alice).
        metrics: liste de noms ("accuracy", "precision", "recall", "f1",
                 "jaccard", "hamming", "confusion_matrix", etc.) ou de callables.
        align: méthode d'alignement des index ("inner", "left", "right").
        confusion_labels: labels des axes pour la matrice de confusion.

    Returns:
        dict avec les métriques calculées, y_true aligné, ainsi que les sorties :
        - `y_pred`: probabilités (si predict_proba disponible) sinon classes.
        - `y_pred_labels`: prédictions binaires issues de `predict`.
    """

    if metrics is None:
        metrics = ["accuracy"]
    else:
        metrics = list(metrics)

    if align not in {"inner", "left", "right"}:
        raise ValueError("`align` doit être parmi {'inner','left','right'}.")

    if not hasattr(score_fn, "predict"):
        raise AttributeError("`score_fn` doit exposer une méthode predict(df).")

    if isinstance(y_true, pd.DataFrame):
        if y_true.shape[1] != 1:
            raise ValueError("`y_true` DataFrame doit contenir une seule colonne binaire.")
        y_series = y_true.iloc[:, 0]
    elif isinstance(y_true, pd.Series):
        y_series = y_true
    else:
        y_seq = pd.Series(y_true)
        if len(y_seq) != len(df_features):
            raise ValueError("`y_true` doit avoir la même longueur que df_features.")
        y_series = pd.Series(y_seq.values, index=df_features.index)

    if align == "inner":
        X_aligned, y_aligned = df_features.align(y_series, join="inner", axis=0)
    elif align == "left":
        X_aligned = df_features.copy()
        y_aligned = y_series.reindex(df_features.index)
    else:  # right
        X_aligned = df_features.reindex(y_series.index)
        y_aligned = y_series.copy()

    if y_aligned.isna().any():
        raise ValueError("y_true contient des valeurs manquantes après alignement.")

    y_aligned = y_aligned.astype(int)
    if X_aligned.empty or y_aligned.empty:
        raise ValueError("L'alignement produit un ensemble vide.")

    y_pred = score_fn.predict(X_aligned)
    if isinstance(y_pred, pd.DataFrame):
        if y_pred.shape[1] != 1:
            raise ValueError("`predict` doit retourner un vecteur 1D ou une colonne unique.")
        y_pred = y_pred.iloc[:, 0]
    y_pred_series = pd.Series(y_pred, index=X_aligned.index, name="y_pred").astype(int)
    if len(y_pred_series) != len(X_aligned):
        raise ValueError("La sortie de `predict` doit avoir la même longueur que df_features.")

    proba_available = hasattr(score_fn, "predict_proba")

    metric_catalog = {name.lower(): cfg for name, cfg in get_metric().items()}
    extra_metrics: Dict[str, Callable[[pd.Series, pd.Series], float]] = {
        "jaccard": lambda yt, yp: jaccard_score(yt, yp, zero_division=0),
        "hamming": lambda yt, yp: hamming_loss(yt, yp),
    }
    supported_names = sorted(set(metric_catalog.keys()) | set(extra_metrics.keys()) | {"confusion_matrix"})

    results: Dict[str, Union[float, np.ndarray]] = {}
    needs_proba = any(
        isinstance(metric, str)
        and metric.lower() in metric_catalog
        and metric_catalog[metric.lower()].get("needs_proba")
        for metric in metrics
    )
    y_pred_proba_series: Optional[pd.Series] = None

    def _plot_confusion(cm: np.ndarray):
        lbls = list(confusion_labels) if confusion_labels is not None else ["0", "1"]
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[f"Prédit {lbls[0]}", f"Prédit {lbls[1]}"],
            yticklabels=[f"Réel {lbls[0]}", f"Réel {lbls[1]}"],
        )
        plt.title("Matrice de confusion")
        plt.ylabel("Classe réelle")
        plt.xlabel("Classe prédite")
        plt.tight_layout()
        plt.show()

    def _extract_positive_proba(
        proba_values: Union[pd.Series, pd.DataFrame, np.ndarray],
        index: pd.Index,
    ) -> pd.Series:
        if isinstance(proba_values, pd.DataFrame):
            if proba_values.shape[1] >= 2:
                series = proba_values.iloc[:, -1]
            elif proba_values.shape[1] == 1:
                series = proba_values.iloc[:, 0]
            else:
                raise ValueError("La DataFrame retournée par predict_proba est vide.")
            return pd.Series(series.values, index=index, name="score_proba")

        if isinstance(proba_values, pd.Series):
            if len(proba_values) != len(index):
                raise ValueError("La Series retournée par predict_proba a une longueur inattendue.")
            return pd.Series(proba_values.values, index=index, name="score_proba")

        arr = np.asarray(proba_values)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            series = arr[:, -1]
        elif arr.ndim == 1:
            series = arr
        else:
            raise ValueError("Format de probabilités non supporté.")

        if len(series) != len(index):
            raise ValueError("La sortie de predict_proba doit avoir la même longueur que df_features.")
        return pd.Series(series, index=index, name="score_proba")

    if proba_available:
        proba_values = score_fn.predict_proba(X_aligned)
        y_pred_proba_series = _extract_positive_proba(proba_values, X_aligned.index)

    if needs_proba and y_pred_proba_series is None:
        raise ValueError(
            "Certaines métriques demandées nécessitent predict_proba, "
            "mais `score_fn` n'expose pas cette méthode."
        )

    for metric in metrics:
        if isinstance(metric, str):
            key = metric.lower()
            if key == "confusion_matrix":
                cm = confusion_matrix(y_aligned, y_pred_series)
                results["confusion_matrix"] = cm
                _plot_confusion(cm)
                continue

            if key in extra_metrics:
                fn = extra_metrics[key]
            elif key in metric_catalog:
                cfg = metric_catalog[key]
                fn = cfg["metric_fn"]
                if cfg.get("needs_proba"):
                    if y_pred_proba_series is None:
                        raise ValueError(
                            f"La métrique '{metric}' nécessite des probabilités mais aucune n'a été calculée."
                        )
                    results[key] = fn(y_aligned.values, y_pred_proba_series.values)
                    continue
            else:
                raise ValueError(
                    f"Métrique inconnue: {metric}. "
                    f"Options disponibles: {', '.join(supported_names)}"
                )
            results[key] = fn(y_aligned, y_pred_series)
        elif callable(metric):
            name = getattr(metric, "__name__", "custom_metric")
            results[name] = metric(y_aligned, y_pred_series)
        else:
            raise TypeError("Chaque métrique doit être un nom str ou une fonction callable.")

    return {
        "metrics": results,
        "y_true": y_aligned,
        "y_pred": y_pred_proba_series if y_pred_proba_series is not None else y_pred_series,
        "y_pred_labels": y_pred_series,
    }


def evaluer_modele_multilabel(
    model,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    seuil: float = 0.5,
    metric_fn=None
) -> dict:
    """
    Évalue un modèle multi-label.

    Parameters
    ----------
    model : sklearn model (MultiOutputClassifier, OneVsRestClassifier, etc.)
    X_test : DataFrame ou array
    y_test : DataFrame multi-label
    seuil : float
        Seuil pour transformer les probabilités en classes.
    metric_fn : callable
        Fonction de métrique prenant (y_true, y_pred).
        Par défaut : F1-score.

    Returns
    -------
    dict : métriques globales et par label
    """

    # -----------------------------
    # Définition métrique par défaut
    # -----------------------------
    if metric_fn is None:
        metric_fn = lambda a, b: f1_score(a, b, zero_division=0)

    # -----------------------------
    # Vérifications
    # -----------------------------
    if not isinstance(y_test, pd.DataFrame):
        raise ValueError("y_test doit être un DataFrame multi-label.")

    label_names = list(y_test.columns)
    y_true = y_test.values

    # -----------------------------
    # Prédictions
    # -----------------------------
    try:
        # Si predict_proba disponible → appliquer seuil
        y_proba = model.predict_proba(X_test)

        # y_proba est une liste d'array : 1 par label
        y_pred = np.column_stack([
            (prob[:, 1] >= seuil).astype(int) for prob in y_proba
        ])

    except Exception:
        # Sinon fallback sur predict()
        print("⚠️ predict_proba indisponible : utilisation de predict().")
        y_pred = model.predict(X_test)

    # -----------------------------
    # Calcul des métriques
    # -----------------------------
    metrics_per_label = {}
    for i, label in enumerate(label_names):
        metrics_per_label[label] = metric_fn(y_true[:, i], y_pred[:, i])

    # Macro-average = moyenne des labels
    macro_score = np.mean(list(metrics_per_label.values()))

    # Micro-average = vue globale
    micro_score = metric_fn(y_true.ravel(), y_pred.ravel())

    # -----------------------------
    # Résultat final
    # -----------------------------
    results = {
        "per_label": metrics_per_label,
        "macro_avg": macro_score,
        "micro_avg": micro_score,
        "y_pred": y_pred
    }

    return results




def show_metrics_binary(metrics):
    """
    Affiche l'accuracy et la matrice de confusion.
    """
    acc = metrics["accuracy"]
    cm = metrics["confusion_matrix"]

    print(f"🔎 Accuracy : {acc:.4f}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Prédit 0", "Prédit 1"],
                yticklabels=["Réel 0", "Réel 1"])
    plt.title("Matrice de confusion")
    plt.ylabel("Classe réelle")
    plt.xlabel("Classe prédite")
    plt.show()



def train_and_optimize_threshold_PR(model,X_train_sc, y_train, X_test_sc, y_test):
    """
    Entraîne un modèle, optimise le seuil de décision
    via la courbe Precision-Recall, et affiche les résultats.
    """

    # ---------------------------------------
    # 1. Entraînement du modèle (balanced)
    # ---------------------------------------
    model.fit(X_train_sc, y_train)

    # Probabilités de la classe positive
    y_proba = model.predict_proba(X_test_sc)[:, 1]


    # ----------------------------------------------------
    # 2. Courbe Precision-Recall + seuil optimal (F1 max)
    # ----------------------------------------------------
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # F1-score pour chaque seuil
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1)

    best_threshold = thresholds[best_idx]
    best_f1 = f1[best_idx]

    print(f"\n🔥 Seuil optimal (PR-F1) : {best_threshold:.3f}")
    print(f"📈 Meilleur F1-score PR : {best_f1:.4f}")

    # Prédiction avec le seuil optimal
    y_pred_opt = (y_proba >= best_threshold).astype(int)


    # -------------------------
    # 3. Affichage des métriques
    # -------------------------
    # print("\n--- Metrics ---")
    # print("Accuracy :", accuracy_score(y_test, y_pred_opt))
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred_opt))


    # ------------------------------
    # 4. Matrice de confusion
    # ------------------------------
    cm = confusion_matrix(y_test, y_pred_opt)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Prédit 0", "Prédit 1"],
                yticklabels=["Réel 0", "Réel 1"])
    plt.title("Matrice de confusion (seuil optimisé PR)")
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.show()


    # ------------------------------
    # 5. Courbe Precision-Recall
    # ------------------------------
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.scatter(recall[best_idx], precision[best_idx], color="red", s=80,
                label=f"Seuil optimal ({best_threshold:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Courbe Precision-Recall")
    plt.legend()
    plt.show()


    return model, best_threshold, y_pred_opt
