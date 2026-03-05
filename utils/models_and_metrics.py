from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance
from skmultilearn.adapt import MLkNN
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    fbeta_score
)
from tabpfn import TabPFNClassifier
import os 

def get_models(
    y_train,
    imbalance_threshold=0.2,
    use_catboost=True,
    multilabel=False,
    random_state=42
):
    """
    Retourne un dictionnaire de modèles.

    Args:
        y_train: labels d'entraînement
        imbalance_threshold: seuil de déséquilibre
        use_catboost: inclure CatBoost ou non
        use_tabpfn: inclure TabPFN si disponible
        multilabel: si True, enveloppe les modèles avec MultiOutputClassifier
    """

    y = np.array(y_train)
    pos_ratio = (y == 1).mean()
    neg_ratio = (y == 0).mean()

    is_imbalanced = (pos_ratio < imbalance_threshold) or (neg_ratio < imbalance_threshold)

    print(f"\n📊 Répartition des classes : pos={pos_ratio:.3f}, neg={neg_ratio:.3f}")
    print(f"⚖️ Dataset déséquilibré ? → {is_imbalanced}")

    # ratio utile pour XGBoost
    if (y == 1).sum() > 0:
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    else:
        scale_pos_weight = 1  # fallback safe


    # ======================
    # 🔧 Modèles avec ou sans class_weight automatiquement
    # ======================

    if is_imbalanced:
        print("🧪 Activation automatique du class_weight / auto-balancing\n")

        base_models = {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=random_state
            ),

            "Random Forest": RandomForestClassifier(
                class_weight="balanced",
                n_estimators=300,
                random_state=random_state
            ),

            "SVM RBF": SVC(
                probability=True,
                class_weight="balanced",
                random_state=random_state
            ),

            "MLP Neural Net": MLPClassifier(
                max_iter=500,
                random_state=random_state
            ),  # pas de class_weight pour MLPClassifier nativement

            "Gaussian Naive Bayes": GaussianNB(),  # no class_weight

            "XGBoost": XGBClassifier(
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=random_state
            ),
            "TabPFN" : TabPFNClassifier(
            device="cpu", ignore_pretraining_limits=True
            )



        }
        if use_catboost:
            from catboost import CatBoostClassifier

            base_models["CatBoost"] = CatBoostClassifier(
                verbose=0,
                auto_class_weights="Balanced",
                random_seed=random_state
            )
    else:
        print("🙂 Dataset équilibré → aucun class_weight ajouté\n")

        base_models = {
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
            "SVM RBF": SVC(probability=True, random_state=random_state),
            "MLP Neural Net": MLPClassifier(max_iter=500, random_state=random_state),
            "Gaussian Naive Bayes": GaussianNB(),
            "XGBoost": XGBClassifier(eval_metric="logloss", random_state=random_state),
            "TabPFN" : TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
        }
        if use_catboost:
            from catboost import CatBoostClassifier

            base_models["CatBoost"] = CatBoostClassifier(
                verbose=0,
                auto_class_weights="Balanced",
                random_seed=random_state
            )
    # Envelopper avec MultiOutputClassifier si multilabel
    if multilabel:
        models = {name: MultiOutputClassifier(model) for name, model in base_models.items()}
        print("🏷️ Mode MULTILABEL activé (MultiOutputClassifier)")

    else:
        models = base_models
 
    return models

# ===========================
# MÉTRIQUES
# ===========================

def at_least_one_correct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Score multilabel :
    1 si au moins un label correct est prédit, 0 sinon.
    """
    assert y_true.shape == y_pred.shape, "Dimensions incompatibles entre y_true et y_pred."
    success = (y_true * y_pred).sum(axis=1) > 0
    return float(success.mean())


def reject_n_lowest_correct(y_true: np.ndarray, y_pred: np.ndarray, n: int = 3) -> float:
    """
    Score multilabel basé sur les plus faibles probabilités.
    """
    assert y_true.shape == y_pred.shape
    success = []

    for yt, yp in zip(y_true, y_pred):
        lowest_idx = np.argsort(yp)[:n]
        success.append(int(np.all(yt[lowest_idx] == 0)))

    return float(np.mean(success))


def negative_predictive_value(y_true, y_pred):
    """
    Calcule la Negative Predictive Value (NPV) :
    NPV = TN / (TN + FN)

    y_true : array-like, valeurs réelles (0/1)
    y_pred : array-like, prédictions (0/1)
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # True Negatives (TN)
    TN = np.sum((y_true == 0) & (y_pred == 0))

    # False Negatives (FN)
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Éviter division par zéro
    if (TN + FN) == 0:
        return 0

    return float( TN / (TN + FN))


def positive_likelihood_ratio(y_true, y_pred):
    """
    LR+ = sensitivity / (1 - specificity)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))

    sensitivity = TP / (TP + FN) if (TP + FN) else 0.0
    specificity = TN / (TN + FP) if (TN + FP) else 0.0
    denom = 1.0 - specificity

    if denom == 0:
        return float("inf")

    return float(sensitivity / denom)


def negative_likelihood_ratio(y_true, y_pred):
    """
    LR- = (1 - sensitivity) / specificity
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))

    sensitivity = TP / (TP + FN) if (TP + FN) else 0.0
    specificity = TN / (TN + FP) if (TN + FP) else 0.0

    if specificity == 0:
        return float("inf")

    return float((1.0 - sensitivity) / specificity)
# ===========================
#        FACTORY MÉTRIQUE
# ===========================

def get_metric(**kwargs):
    """
    Retourne toutes les métriques disponibles sous forme d'un dictionnaire.

    Chaque entrée contient :
        - metric_fn : callable(y_true, y_pred)
        - needs_proba : True/False (si nécessite predict_proba)

    kwargs permet de passer des paramètres comme:
        - beta pour f-beta
        - n pour reject_n_lowest_correct
    """

    beta = kwargs.get("beta", 1.0)
    n = kwargs.get("n", 3)

    metrics = {
        # =======================
        # MÉTRIQUES SUR CLASSES
        # =======================
        "accuracy": {
            "metric_fn": accuracy_score,
            "needs_proba": False
        },
        "precision": {
            "metric_fn": precision_score,
            "needs_proba": False
        },
        "recall": {
            "metric_fn": recall_score,
            "needs_proba": False
        },
        "f1": {
            "metric_fn": lambda yt, yp: f1_score(yt, yp, zero_division=0),
            "needs_proba": False
        },
        "f_beta": {
            "metric_fn": lambda yt, yp: fbeta_score(yt, yp, beta=beta, zero_division=0),
            "needs_proba": False
        },
        "negative_predictive_value": {
            "metric_fn": negative_predictive_value,
            "needs_proba": False
        },
        "lr_positive": {
            "metric_fn": positive_likelihood_ratio,
            "needs_proba": False
        },
        "lr_negative": {
            "metric_fn": negative_likelihood_ratio,
            "needs_proba": False
        },

        # =======================
        # MÉTRIQUES PROBABILITÉS
        # =======================
        "roc_auc": {
            "metric_fn": roc_auc_score,
            "needs_proba": True
        },
        "pr_auc": {
            "metric_fn": average_precision_score,
            "needs_proba": True
        },
        "logloss": {
            "metric_fn": log_loss,
            "needs_proba": True
        },
        "brier": {
            "metric_fn": brier_score_loss,
            "needs_proba": True
        },

        # =======================
        # MULTILABEL CUSTOM
        # =======================
        "at_least_one_correct": {
            "metric_fn": at_least_one_correct,
            "needs_proba": False
        },

        "reject_n_lowest_correct": {
            "metric_fn": lambda yt, yp: reject_n_lowest_correct(yt, yp, n=n),
            "needs_proba": True
        }
    }

    return metrics

def f1_metric_xgb(preds, dtrain):
    y_true = dtrain.get_label()
    y_pred = (preds > 0.5).astype(int)
    return "f1_custom", f1_score(y_true, y_pred), True

def get_models_multilabel(use_catboost=False):
    """
    Retourne un dictionnaire de modèles adaptés à la classification MULTILABEL.

    Stratégies implémentées :
    - OVR (One-vs-Rest) : XGBoost, LightGBM, Random Forest
    - BR (Binary Relevance) : Logistic Regression
    - Classifier Chains : chaîne de classifieurs dépendants
    - MLkNN : k-Nearest Neighbors multilabel
    - RAkEL : Random k-labelsets
    - Label Powerset : transformer multilabel en single-label

    Returns
    -------
    dict : {nom_modele: estimator}
    """


    models = {}

    # XGBoost OVR
    models["XGBoost OVR"] = MultiOutputClassifier(
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9
        )
    )



    # Random Forest OVR (n_estimators augmenté ; sklearn RF n'a pas de param verbose)
    models["Random Forest OVR"] = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=500,
            n_jobs=-1
        )
    )

    # Logistic Regression BR (plus d'itérations, verbose si disponible selon solver)
    models["Logistic Regression BR"] = MultiOutputClassifier(
        LogisticRegression(
            max_iter=5000,
            verbose=1
        )
    )

    # Classifier Chains (utilise une LogisticRegression avec plus d'itérations / verbose)
    models["Classifier Chains"] = ClassifierChain(
         RandomForestClassifier(
                class_weight="balanced",
                n_estimators=300
            ),
        order="random",
        cv=5
    )

    # MLkNN (skmultilearn) -- pas de verbose
    models["MLkNN"] = MLkNN(k=3)

    # RAkEL (RakelD) -- base classifier avec plus d'itérations/verbose si supporté
    models["RAkEL"] = RakelD(
        base_classifier= RandomForestClassifier(
                class_weight="balanced",
                n_estimators=300
            ),
        labelset_size=3
    )

    # Label Powerset -- base classifier avec plus d'itérations/verbose si supporté
    models["Label Powerset"] = LabelPowerset(
        classifier= RandomForestClassifier(
                class_weight="balanced",
                n_estimators=300
            ),
    )

    print(f"\n🏷️ Mode MULTILABEL spécialisé : {len(models)} modèles chargés")
    return models


def save_best_combo_config(target_col,
                           model_name,
                           augmentation_name,
                           metric_name,
                           score,
                           threshold,
                           random_seed=None,
                           filepath = None):
    """Sauvegarde les informations du meilleur combo dans un fichier config_<diagnosis>.yaml."""
    if filepath is None:
        filepath = os.getcwd() + '\\configs\\'
    filename = f"config_{target_col}.yaml"
    lines = [
        f'diagnosis: "{target_col}"',
        f'model: "{model_name}"' if model_name else 'model: null',
        f'augmentation: "{augmentation_name}"' if augmentation_name else 'augmentation: null',
        f'main_metric: "{metric_name}"',
        f'score: {score if score is not None else "null"}',
        f'threshold: {threshold}',
        f'random_seed: {random_seed if random_seed is not None else "null"}'
    ]
    with open(filepath + filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Configuration sauvegard?e dans {filename}")
