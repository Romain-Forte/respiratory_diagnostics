from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

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

def get_models(y_train,imbalance_threshold=0.2):

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

        models = {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced",
                max_iter=2000
            ),

            "Random Forest": RandomForestClassifier(
                class_weight="balanced",
                n_estimators=300
            ),

            "SVM RBF": SVC(
                probability=True,
                class_weight="balanced"
            ),

            "MLP Neural Net": MLPClassifier(
                max_iter=500
            ),  # pas de class_weight pour MLPClassifier nativement

            "Gaussian Naive Bayes": GaussianNB(),  # no class_weight

            "XGBoost": XGBClassifier(
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight
            ),

            "LightGBM": LGBMClassifier(
                class_weight="balanced"
            ),

            "CatBoost": CatBoostClassifier(
                verbose=0,
                auto_class_weights="Balanced"
            ),

        }

    else:
        print("🙂 Dataset équilibré → aucun class_weight ajouté\n")

        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Random Forest": RandomForestClassifier(n_estimators=300),
            "SVM RBF": SVC(probability=True),
            "MLP Neural Net": MLPClassifier(max_iter=500),
            "Gaussian Naive Bayes": GaussianNB(),
            "XGBoost": XGBClassifier(eval_metric="logloss"),
            "LightGBM": LGBMClassifier(),
            "CatBoost": CatBoostClassifier(verbose=0)
        }

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

def compare_models_metric(
    models: dict,
    metric_fn,
    X_train,
    y_train,
    X_test,
    y_test,
    needs_proba: bool = False
):
    """
    Compare plusieurs modèles selon une métrique donnée.

    Parameters
    ----------
    models : dict
        Dictionnaire : {"nom": modele}
    metric_fn : callable
        Fonction de métrique sklearn ou perso. Ex: f1_score(y, y_pred)
    X_train, y_train : training set
    X_test, y_test : test set
    needs_proba : bool
        True si la métrique doit recevoir des probabilités (ex: roc_auc_score)
        False si la métrique utilise des classes (ex: f1_score)

    Returns
    -------
    df_results : DataFrame
        Tableau trié des modèles selon la métrique choisie.
    """

    results = []

    for name, model in models.items():
        print(f"\n🔄 Entraînement du modèle : {name}")

        # --- Entraînement ---
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"⚠️ {name} : erreur during training → ignoré.")
            print(e)
            continue

        # --- Prédiction ---
        try:
            if needs_proba:
                # On attend des probabilités
                if not hasattr(model, "predict_proba"):
                    print(f"⚠️ {name} : predict_proba manquant → ignoré.")
                    continue

                y_pred_input = model.predict_proba(X_test)[:, 1]

            else:
                # On attend des classes
                if hasattr(model, "predict"):
                    y_pred_input = model.predict(X_test)
                else:
                    print(f"⚠️ {name} : predict manquant → ignoré.")
                    continue

        except Exception as e:
            print(f"⚠️ {name} : erreur prediction → ignoré.")
            print(e)
            continue

        # --- Calcul métrique ---
        try:
            score = metric_fn(y_test, y_pred_input)
        except Exception as e:
            print(f"⚠️ {name} : erreur calcul métrique → ignoré.")
            print(e)
            continue

        print(f"➡️ {name} : {metric_fn.__name__} = {score:.4f}")
        results.append((name, score))

    # --- Tri ---
    df_results = pd.DataFrame(results, columns=["Modèle", metric_fn.__name__])
    df_results = df_results.sort_values(metric_fn.__name__, ascending=False)

    print(f"\n🏆 CLASSEMENT DES MODÈLES PAR {metric_fn.__name__.upper()} :\n")
    print(df_results)

    return df_results
