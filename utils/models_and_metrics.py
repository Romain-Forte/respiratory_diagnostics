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


def get_models(
    y_train,
    imbalance_threshold=0.2,
    use_catboost=True,
    multilabel=False
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
            "TabPFN" : TabPFNClassifier(
            device="cpu", ignore_pretraining_limits=True
            )



        }
        if use_catboost:
            from catboost import CatBoostClassifier

            base_models["CatBoost"] = CatBoostClassifier(
                verbose=0,
                auto_class_weights="Balanced"
            )
    else:
        print("🙂 Dataset équilibré → aucun class_weight ajouté\n")

        base_models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Random Forest": RandomForestClassifier(n_estimators=300),
            "SVM RBF": SVC(probability=True),
            "MLP Neural Net": MLPClassifier(max_iter=500),
            "Gaussian Naive Bayes": GaussianNB(),
            "XGBoost": XGBClassifier(eval_metric="logloss"),
            "TabPFN" : TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
        }
        if use_catboost:
            from catboost import CatBoostClassifier

            base_models["CatBoost"] = CatBoostClassifier(
                verbose=0,
                auto_class_weights="Balanced"
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

    return TN / (TN + FN)
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
    needs_proba: bool = False,
    metric_kwargs: dict | None = None,
    metric_name: str | None = None
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
    metric_kwargs : dict | None
        Paramètres supplémentaires passés à la fonction de métrique
        (ex: {"average": "weighted"} pour f1_score).
    metric_name : str | None
        Nom à utiliser pour la colonne des résultats. Si None, on tente de
        récupérer __name__ ou le nom de la classe du callable.

    Returns
    -------
    df_results : DataFrame
        Tableau trié des modèles selon la métrique choisie.
    """

    metric_kwargs = metric_kwargs or {}
    results = []

    for name, model in models.items():
        # print(f"\n🔄 Entraînement du modèle : {name}")

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

                proba = model.predict_proba(X_test)

                # MultiOutputClassifier renvoie une liste (une matrice par label).
                if isinstance(proba, (list, tuple)):
                    y_pred_input = np.column_stack(
                        [p[:, 1] if getattr(p, "ndim", 1) > 1 else p for p in proba]
                    )
                elif isinstance(proba, np.ndarray) and proba.ndim == 3:
                    y_pred_input = proba[:, :, 1]
                elif isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                    y_pred_input = proba[:, 1]
                else:
                    print(f"⚠️ {name} : format de probabilités inattendu → ignoré.")
                    continue

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

        score = metric_fn(y_test, y_pred_input, **metric_kwargs)
        

        # print(f"➡️ {name} : {metric_fn.__name__} = {score:.4f}")
        results.append((name, score))

    # --- Tri ---
    metric_label = (
        metric_name
        or getattr(metric_fn, "__name__", None)
        or metric_fn.__class__.__name__
    )
    df_results = pd.DataFrame(results, columns=["Modèle", metric_label])
    df_results = df_results.sort_values(metric_label, ascending=False)

    print(f"\n🏆 CLASSEMENT DES MODÈLES PAR {metric_label.upper()} :\n")
    print(df_results)

    return df_results


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
