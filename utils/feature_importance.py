import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def _ensure_estimators(model):
    """
    Retourne une liste d'estimateurs à partir d'un modèle.
    - Si le modèle possède .estimators_ (ex: OneVsRestClassifier), on les utilise.
    - Sinon on retourne [model] pour gérer les modèles simples (ex: XGBClassifier).
    """
    if hasattr(model, "estimators_"):
        try:
            return list(model.estimators_)
        except Exception:
            return model.estimators_
    return [model]


def _get_importances(estimator, feature_names):
    """
    Récupère un vecteur d'importance de features depuis différents types de modèles.
    - coef_ (modèles linéaires)
    - feature_importances_ (arbres / xgboost)
    """
    if hasattr(estimator, "coef_"):
        coefs = np.ravel(estimator.coef_)
    elif hasattr(estimator, "feature_importances_"):
        coefs = np.ravel(estimator.feature_importances_)
    else:
        raise AttributeError("L'estimateur ne possède ni coef_ ni feature_importances_.")
    feature_names = np.asarray(feature_names)
    return feature_names, coefs


def plot_top10_features_per_estimator(model, feature_names, col_names, top_n=10, to_save=False, dir_save=""):
    """
    Affiche (ou sauvegarde) les top features pour chaque estimateur.
    Compatible modèles OneVsRest (.estimators_) et modèles simples (ex: XGBClassifier).
    """
    estimators = _ensure_estimators(model)

    for idx, clf in enumerate(estimators):
        feat_names, coefs = _get_importances(clf, feature_names)
        abs_coefs = np.abs(coefs)

        top_idx = np.argsort(abs_coefs)[::-1][:top_n]
        top_features = feat_names[top_idx]
        top_values = abs_coefs[top_idx]

        plt.figure(figsize=(8, 4))
        plt.barh(top_features[::-1], top_values[::-1])
        plt.title(f"Top {top_n} features - Estimateur {col_names[idx]}")
        plt.xlabel("Importance (|coefficient|)")
        plt.tight_layout()

        if to_save:
            os.makedirs(dir_save, exist_ok=True)
            plt.gcf().canvas.draw()
            plt.savefig(os.path.join(dir_save, f"{col_names[idx]}_summary_bar.png"))
            plt.close()
        else:
            plt.show()


def shap_top10_per_estimator(model, X, col_names, to_save=False, dir_save=""):
    """
    SHAP pour chaque estimateur ; fonctionne aussi si le modèle n'a pas estimators_ (ex: XGBClassifier).
    """
    feature_names = X.columns
    estimators = _ensure_estimators(model)

    for idx, clf in enumerate(estimators):
        print(f"\n===== SHAP pour estimateur (classe {col_names[idx]}) =====")

        try:
            explainer = shap.LinearExplainer(clf, X)
        except Exception:
            explainer = shap.Explainer(clf, X)

        try:
            shap_res = explainer.shap_values(X)
        except Exception:
            shap_res = explainer(X)

        if hasattr(shap_res, "values"):
            shap_vals = shap_res.values
        else:
            shap_vals = shap_res

        if isinstance(shap_vals, list) and len(shap_vals) > 0:
            shap_array = np.asarray(shap_vals[0])
        else:
            shap_array = np.asarray(shap_vals)

        importance = np.abs(shap_array).sum(axis=1)
        k = 5
        top_idx = np.argsort(-importance)[:k]

        for i in top_idx:
            try:
                base_val = getattr(explainer, "expected_value", None)
                expl_shap = shap.Explanation(
                    values=shap_array[i],
                    base_values=base_val,
                    data=X.iloc[i].values,
                    feature_names=feature_names,
                )
                shap.plots.waterfall(expl_shap)
            except Exception:
                try:
                    shap.plots.waterfall(shap_array[i])
                except Exception:
                    pass

        magnitudes = importance
        top5_idx = np.argsort(magnitudes)[-5:][::-1]

        if to_save:
            os.makedirs(dir_save, exist_ok=True)
            plt.figure()
            shap.summary_plot(shap_array, X, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(dir_save, f"{col_names[idx]}_summary_bar.png"))
            plt.close()
        else:
            plt.figure()
            shap.summary_plot(shap_array, X, feature_names=feature_names)
            plt.tight_layout()

        for rank, obs_id in enumerate(top5_idx):
            print(f"Waterfall #{rank+1} - Index observation : {obs_id}")
            shap_single = shap_array[obs_id]
            plt.figure()
            base_val = getattr(explainer, "expected_value", None)
            if isinstance(base_val, (list, np.ndarray)):
                base_val = np.asarray(base_val).ravel()[0] if np.asarray(base_val).size > 0 else None

            try:
                shap.plots._waterfall.waterfall_legacy(
                    base_val,
                    shap_single,
                    feature_names=feature_names,
                    max_display=15,
                )
            except Exception:
                try:
                    expl_shap = shap.Explanation(
                        values=shap_single,
                        base_values=base_val,
                        data=X.iloc[obs_id].values,
                        feature_names=feature_names,
                    )
                    shap.plots.waterfall(expl_shap)
                except Exception:
                    pass

            if to_save:
                plt.savefig(os.path.join(dir_save, f"{col_names[idx]}_waterfall_{rank+1}.png"))
                plt.close()
            else:
                plt.show()
