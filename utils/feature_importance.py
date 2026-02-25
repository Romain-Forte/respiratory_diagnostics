import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance


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


def plot_top10_features_per_estimator(model, 
                                      feature_names, 
                                      col_names,
                                      method='native_importance', 
                                      top_n=10, 
                                      to_save=False, 
                                      dir_save="",
                                      X_test = None,
                                      y_test = None):
    """
    Affiche (ou sauvegarde) les top features pour chaque estimateur.
    Compatible modèles OneVsRest (.estimators_) et modèles simples (ex: XGBClassifier).
    """

    estimators = [model]
    for idx, clf in enumerate(estimators):
        if method == 'native_importance':
            feat_names, coefs = _get_importances(clf, feature_names)
        elif method == 'permutation':
            if X_test is None or y_test is None:
                raise ValueError("X_test et y_test doivent être fournis pour l'importance par permutation.")
            result = permutation_importance(
                    clf, X_test, y_test, n_repeats=10, random_state=42
                )
            coefs = result.importances_mean
            feat_names = np.asarray(X_test.columns)
        else:
            raise KeyError('Method not supported, only permutation and native_importance are supported currently')
        abs_coefs = np.abs(coefs)
        top_idx = np.argsort(abs_coefs)[::-1][:top_n]
        top_features = feat_names[top_idx]
        top_values = coefs[top_idx]
        colors = np.where(top_values >= 0, "tab:blue", "tab:red")

        plt.figure(figsize=(8, 4))
        plt.barh(top_features[::-1], top_values[::-1], color=colors[::-1])
        plt.title(f"Top {top_n} features - Estimateur {col_names[idx]}")
        plt.xlabel("Importance (mean decrease of impurity)")
        plt.tight_layout()

        if to_save:
            os.makedirs(dir_save, exist_ok=True)
            plt.gcf().canvas.draw()
            plt.savefig(os.path.join(dir_save, f"{col_names[idx]}_summary_bar.png"))
            plt.close()
        else:
            plt.show()


def shap_top10(model,
               X_test,
               col_names,
               top_n=10,
               to_save=False,
               dir_save="",
               max_display=15):
    """
    Affiche les top features selon SHAP pour chaque sortie du modèle.
    Flux simplifié : shap.Explainer -> bar chart + summary plot par classe.
    """
    feature_names = np.asarray(X_test.columns)
    features_matrix = np.asarray(X_test)

    try:
        explainer = shap.Explainer(model, X_test)
    except Exception:
        explainer = shap.TreeExplainer(model)

    try:
        shap_values = explainer(X_test)
        values = np.asarray(shap_values.values)
        data = np.asarray(shap_values.data)
    except Exception:
        values = np.asarray(explainer.shap_values(X_test))
        data = features_matrix

    if values.ndim == 2:
        value_mats = [values]
    elif values.ndim == 3:
        value_mats = [values[:, i, :] for i in range(values.shape[1])]
    else:
        value_mats = [values.reshape(values.shape[0], -1)]

    if isinstance(col_names, (str, int)):
        col_names = [col_names]
    col_names = list(col_names)
    if len(col_names) < len(value_mats):
        col_names += [f"Classe {i}" for i in range(len(col_names), len(value_mats))]

    if to_save:
        os.makedirs(dir_save, exist_ok=True)

    for idx, shap_matrix in enumerate(value_mats):
        mean_signed = shap_matrix.mean(axis=0)
        order = np.argsort(np.abs(mean_signed))[::-1][:top_n]
        top_features = feature_names[order]
        top_values = mean_signed[order]
        colors = ["tab:blue" if val >= 0 else "tab:red" for val in top_values]

        plt.figure(figsize=(8, 4))
        plt.barh(top_features[::-1], top_values[::-1], color=colors[::-1])
        plt.title(f"Top {top_n} SHAP - {col_names[idx]}")
        plt.xlabel("SHAP moyen")
        plt.tight_layout()
        if to_save:
            plt.savefig(os.path.join(dir_save, f"{col_names[idx]}_shap_top{top_n}.png"))
            plt.close()
        else:
            plt.show()

        summary_features = (
            data[:, idx, :]
            if data.ndim == 3 and data.shape[1] == len(value_mats)
            else features_matrix
        )
        plt.figure()
        shap.summary_plot(
            shap_matrix,
            summary_features,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        if to_save:
            plt.savefig(os.path.join(dir_save, f"{col_names[idx]}_shap_summary.png"))
            plt.close()
        else:
            plt.show()
