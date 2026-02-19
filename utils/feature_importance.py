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
    Affiche les top features selon SHAP pour chaque classe, en conservant le signe
    moyen des valeurs SHAP (bleu = effet positif, rouge = effet négatif).
    """
    feature_names = np.asarray(X_test.columns)
    feature_matrix = np.asarray(X_test)
    n_features = feature_names.shape[0]

    def _flatten_shap(values):
        arr = np.asarray(values)
        if arr.ndim == 1:
            if arr.size != n_features:
                raise ValueError("Dimension des valeurs SHAP incompatible avec le nombre de features.")
            return arr.reshape(1, n_features)
        if arr.ndim == 2:
            if arr.shape[1] == n_features:
                return arr
            if arr.shape[0] == n_features:
                return arr.T
        feature_axis = None
        for axis, size in enumerate(arr.shape):
            if size == n_features and (axis != 0 or arr.ndim == 1):
                feature_axis = axis
                break
        if feature_axis is None:
            for axis, size in enumerate(arr.shape):
                if size == n_features:
                    feature_axis = axis
                    break
        if feature_axis is None:
            raise ValueError("Impossible d'identifier l'axe correspondant aux features dans les valeurs SHAP.")
        arr = np.moveaxis(arr, feature_axis, -1)
        arr = arr.reshape(-1, n_features)
        return arr

    explainer = None
    if hasattr(model, "estimators_") and len(getattr(model, "estimators_", [])) > 0:
        first_est = model.estimators_[0]
        if hasattr(first_est, "tree_"):
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                explainer = None

    if explainer is None:
        try:
            explainer = shap.LinearExplainer(model, X_test)
        except Exception:
            explainer = shap.Explainer(model, X_test)

    try:
        shap_res = explainer.shap_values(X_test)
    except Exception:
        shap_res = explainer(X_test)

    if isinstance(shap_res, shap.Explanation):
        shap_arrays = [shap_res.values]
    elif isinstance(shap_res, list):
        shap_arrays = [np.asarray(values) for values in shap_res]
    else:
        shap_arrays = [np.asarray(shap_res)]

    if not isinstance(col_names, (list, tuple, np.ndarray)):
        col_names = [col_names]
    col_names = list(col_names)
    if len(col_names) < len(shap_arrays):
        col_names += [f"Classe {i}" for i in range(len(col_names), len(shap_arrays))]

    for idx, shap_array in enumerate(shap_arrays):
        flattened = _flatten_shap(shap_array)
        mean_signed = flattened.mean(axis=0)
        order = np.argsort(np.abs(mean_signed))[::-1][:top_n]
        top_features = feature_names[order]
        top_values = mean_signed[order]
        colors = ["tab:blue" if val >= 0 else "tab:red" for val in top_values]

        plt.figure(figsize=(8, 4))
        plt.barh(top_features[::-1], top_values[::-1], color=list(colors[::-1]))
        plt.title(f"Top {top_n} SHAP - {col_names[idx]}")
        plt.xlabel("SHAP moyen")
        plt.tight_layout()
        if to_save:
            os.makedirs(dir_save, exist_ok=True)
            plt.savefig(os.path.join(dir_save, f"{col_names[idx]}_shap_top{top_n}.png"))
            plt.close()
        else:
            plt.show()

        plt.figure()
        shap_for_plot = flattened
        features_arg = feature_matrix
        if features_arg.shape[0] != shap_for_plot.shape[0]:
            if features_arg.shape[0] > shap_for_plot.shape[0]:
                features_arg = features_arg[:shap_for_plot.shape[0]]
            else:
                repeats = int(np.ceil(shap_for_plot.shape[0] / features_arg.shape[0]))
                features_arg = np.tile(features_arg, (repeats, 1))[:shap_for_plot.shape[0]]
        shap.summary_plot(
            shap_for_plot,
            features_arg,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        if to_save:
            os.makedirs(dir_save, exist_ok=True)
            plt.savefig(os.path.join(dir_save, f"{col_names[idx]}_shap_summary.png"))
            plt.close()
        else:
            plt.show()
