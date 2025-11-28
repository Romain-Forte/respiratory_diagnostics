import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os

def plot_top10_features_per_estimator(model, feature_names, col_names, top_n=10,to_save = False, dir_save = ''):
    estimators = model.estimators_
    n_classes = len(estimators)

    for idx, clf in enumerate(estimators):
        coefs = clf.coef_.ravel()
        abs_coefs = np.abs(coefs)

        # top features indices
        top_idx = np.argsort(abs_coefs)[::-1][:top_n]

        top_features = feature_names[top_idx]
        top_values = abs_coefs[top_idx]

        # plot

        if to_save:
            plt.figure(figsize=(8, 4))
            plt.barh(top_features[::-1], top_values[::-1])
            plt.title(f"Top {top_n} features — Estimateur classe {col_names[idx]}")
            plt.xlabel("Importance (|coefficient|)")
            plt.tight_layout()
            os.makedirs(dir_save, exist_ok=True)
            plt.gcf().canvas.draw()
            plt.savefig(dir_save +f"/{col_names[idx]}_summary_bar.png")
            plt.close()
        else:
            plt.figure(figsize=(8, 4))
            plt.barh(top_features[::-1], top_values[::-1])
            plt.title(f"Top {top_n} features — Estimateur classe {col_names[idx]}")
            plt.xlabel("Importance (|coefficient|)")
            plt.tight_layout()
            plt.show()
# Exemple d'utilisation :
# plot_top10_features_per_estimator(model, X.columns)


def shap_top10_per_estimator(model, X, col_names, to_save=False, dir_save=''):
    feature_names = X.columns
    estimators = model.estimators_

    for idx, clf in enumerate(estimators):

        print(f"\n===== SHAP pour estimateur (classe {col_names[idx]}) =====")

        # create a SHAP explainer (try LinearExplainer, fallback to generic Explainer)
        try:
            explainer = shap.LinearExplainer(clf, X)
        except Exception:
            explainer = shap.Explainer(clf, X)

        # obtain SHAP values in a robust way and ensure we have a numpy array
        try:
            shap_res = explainer.shap_values(X)
        except Exception:
            shap_res = explainer(X)

        # shap_res can be an Explanation, array or list (multiclass); normalize to ndarray
        if hasattr(shap_res, "values"):
            shap_vals = shap_res.values
        else:
            shap_vals = shap_res

        # if the explainer returned a list (e.g., multiclass), pick the first element
        if isinstance(shap_vals, list) and len(shap_vals) > 0:
            shap_array = np.asarray(shap_vals[0])
        else:
            shap_array = np.asarray(shap_vals)

        # 1. Calcul de l’importance globale pour chaque ligne (somme des valeurs absolues des features)
        importance = np.abs(shap_array).sum(axis=1)

        # 2. Top k exemples (par ex. 5)
        k = 5
        top_idx = np.argsort(-importance)[:k]

        # 3. Waterfall plot pour chaque observation sélectionnée
        for i in top_idx:
            try:
                # prefer the new Explanation-based waterfall when possible
                base_val = getattr(explainer, "expected_value", None)
                expl_shap = shap.Explanation(values=shap_array[i], base_values=base_val, data=X.iloc[i].values, feature_names=feature_names)
                shap.plots.waterfall(expl_shap)
            except Exception:
                # fallback to direct call if needed
                try:
                    shap.plots.waterfall(shap_array[i])
                except Exception:
                    pass

        # Sélection des 5 observations les plus extrêmes (mêmes que importance)
        magnitudes = importance
        top5_idx = np.argsort(magnitudes)[-5:][::-1]  # top 5 décroissant

        # ----- Summary plot -----
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

        # ----- Waterfall plots pour 5 cas extrêmes -----
        for rank, obs_id in enumerate(top5_idx):
            print(f"Waterfall #{rank+1} — Index observation : {obs_id}")

            # SHAP values pour une observation donnée
            shap_single = shap_array[obs_id]

            # Diagramme waterfall (use legacy API but guard for types)
            plt.figure()
            base_val = getattr(explainer, "expected_value", None)
            if isinstance(base_val, (list, np.ndarray)):
                # choose first base value if an array is returned
                base_val = np.asarray(base_val).ravel()[0] if np.asarray(base_val).size > 0 else None

            try:
                shap.plots._waterfall.waterfall_legacy(
                    base_val,
                    shap_single,
                    feature_names=feature_names,
                    max_display=15
                )
            except Exception:
                try:
                    expl_shap = shap.Explanation(values=shap_single, base_values=base_val, data=X.iloc[obs_id].values, feature_names=feature_names)
                    shap.plots.waterfall(expl_shap)
                except Exception:
                    pass

            if to_save:
                plt.savefig(os.path.join(dir_save, f"{col_names[idx]}_waterfall_{rank+1}.png"))
                plt.close()
            else:
                plt.show()