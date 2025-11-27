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

        # SHAP explainer pour logistic regression
        explainer = shap.LinearExplainer(clf, X)
        shap_values = explainer.shap_values(X)
                
        # 1. Calcul de l’importance globale pour chaque ligne
        importance = np.abs(shap_values.values).sum(axis=1)

<<<<<<< HEAD
        # 2. Top k exemples (par ex. 5)
        k = 5
        top_idx = np.argsort(-importance)[:k]

        # 3. Waterfall plot pour chaque observation sélectionnée
        for i in top_idx:
            shap.plots.waterfall(shap_values[i])
        # Plot officiel SHAP (optionnel)
=======
        # Récupération du vecteur shap pour la classe (logistic regression → 1 vecteur)
        shap_matrix = np.array(shap_values)
>>>>>>> 305815bb2c7e8079d5deb225bcbcd08600f58ba3

        # Sélection des 5 observations les plus extrêmes
        magnitudes = np.abs(shap_matrix).sum(axis=1)
        top5_idx = np.argsort(magnitudes)[-5:][::-1]  # top 5 décroissant

        # ----- Summary plot -----
        if to_save:
            os.makedirs(dir_save, exist_ok=True)
            plt.figure()
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(f"{dir_save}/{col_names[idx]}_summary_bar.png")
            plt.close()
        else:
            plt.figure()
            shap.summary_plot(shap_values, X, feature_names=feature_names)
            plt.tight_layout()

        # ----- Waterfall plots pour 5 cas extrêmes -----
        for rank, obs_id in enumerate(top5_idx):
            print(f"Waterfall #{rank+1} — Index observation : {obs_id}")

            # SHAP values pour une observation donnée
            shap_single = shap_values[obs_id]

            # Diagramme waterfall
            plt.figure()
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value,
                shap_single,
                feature_names=feature_names,
                max_display=15
            )

            if to_save:
                plt.savefig(f"{dir_save}/{col_names[idx]}_waterfall_{rank+1}.png")
                plt.close()
            else:
                plt.show()