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

def shap_top10_per_estimator(model, X,col_names, to_save = False, dir_save = ''):
    feature_names = X.columns
    estimators = model.estimators_

    for idx, clf in enumerate(estimators):


        # SHAP explainer pour logistic regression
        explainer = shap.LinearExplainer(clf, X)
        shap_values = explainer.shap_values(X)
                
        # 1. Calcul de l’importance globale pour chaque ligne
        importance = np.abs(shap_values.values).sum(axis=1)

        # 2. Top k exemples (par ex. 5)
        k = 5
        top_idx = np.argsort(-importance)[:k]

        # 3. Waterfall plot pour chaque observation sélectionnée
        for i in top_idx:
            shap.plots.waterfall(shap_values[i])
        # Plot officiel SHAP (optionnel)

        if to_save:

            os.makedirs(dir_save, exist_ok=True)
            plt.figure()
            shap.summary_plot(shap_values, X, feature_names=feature_names,show=False)
            plt.tight_layout()
            plt.gcf().canvas.draw()
            plt.savefig(dir_save +f"/{col_names[idx]}_summary_bar.png")
            plt.close()
        else:
            print(f"\n===== SHAP pour estimateur (classe {col_names[idx]}) =====")
            plt.figure()
            shap.summary_plot(shap_values, X, feature_names=feature_names,show=False)
            plt.tight_layout()
