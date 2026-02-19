import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import numpy as np
import matplotlib.pyplot as plt


def mean_pred(pipeline, X, proba_class_index=None):
    """
    Moyenne des prédictions sur un dataset.

    Parameters
    ----------
    pipeline : Pipeline
    X : pd.DataFrame
    proba_class_index : int | None
        None -> predict() ; sinon -> predict_proba()[:, index]

    Returns
    -------
    float
    """
    if proba_class_index is None:
        return pipeline.predict(X).mean()
    return pipeline.predict_proba(X)[:, proba_class_index].mean()

def force_group_all_zero(X, group_cols):
    """
    Force un groupe de colonnes à 0 pour toutes les lignes.

    Parameters
    ----------
    X : pd.DataFrame
    group_cols : list[str]

    Returns
    -------
    pd.DataFrame
    """
    X2 = X.copy()
    X2.loc[:, group_cols] = 0
    return X2

def analyse_sensibilite(pipeline,X_test,features,type_sensi= 'all'):

    if  type_sensi == 'any':
        
        baseline = mean_pred(pipeline, X_test, proba_class_index=1)  # mets None si régression
        all_zero = mean_pred(pipeline, force_group_all_zero(X_test, features), proba_class_index=1)
        
        print("Baseline :", baseline)
        print("Force all zero :", all_zero, "Delta =", all_zero - baseline)
    elif type_sensi == 'all':
        disp = PartialDependenceDisplay.from_estimator(
            estimator=pipeline,
            X=X_test,
            features=features,
            kind="both",
            grid_resolution=50,
        )

        # --- Titre global ---
        disp.figure_.suptitle(
            "Analyse de sensibilité (Partial Dependence)",
            fontsize=14
        )

        # --- Personnalisation de chaque sous-graphe ---
        for ax, feature in zip(disp.axes_.ravel(), features):

            ax.set_title(f"Effet de : {feature}", fontsize=11)

            ax.set_xlabel(f"Valeur de {feature}")
            ax.set_ylabel("Prédiction moyenne du modèle")

            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    else:
        raise NameError("Features is not of a list or str type")