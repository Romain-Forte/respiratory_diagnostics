import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Sequence, Union


class _HematoPDPProxy:
    """
    Estimator wrapper that maps a synthetic 'hemato' feature to the underlying
    group of binary columns before delegating predictions to the original
    pipeline.
    """

    def __init__(self, base_estimator, reference_X, group_cols: Sequence[str]):
        self.base_estimator = base_estimator
        self.reference_X = reference_X.copy()
        self.group_cols = list(group_cols)
        self.feature_names_in_ = np.array(["hemato"])
        self.n_features_in_ = 1
        if hasattr(base_estimator, "classes_"):
            self.classes_ = base_estimator.classes_

    def _expand_matrix(self, hemato_values: Union[pd.DataFrame, pd.Series, np.ndarray]):
        if isinstance(hemato_values, pd.DataFrame):
            values = hemato_values.iloc[:, 0].to_numpy(dtype=float)
        elif isinstance(hemato_values, pd.Series):
            values = hemato_values.to_numpy(dtype=float)
        else:
            values = np.asarray(hemato_values, dtype=float)

        values = np.clip(np.rint(values), 0, 1)

        if values.ndim == 2:
            # Flatten shapes like (n_samples, 1)
            values = values.reshape(-1)

        if values.shape[0] != len(self.reference_X):
            raise ValueError(
                f"IncohÃ©rence des dimensions pour hemato ({values.shape[0]}) "
                f"vs X de rÃ©fÃ©rence ({len(self.reference_X)})."
            )

        X_mod = self.reference_X.copy()
        for col in self.group_cols:
            X_mod.loc[:, col] = values
        return X_mod

    def _delegate(self, method_name, X):
        if not hasattr(self.base_estimator, method_name):
            raise AttributeError(f"L'estimateur ne possÃ¨de pas {method_name}.")
        X_full = self._expand_matrix(X)
        method = getattr(self.base_estimator, method_name)
        return method(X_full)

    def predict(self, X):
        return self._delegate("predict", X)

    def predict_proba(self, X):
        return self._delegate("predict_proba", X)

    def decision_function(self, X):
        return self._delegate("decision_function", X)

    def __getattr__(self, item):
        return getattr(self.base_estimator, item)


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

def analyse_sensibilite(pipeline,X_test,features,type_sensi= 'all', save_path=None):

    if  type_sensi == 'any':
        
        baseline = mean_pred(pipeline, X_test, proba_class_index=1)  # mets None si régression
        all_zero = mean_pred(pipeline, force_group_all_zero(X_test, features), proba_class_index=1)
        
        print("Baseline :", baseline)
        print("Force all zero :", all_zero, "Delta =", all_zero - baseline)
    elif type_sensi == 'all':
        plot_X = X_test
        plot_features = features
        plot_estimator = pipeline
        grid_resolution = 20
        custom_values = None

        if isinstance(features, (list, tuple)):
            if len(features) == 0:
                raise ValueError("La liste de features ne peut pas être vide.")
            if not isinstance(X_test, pd.DataFrame):
                raise TypeError("X_test doit être un DataFrame pour créer hemato.")
            missing = [col for col in features if col not in X_test.columns]
            if missing:
                raise ValueError(f"Colonnes manquantes pour hemato: {missing}")

            hemato_series = (X_test[list(features)] > 0).all(axis=1).astype(int)
            plot_X = hemato_series.to_frame(name="hemato")
            plot_features = ["hemato"]
            plot_estimator = _HematoPDPProxy(pipeline, X_test, list(features))
            grid_resolution = 2
            custom_values = {0: np.array([0.0, 1.0])}
        elif isinstance(features, str):
            plot_features = [features]

        pd_line_style = {"color": "#1f77b4", "linewidth": 2.0}
        ice_line_style = {"color": "#e67e22", "linewidth": 1.0, "alpha": 0.8}

        disp = PartialDependenceDisplay.from_estimator(
            estimator=plot_estimator,
            X=plot_X,
            features=plot_features,
            kind="both",
            grid_resolution=grid_resolution,
            custom_values=custom_values,
            subsample=None,
            pd_line_kw=pd_line_style,
            ice_lines_kw=ice_line_style,
        )

        # --- Titre global ---
        disp.figure_.suptitle(
            "Analyse de sensibilité (Partial Dependence)",
            fontsize=14
        )

        # --- Personnalisation de chaque sous-graphe ---
        for ax, feature in zip(disp.axes_.ravel(), plot_features):
            ax.set_title(f"Effet de : {feature}", fontsize=11)
            ax.set_xlabel(f"Valeur de {feature}")
            ax.set_ylabel("Prédiction moyenne du modèle")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            disp.figure_.savefig(save_path, bbox_inches="tight")
        plt.show()
    else:
        raise NameError("Features is not of a list or str type")
