from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _prepare_datasets(
    df_features_clean: pd.DataFrame,
    df_labels_fusion: pd.DataFrame,
    target_col: str,
    X_train: Optional[pd.DataFrame],
    X_test: Optional[pd.DataFrame],
    y_train: Optional[Sequence],
    y_test: Optional[Sequence],
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Garantit la présence de splits train/test. Si l'utilisateur ne fournit pas
    X_train/X_test/y_train/y_test, on effectue un split à partir des DataFrames.
    """
    if target_col not in df_labels_fusion.columns:
        raise ValueError(f"{target_col} est absent de df_labels_fusion.")

    provided_split = all(v is not None for v in (X_train, X_test, y_train, y_test))

    if not provided_split:
        y = df_labels_fusion[target_col].astype(int)
        X = df_features_clean.loc[y.index].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    else:
        if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
            raise TypeError("X_train et X_test doivent être des DataFrame pandas.")
        y_train = pd.Series(y_train, index=X_train.index).astype(int)
        y_test = pd.Series(y_test, index=X_test.index).astype(int)

    return X_train, X_test, y_train, y_test


def run_univariate_log_reg(
    df_features_clean: pd.DataFrame,
    df_labels_fusion: pd.DataFrame,
    target_col: str,
    *,
    X_train: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    y_train: Optional[Sequence] = None,
    y_test: Optional[Sequence] = None,
    MAIN_METRIC_NAME: str = "roc_auc",
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    top_k: int = 5,
    max_iter: int = 500,
) -> pd.DataFrame:
    """
    Lance des régressions logistiques univariées sur chaque colonne de features.

    Returns
    -------
    pd.DataFrame trié contenant au moins les colonnes :
    ['feature', MAIN_METRIC_NAME, 'coefficient', 'intercept'].
    """
    if metric_fn is None:
        from sklearn.metrics import roc_auc_score

        metric_fn = roc_auc_score

    X_train, X_test, y_train, y_test = _prepare_datasets(
        df_features_clean,
        df_labels_fusion,
        target_col,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Aucune colonne numérique disponible pour la régression logistique.")

    results: list[dict] = []

    for feature in numeric_cols:
        train_col = X_train[[feature]].copy()
        test_col = X_test[[feature]].copy()

        if train_col[feature].nunique(dropna=False) <= 1:
            # Rien à apprendre sur une colonne constante.
            continue

        median_value = train_col[feature].median()
        train_col = train_col.fillna(median_value)
        test_col = test_col.fillna(median_value)

        model = LogisticRegression(
            solver="liblinear",
            max_iter=max_iter,
        )

        try:
            model.fit(train_col, y_train)
        except ValueError:
            continue

        proba_pred = model.predict_proba(test_col)[:, 1]

        try:
            score = float(metric_fn(y_test, proba_pred))
        except Exception:
            label_pred = (proba_pred >= 0.5).astype(int)
            score = float(metric_fn(y_test, label_pred))

        results.append(
            {
                "feature": feature,
                MAIN_METRIC_NAME: score,
                "coefficient": float(model.coef_[0][0]),
                "intercept": float(model.intercept_[0]),
            }
        )

    if not results:
        raise RuntimeError("Aucun modèle univarié n'a pu être entraîné.")

    results_df = pd.DataFrame(results).sort_values(by=MAIN_METRIC_NAME, ascending=False)

    top_df = results_df.head(top_k).reset_index(drop=True)

    print(f"\nTop {min(top_k, len(results_df))} variables explicatives (metric = {MAIN_METRIC_NAME})")
    print(top_df.to_string(index=False))

    return top_df
