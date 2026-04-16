import json

import numpy as np

from utils.feature_importance import plot_top_odds_ratios
from utils.models_and_metrics import get_metric, negative_predictive_value
from utils.run_model import (
    _compute_binary_outputs,
    _compute_metric_scores,
    _compute_youden_threshold,
    _resolve_target_save_dir,
    find_best_model_and_aug_from_split,
)
from utils.visualisation import show_roc_curve


DEFAULT_MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "Gaussian Naive Bayes",
    "XGBoost",
    "CatBoost",
]


def _save_validation_metrics(target_save_dir, metric_scores):
    metrics_path = target_save_dir / "metrics_score.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metric_scores, handle, indent=2)


def validation_save(diagnostique,
                    save_dir,
                    loaded,
                    grroh_features,
                    grroh_diag,
                    df_features_clean,
                    df_labels_fusion):
    """
    Lance la validation externe et sauvegarde les sorties dans un dossier dedie.
    """
    pipe_inference = loaded["pipe_inference"]
    threshold = loaded["Youden_threshold"]
    feature_columns = pipe_inference.named_steps["scaler"].colonnes_numeriques

    X_validation = grroh_features[feature_columns]
    y_validation = grroh_diag[diagnostique]

    target_save_dir = _resolve_target_save_dir(save_dir, diagnostique, text_save="validation")
    y_pred_proba, y_pred_discrete = _compute_binary_outputs(pipe_inference, X_validation, threshold)

    if y_pred_proba is None:
        raise AttributeError("La validation externe requiert un modele avec predict_proba.")

    youden_threshold, roc_points = _compute_youden_threshold(y_validation, y_pred_proba)
    show_roc_curve(
        y_validation,
        y_pred_proba,
        roc_points=roc_points,
        highlight_threshold=youden_threshold,
        highlight_label=f"Youden = {youden_threshold:.2f}",
        highlight_color="crimson",
        save_path=str(target_save_dir / "roc_curve.png"),
        model_name=loaded.get("model_name")
    )

    y_pred_bin_roc = (y_pred_proba > youden_threshold).astype(int)
    print(
        "Negative Predictive Value youden:",
        negative_predictive_value(y_validation, y_pred_bin_roc),
        "threshold",
        youden_threshold
    )

    plot_top_odds_ratios(
        X_validation,
        y_validation,
        feature_names=X_validation.columns,
        top_n=10,
        ridge_alpha=1.0,
        n_bootstrap=500,
        random_state=None,
        to_save=True,
        dir_save=str(target_save_dir),
        title=f"Top 10 odds ratios for {diagnostique}",
    )

    metric_scores = _compute_metric_scores(
        get_metric(),
        np.asarray(y_validation),
        y_pred_proba,
        y_pred_discrete
    )
    _save_validation_metrics(target_save_dir, metric_scores)

    X_model_search = grroh_features[df_features_clean.columns]
    y_model_search = grroh_diag[diagnostique]

    return find_best_model_and_aug_from_split(
        X_train=df_features_clean,
        X_test=X_model_search,
        y_train=df_labels_fusion[diagnostique],
        y_test=np.asarray(y_model_search),
        target_col=diagnostique,
        MODEL_NAMES=DEFAULT_MODEL_NAMES,
        MAIN_METRIC_NAME="roc_auc",
        montecarlo=10,
        write_config=False,
        to_save=True,
        save_dir=target_save_dir
    )
