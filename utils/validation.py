from utils.model_saving import save_model, load_model
from utils.models_and_metrics import  get_metric,negative_predictive_value
from utils.visualisation import  show_roc_curve
from sklearn.metrics import roc_curve
from utils.feature_importance import  plot_top_odds_ratios
from utils.run_model import _slugify, find_best_model_and_aug_from_split
import os 
import numpy as np
from pathlib import Path
import json

def validation_save(diagnostique,
                    save_dir,
                    loaded ,
                    grroh_features ,
                    grroh_diag ,
                    df_features_clean ,
                    df_labels_fusion 
                    ):
    
    pipe_inference = loaded["pipe_inference"]
    threshold = loaded["Youden_threshold"]
    columns = pipe_inference.named_steps["scaler"].colonnes_numeriques
    X_test = grroh_features[columns]
    y_test = grroh_diag[diagnostique]
    y_test_array = np.asarray(y_test)
    proba_raw = pipe_inference.predict_proba(X_test)

    metrics = get_metric()
    y_test_array = np.asarray(y_test)
    y_pred_proba = None
    try:
        proba_raw = pipe_inference.predict_proba(X_test)
    except AttributeError:
        proba_raw = None
    if y_pred_proba is not None:
        y_pred_discrete = (y_pred_proba > threshold).astype(int)
    else:
        y_pred_discrete = pipe_inference.predict(X_test)

    if proba_raw is not None:
        if isinstance(proba_raw, list):
            processed = []
            for arr in proba_raw:
                arr_np = np.asarray(arr)
                if arr_np.ndim == 2 and arr_np.shape[1] > 1:
                    processed.append(arr_np[:, 1])
                else:
                    processed.append(arr_np.ravel())
            try:
                y_pred_proba = np.column_stack(processed)
            except ValueError:
                y_pred_proba = None
        else:
            proba_np = np.asarray(proba_raw)
            if proba_np.ndim == 2 and proba_np.shape[1] > 1:
                y_pred_proba = proba_np[:, 1]
            else:
                y_pred_proba = proba_np.ravel()
    target_save_dir = save_dir + _slugify(diagnostique) + r"\validation\\"
    target_save_dir = Path(target_save_dir)
    target_save_dir.mkdir(parents=True, exist_ok=True)



    # ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    j_scores = tpr - fpr
    idx = np.argmax(j_scores)
    youden_threshold = thresholds[idx]
    show_roc_curve(
        y_test,
        y_pred_proba,
        roc_points=(fpr, tpr, thresholds),
        highlight_threshold=youden_threshold,
        highlight_label=f"Youden = {youden_threshold:.2f}",
        highlight_color="crimson",
        save_path=str((target_save_dir / "roc_curve.png")) if target_save_dir is not None else None
    )
    y_pred_bin_roc = (y_pred_proba > youden_threshold).astype(int)
    print("Negative Predictive Value youden:", negative_predictive_value(y_test, y_pred_bin_roc), 'threshold ',youden_threshold)
    # importance
    odds_dir = str(target_save_dir ) if target_save_dir is not None else ""
    plot_top_odds_ratios(
        X_test,
        y_test,
        feature_names=X_test.columns,
        top_n=10,
        ridge_alpha=1.0,
        n_bootstrap=500,
        random_state=None,
        to_save=True,
        dir_save=odds_dir,
        title=f"Top 10 odds ratios for {diagnostique}",
    )
        
    # metrics

    metric_scores = {}
    for metric_name, metric_info in metrics.items():
        use_proba = metric_info["needs_proba"]
        if use_proba and y_pred_proba is None:
            print(f"Impossible de calculer {metric_name} (predict_proba indisponible).")
            continue
        preds_input = y_pred_proba if use_proba else y_pred_discrete
        metric_scores[metric_name] = metric_info["metric_fn"](y_test_array, preds_input)
    best_summary_path = target_save_dir / "metrics_score.json"
    with best_summary_path.open("w", encoding="utf-8") as handle:
            json.dump(metric_scores, handle, indent=2)

    X_test = grroh_features[df_features_clean.columns]
    y_test = grroh_diag[diagnostique]
    MODEL_NAMES = ['Logistic Regression', 'Random Forest', 'Gaussian Naive Bayes', 'XGBoost',"CatBoost"]

    best = find_best_model_and_aug_from_split(
        X_train=df_features_clean,
        X_test=X_test,
        y_train=df_labels_fusion[diagnostique],
        y_test=y_test_array,
        target_col=diagnostique,
        MODEL_NAMES=MODEL_NAMES,
        MAIN_METRIC_NAME="roc_auc",
        montecarlo=10,
        write_config=False,
        to_save=True,
        save_dir=target_save_dir
    )