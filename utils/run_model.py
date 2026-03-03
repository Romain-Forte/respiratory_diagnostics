
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import roc_curve
from pathlib import Path
import re
import numpy as np
import os 
from utils.algo_prediction import preparer_jeu_xy, AutoStandardScaler
from utils.models_and_metrics import get_models, get_metric, negative_predictive_value
from utils.visualisation import show_metrics_binary, show_roc_curve,plot_model_bars
from utils.analyse_sensibilite import analyse_sensibilite
from utils.feature_importance import  plot_top10_features_per_estimator,shap_top10
from utils.data_aug import get_augmentation_methods
import pandas as pd


def run_model_aug(model_name,
                  base_model,
                  augmentation_name,
                  augmentation,
                  X_train,
                  X_test,
                  y_train,
                  y_test,
                  MAIN_METRIC_NAME,
                  metric_fn,
                  needs_proba,
                  THRESHOLD,
                  target_col,
                  feature_names,
                  show_roc = False,
                  show_importance = False,
                  show_shap = False,
                  method_importance ='native_importance',
                  sensibilite = False,
                  features_sensibilite =  ["Neutropenie", "Prophylaxis_antifungal"],
                  type_sensi = 'all',
                  verbose = False):
    """Entraine un modele avec une augmentation et affiche les details demandes."""
    if verbose:
        print("----- run_model_aug -----")
        print(f"target_col={target_col}")
        print(f"model={model_name}")
        print(f"augmentation={augmentation_name}")
        print(f"metric={MAIN_METRIC_NAME} | threshold={THRESHOLD}")
        print(f"show_roc={show_roc}, show_importance={show_importance}, show_shap={show_shap}")
        print(f"sensibilite={sensibilite}, type_sensi={type_sensi}")
        print(f"features_sensibilite={features_sensibilite}")

    steps = [("scaler", AutoStandardScaler())]
    if augmentation is not None:
        steps.append(("augmentation", clone(augmentation)))
    steps.append(("model", clone(base_model)))

    pipe_train = Pipeline(steps)
    pipe_train.fit(X_train, y_train)
    pipe_inference = Pipeline([
        ("scaler", pipe_train.named_steps["scaler"]),
        ("model", pipe_train.named_steps["model"])
    ])

    if needs_proba or show_roc:
        y_pred = pipe_inference.predict_proba(X_test)[:, 1]
    else:
        y_pred = pipe_inference.predict(X_test)

    score = metric_fn(y_test, y_pred)
    if verbose:
        print(f"{model_name} | {augmentation_name} -> {MAIN_METRIC_NAME}: {score:.4f}")

    y_pred_bin = (y_pred > THRESHOLD).astype(int)
    if verbose:
        print("Negative Predictive Value:", negative_predictive_value(y_test, y_pred_bin))

    if show_roc:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        j_scores = tpr - fpr
        idx = np.argmax(j_scores)
        youden_threshold = thresholds[idx]
        show_roc_curve(
            y_test,
            y_pred,
            roc_points=(fpr, tpr, thresholds),
            highlight_threshold=youden_threshold,
            highlight_label=f"Youden = {youden_threshold:.2f}",
            highlight_color="crimson"
        )
        y_pred_bin_roc = (y_pred > youden_threshold).astype(int)
        print("Negative Predictive Value youden:", negative_predictive_value(y_test, y_pred_bin_roc), 'threshold ',youden_threshold)

    if show_importance:
        plot_top10_features_per_estimator(
            pipe_train.named_steps["model"],
            feature_names=feature_names,
            col_names=[target_col],
            method=method_importance,
            X_test=X_test,
            y_test=y_test,
            to_save=False,
            dir_save='D:/graphs_bdd/importance'
        )

    if show_shap:
        shap_top10(
            model=pipe_train.named_steps["model"],
            X_test=X_test,
            col_names=target_col,
            to_save=False,
            dir_save='D:/graphs_bdd/shap'
        )

    if sensibilite:
        if type_sensi == 'drop':
            drop_scores = {"baseline": score}
            for feature in features_sensibilite:
                if feature not in X_train.columns:
                    print(f"Feature {feature} absente du jeu de donnees, ignoree pour le drop.")
                    continue
                pipe_steps = [("scaler", AutoStandardScaler())]
                if augmentation is not None:
                    pipe_steps.append(("augmentation", clone(augmentation)))
                pipe_steps.append(("model", clone(base_model)))

                pipe_train_drop = Pipeline(pipe_steps)
                X_train_drop = X_train.drop(columns=feature)
                X_test_drop = X_test.drop(columns=feature)
                pipe_train_drop.fit(X_train_drop, y_train)
                pipe_inference_drop = Pipeline([
                    ("scaler", pipe_train_drop.named_steps["scaler"]),
                    ("model", pipe_train_drop.named_steps["model"])
                ])
                if needs_proba or show_roc:
                    y_pred_drop = pipe_inference_drop.predict_proba(X_test_drop)[:, 1]
                else:
                    y_pred_drop = pipe_inference_drop.predict(X_test_drop)
                drop_scores[feature] = metric_fn(y_test, y_pred_drop)
            plot_model_bars(drop_scores, title=f"Chute {MAIN_METRIC_NAME.upper()} lorsqu'on drop certaines colonnes ")
        else:
            analyse_sensibilite(pipe_inference, X_test, features_sensibilite, type_sensi=type_sensi)

    return {
        "score": score,
        "pipe_train": pipe_train,
        "pipe_inference": pipe_inference,
        "y_pred": y_pred
    }


def load_config_for_target(target_col, config_dir=None):
    """
    Charge le fichier config_<diagnosis>.yaml et retourne (config_dict, Path).
    """
    base_dir = Path(config_dir) if config_dir is not None else Path.cwd() / "configs"
    config_path = base_dir / f"config_{target_col}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable pour {target_col} ({config_path})."
        )

    config = {}
    with config_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value.lower() == "null":
                parsed = None
            elif value.startswith('"') and value.endswith('"'):
                parsed = value[1:-1]
            else:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
            config[key] = parsed

    return config, config_path


def run_config_for_target(target_col,
                          df_features_clean,
                          df_labels_fusion,
                          sensibilite = False,
                          show_importance = False,
                          show_roc = False,
                          show_shap = False,
                          features_sensibilite = None,
                          type_sensi = 'all',
                          method_importance = 'native_importance',
                          config_dir =  os.getcwd() + '\\configs\\',
                          condition_test = None):
    """
    Recharge la configuration enregistrée pour target_col puis relance run_model_aug.
    condition_test : bool mask / index / callable (X_test, y_test) -> mask pour filtrer le set de test.
    """
    config, config_path = load_config_for_target(target_col, config_dir=config_dir)
    model_name = config.get("model")
    augmentation_name = config.get("augmentation", "No Augmentation")
    MAIN_METRIC_NAME = config.get("main_metric", "roc_auc")
    THRESHOLD = config.get("threshold", 0.5)
    random_seed = config.get("random_seed", 42)
    if random_seed is None:
        random_seed = 42
    else:
        try:
            random_seed = int(random_seed)
        except (TypeError, ValueError):
            random_seed = 42

    if features_sensibilite is None:
        features_sensibilite = ["Neutropenie", "Prophylaxis_antifungal"]

    if not model_name:
        raise ValueError(f"Le fichier {config_path} ne contient pas de modèle.")

    df_labels_1 = df_labels_fusion[target_col].to_frame()
    X_train, X_test, y_train, y_test, labels = preparer_jeu_xy(
        df_features_clean, df_labels_1, random_state=random_seed
    )

    if condition_test is not None:
        if callable(condition_test):
            try:
                mask = condition_test(X_test)
            except TypeError:
                mask = condition_test(X_test, y_test)
        else:
            mask = condition_test

        if hasattr(mask, "reindex"):
            mask = mask.reindex(X_test.index)
        # Important: pour indexer un ndarray, il faut un masque numpy (même longueur)
        mask_np = np.asarray(mask, dtype=bool)

        X_test = X_test.loc[mask_np]

        if isinstance(y_test, pd.Series):
            y_test = y_test.loc[X_test.index]
        else:
        # y_test est ndarray/list : on applique le même masque
            y_test = np.asarray(y_test)[mask_np]

        print(f"Filtrage du jeu de test via condition_test -> {len(X_test)} échantillons.")


    all_models = get_models(y_train, use_catboost=False, imbalance_threshold=1, random_state=random_seed)
    augmentations = get_augmentation_methods(random_state=random_seed)
    metrics = get_metric()



    base_model = all_models[model_name]
    augmentation = augmentations[augmentation_name]
    metric_entry = metrics[MAIN_METRIC_NAME]

    print(f"Chargement configuration : {config_path}")
    print(
        f"Modèle = {model_name} | Augmentation = {augmentation_name} | Métrique = {MAIN_METRIC_NAME}"
    )

    return run_model_aug(
        model_name=model_name,
        base_model=base_model,
        augmentation_name=augmentation_name,
        augmentation=augmentation,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        MAIN_METRIC_NAME=MAIN_METRIC_NAME,
        metric_fn=metric_entry["metric_fn"],
        needs_proba=metric_entry["needs_proba"],
        THRESHOLD=THRESHOLD,
        target_col=target_col,
        feature_names=X_train.columns,
        show_roc=show_roc,
        show_importance=show_importance,
        show_shap=show_shap,
        method_importance=method_importance,
        sensibilite=sensibilite,
        features_sensibilite=features_sensibilite,
        type_sensi=type_sensi,
        verbose=True
    )
