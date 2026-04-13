import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import roc_curve

from utils.algo_prediction import AutoStandardScaler, preparer_jeu_xy
from utils.analyse_sensibilite import analyse_sensibilite
from utils.data_aug import get_augmentation_methods
from utils.feature_importance import plot_top_odds_ratios, shap_top10
from utils.models_and_metrics import get_metric, get_models, negative_predictive_value, save_best_combo_config
from utils.visualisation import plot_model_bars, show_roc_curve
from utils.calibration import CalibratedModel


# ---------------------------------------------------------------------------
# Path and naming helpers
# ---------------------------------------------------------------------------

def _slugify(value):
    clean = "".join(char if char.isalnum() else "_" for char in (value or ""))
    clean = clean.strip("_")
    return clean or "target"


def _resolve_target_save_dir(save_dir, target_col, text_save=None):
    if not save_dir:
        raise ValueError("save_dir doit etre fourni lorsque to_save=True.")
    target_dir = Path(save_dir).expanduser() / _slugify(target_col)
    if text_save:
        parts = [part for part in str(text_save).replace("\\", "/").split("/") if part]
        if parts:
            target_dir = target_dir.joinpath(*parts)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


# ---------------------------------------------------------------------------
# Prediction and metric helpers
# ---------------------------------------------------------------------------

def _extract_positive_class_scores(proba_raw):
    if proba_raw is None:
        return None
    if isinstance(proba_raw, list):
        processed = []
        for arr in proba_raw:
            arr_np = np.asarray(arr)
            processed.append(arr_np[:, 1] if arr_np.ndim == 2 and arr_np.shape[1] > 1 else arr_np.ravel())
        try:
            return np.column_stack(processed)
        except ValueError:
            return None
    proba_np = np.asarray(proba_raw)
    return proba_np[:, 1] if proba_np.ndim == 2 and proba_np.shape[1] > 1 else proba_np.ravel()


def _predict_with_optional_proba(pipe_inference, X_test):
    try:
        proba_raw = pipe_inference.predict_proba(X_test)
    except AttributeError:
        proba_raw = None
    return _extract_positive_class_scores(proba_raw)


def _compute_binary_outputs(pipe_inference, X_test, threshold):
    y_pred_proba = _predict_with_optional_proba(pipe_inference, X_test)
    y_pred_discrete = (y_pred_proba > threshold).astype(int) if y_pred_proba is not None else pipe_inference.predict(X_test)
    return y_pred_proba, y_pred_discrete


def _compute_metric_scores(metrics, y_true, y_pred_proba, y_pred_discrete):
    metric_scores = {}
    y_true_array = np.asarray(y_true)
    for metric_name, metric_info in metrics.items():
        use_proba = metric_info["needs_proba"]
        if use_proba and y_pred_proba is None:
            print(f"Impossible de calculer {metric_name} (predict_proba indisponible).")
            continue
        preds_input = y_pred_proba if use_proba else y_pred_discrete
        try:
            metric_scores[metric_name] = metric_info["metric_fn"](y_true_array, preds_input)
        except Exception as exc:
            print(f"Impossible de calculer {metric_name} : {exc}")
    return metric_scores


def _compute_youden_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, (fpr, tpr, thresholds)
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx]), (fpr, tpr, thresholds)


# ---------------------------------------------------------------------------
# Dataset filtering helpers
# ---------------------------------------------------------------------------

def _apply_condition_test(X_test, y_test, condition_test):
    if condition_test is None:
        return X_test, y_test
    if callable(condition_test):
        try:
            mask = condition_test(X_test)
        except TypeError:
            mask = condition_test(X_test, y_test)
    else:
        mask = condition_test
    if hasattr(mask, "reindex"):
        mask = mask.reindex(X_test.index)
        mask_bool = np.asarray(mask, dtype=bool)
        X_filtered = X_test.loc[~mask]
        y_filtered = y_test.loc[X_filtered.index] if hasattr(y_test, "loc") else np.asarray(y_test)[~mask_bool]
        return X_filtered, y_filtered
    mask_bool = np.asarray(mask, dtype=bool)
    X_filtered = X_test.loc[~mask_bool] if hasattr(X_test, "loc") else X_test[~mask_bool]
    y_filtered = y_test.loc[X_filtered.index] if hasattr(y_test, "loc") else np.asarray(y_test)[~mask_bool]
    return X_filtered, y_filtered


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _export_model_bars(scores_dict, title, output_path):
    model_names = list(scores_dict.keys())
    means, stds = [], []
    for value in scores_dict.values():
        mean, std = value, 0.0
        if isinstance(value, dict):
            mean = value.get("mean", value.get("avg_score", 0.0))
            std = value.get("std", value.get("std_score", 0.0))
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            mean, std = value[0], value[1]
        means.append(float(mean))
        stds.append(abs(float(std)))
    has_errors = any(std > 0 for std in stds)
    plt.figure(figsize=(8, 6))
    bars = plt.bar(model_names, means, yerr=stds if has_errors else None, capsize=5 if has_errors else None)
    plt.title(title)
    plt.xlabel("Modeles")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    for idx, bar in enumerate(bars):
        label = f"{bar.get_height():.3f}" if stds[idx] == 0 else f"{bar.get_height():.3f} +/- {stds[idx]:.3f}"
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


# ---------------------------------------------------------------------------
# Single-run evaluation
# ---------------------------------------------------------------------------

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
                  show_roc=False, 
                  show_importance=False, 
                  show_shap=False, 
                  method_importance='native_importance',
                  sensibilite=False, 
                  features_sensibilite=None, 
                  type_sensi='all', 
                  verbose=False,
                  text_save=None,
                  to_save=False, 
                  save_dir=None,
                  calibration = False,
                  calibration_method = "isotonic", # or "sigmoid" / "platt"
                  show_calibration_fitness=False,
                  ):

    if features_sensibilite is None:
        features_sensibilite = ["Neutropenie", "Prophylaxis_antifungal"]
    if verbose:
        print("----- run_model_aug -----")
        print(f"target_col={target_col}")
        print(f"model={model_name}")
        print(f"augmentation={augmentation_name}")
        print(f"metric={MAIN_METRIC_NAME} | threshold={THRESHOLD}")
    steps = [("scaler", AutoStandardScaler())]
    if augmentation is not None:
        steps.append(("augmentation", clone(augmentation)))
    steps.append(("model", clone(base_model)))
    pipe_train = Pipeline(steps)
    pipe_train.fit(X_train, y_train)
    pipe_inference = Pipeline([("scaler", pipe_train.named_steps["scaler"]), ("model", pipe_train.named_steps["model"])])
    if calibration:
        pipe_inference = CalibratedModel(
            predictor=pipe_inference,
            X_calibration=X_test,
            y_calibration=y_test,
            method=calibration_method,   # or "sigmoid" / "platt"
            show_calibration_fitness=show_calibration_fitness,
        )
    
    y_pred_proba = _predict_with_optional_proba(pipe_inference, X_test)
    if needs_proba or show_roc:
        if y_pred_proba is None:
            raise AttributeError("Le modele ne fournit pas predict_proba alors qu'il est requis.")
        y_pred = y_pred_proba
    else:
        y_pred = pipe_inference.predict(X_test)
    score = metric_fn(y_test, y_pred)
    y_pred_bin = (y_pred > THRESHOLD).astype(int)
    if verbose:
        print(f"{model_name} | {augmentation_name} -> {MAIN_METRIC_NAME}: {score:.4f}")
        print("Negative Predictive Value:", negative_predictive_value(y_test, y_pred_bin))
    target_save_dir = _resolve_target_save_dir(save_dir, target_col, text_save=text_save) if to_save else None
    youden_threshold, roc_points = _compute_youden_threshold(y_test, y_pred)
    if show_roc:
        show_roc_curve(
            y_test, y_pred, roc_points=roc_points, highlight_threshold=youden_threshold,
            highlight_label=f"Youden = {youden_threshold:.2f}", highlight_color="crimson",
            save_path=str(target_save_dir / "roc_curve.png") if target_save_dir is not None else None
        )
        print("Negative Predictive Value youden:", negative_predictive_value(y_test, (y_pred > youden_threshold).astype(int)), "threshold", youden_threshold)
    if show_importance:
        odds_dir = str(target_save_dir / "odds_ratios") if target_save_dir is not None else ""
        plot_top_odds_ratios(X_test, y_test, feature_names=feature_names, top_n=10, ridge_alpha=1.0,
                             n_bootstrap=500, random_state=None, to_save=to_save, dir_save=odds_dir,
                             title=f"Top 10 odds ratios for {target_col}")
    if show_shap:
        shap_top10(model=pipe_train.named_steps["model"], X_test=X_test, col_names=target_col, to_save=False, dir_save='D:/graphs_bdd/shap')
    if sensibilite:
        if type_sensi == 'drop':
            drop_scores = {"baseline": score}
            for feature in features_sensibilite:
                if feature not in X_train.columns:
                    print(f"Feature {feature} absente du jeu de donnees, ignoree pour le drop.")
                    continue
                drop_steps = [("scaler", AutoStandardScaler())]
                if augmentation is not None:
                    drop_steps.append(("augmentation", clone(augmentation)))
                drop_steps.append(("model", clone(base_model)))
                pipe_train_drop = Pipeline(drop_steps)
                X_train_drop = X_train.drop(columns=feature)
                X_test_drop = X_test.drop(columns=feature)
                pipe_train_drop.fit(X_train_drop, y_train)
                pipe_inference_drop = Pipeline([("scaler", pipe_train_drop.named_steps["scaler"]), ("model", pipe_train_drop.named_steps["model"])])
                y_pred_drop = pipe_inference_drop.predict_proba(X_test_drop)[:, 1] if needs_proba or show_roc else pipe_inference_drop.predict(X_test_drop)
                drop_scores[feature] = metric_fn(y_test, y_pred_drop)
            sensi_path = str(target_save_dir / "sensibilite_drop.png") if target_save_dir is not None else None
            plot_model_bars(drop_scores, title=f"Chute {MAIN_METRIC_NAME.upper()} lorsqu'on drop certaines colonnes ", save_path=sensi_path)
        else:
            sensi_path = str(target_save_dir / f"sensibilite_{type_sensi}.png") if target_save_dir is not None else None
            analyse_sensibilite(pipe_inference, X_test, features_sensibilite, type_sensi=type_sensi, save_path=sensi_path)
    return {"score": score, "pipe_train": pipe_train, "pipe_inference": pipe_inference, "y_pred": y_pred, "Youden_threshold": youden_threshold}


# ---------------------------------------------------------------------------
# Monte Carlo search internals and public search API
# ---------------------------------------------------------------------------

def _find_best_combo_from_splits(split_iterable, total_runs, MODEL_NAMES, MAIN_METRIC_NAME, metric_fn, needs_proba,
                                 base_threshold, target_col, target_save_dir, to_save, write_config, random_seed):
    actual_runs = 0
    combo_tracker = defaultdict(lambda: {"scores": [], "best_entry": None})
    best_result = {"score": float("-inf"), "score_mean": float("-inf"), "score_std": 0.0, "model_name": None,
                   "augmentation_name": None, "pipe_train": None, "pipe_inference": None, "y_pred": None,
                   "y_test": None, "youden_threshold": None, "montecarlo_runs": 0}
    model_best_scores, model_best_aug, model_best_entries, combined_roc_data = {}, {}, {}, []
    for split_idx, split in enumerate(split_iterable, start=1):
        X_train, X_test, y_train, y_test = split["X_train"], split["X_test"], split["y_train"], split["y_test"]
        current_seed = split.get("seed", random_seed)
        actual_runs = split_idx
        y_test_array = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.asarray(y_test)
        print(f"Monte Carlo {split_idx}/{total_runs} (seed={current_seed})")
        all_models = get_models(y_train, use_catboost=True, imbalance_threshold=0.1, random_state=current_seed)
        augmentations = get_augmentation_methods(random_state=current_seed)
        for model_name in MODEL_NAMES:
            if model_name not in all_models:
                print(f"{model_name} indisponible dans la librairie de modeles, combo ignore.")
                continue
            for augmentation_name, augmentation in augmentations.items():
                result = run_model_aug(model_name, all_models[model_name], augmentation_name, augmentation, X_train, X_test, y_train, y_test,
                                       MAIN_METRIC_NAME, metric_fn, needs_proba, base_threshold, target_col, X_train.columns, verbose=False)
                entry = combo_tracker[(model_name, augmentation_name)]
                entry["scores"].append(result["score"])
                if entry["best_entry"] is None or result["score"] > entry["best_entry"]["score"]:
                    entry["best_entry"] = {"score": result["score"], "result": result, "y_test": y_test_array}
    if not combo_tracker:
        print("Aucun combo valide.")
        best_result["montecarlo_runs"] = actual_runs
        return best_result
    for (model_name, augmentation_name), entry in combo_tracker.items():
        avg_score = float(np.mean(entry["scores"]))
        std_score = float(np.std(entry["scores"])) if len(entry["scores"]) > 1 else 0.0
        if avg_score > best_result["score_mean"]:
            best_entry = entry["best_entry"]
            best_result.update({"score": best_entry["score"], "score_mean": avg_score, "score_std": std_score,
                                "model_name": model_name, "augmentation_name": augmentation_name,
                                "pipe_train": best_entry["result"]["pipe_train"], "pipe_inference": best_entry["result"]["pipe_inference"],
                                "y_pred": best_entry["result"]["y_pred"], "y_test": best_entry["y_test"]})
        if avg_score > model_best_scores.get(model_name, float("-inf")):
            model_best_scores[model_name] = avg_score
            model_best_aug[model_name] = augmentation_name
            model_best_entries[model_name] = {"avg_score": avg_score, "std_score": std_score, "entry": entry["best_entry"]}
    if best_result["model_name"] is not None:
        best_avg = best_result["score_mean"] if np.isfinite(best_result["score_mean"]) else best_result["score"]
        print(f"Meilleur combo -> Modele: {best_result['model_name']} | Augmentation: {best_result['augmentation_name']} | {MAIN_METRIC_NAME} (moyenne MC): {best_avg:.4f} | std: {best_result['score_std']:.4f} | meilleur run: {best_result['score']:.4f}")
    if best_result["y_pred"] is not None and best_result["y_test"] is not None:
        best_threshold, _ = _compute_youden_threshold(best_result["y_test"], best_result["y_pred"])
        best_result["youden_threshold"] = best_threshold
        print("Negative Predictive Value (best combo, seuil Youden {:.3f}): {}".format(best_threshold, negative_predictive_value(best_result["y_test"], (best_result["y_pred"] > best_threshold).astype(int))))
    if model_best_entries:
        labelled_scores = {f"{model} ({model_best_aug[model]})": (data["avg_score"], data["std_score"]) for model, data in model_best_entries.items()}
        plot_model_bars(labelled_scores, title=f"Meilleure augmentation par modele - {target_col}")
        if to_save and target_save_dir is not None:
            score_records = []
            for model_name, data in model_best_entries.items():
                youden_threshold, _ = _compute_youden_threshold(data["entry"]["y_test"], data["entry"]["result"]["y_pred"])
                score_records.append({"target": target_col, "model_name": model_name, "augmentation_name": model_best_aug[model_name],
                                      "avg_score": float(data["avg_score"]), "std_score": float(data["std_score"]),
                                      "best_run_score": float(data["entry"]["score"]), "youden_threshold": float(youden_threshold),
                                      "main_metric": MAIN_METRIC_NAME, "montecarlo_runs": actual_runs, "random_seed": random_seed})
            if score_records:
                with (target_save_dir / "scores_summary.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(score_records[0].keys()))
                    writer.writeheader()
                    writer.writerows(score_records)
                with (target_save_dir / "scores_summary.json").open("w", encoding="utf-8") as handle:
                    json.dump(score_records, handle, indent=2)
                _export_model_bars(labelled_scores, title=f"Meilleure augmentation par modele - {target_col}", output_path=target_save_dir / "scores_bar.png")
        print("\nCourbes ROC des meilleures augmentations par modele :")
        for model_name, data in model_best_entries.items():
            youden_threshold, roc_points = _compute_youden_threshold(data["entry"]["y_test"], data["entry"]["result"]["y_pred"])
            print(f"- {model_name} ({model_best_aug[model_name]}) - {MAIN_METRIC_NAME} moyen = {data['avg_score']:.4f} +/- {data['std_score']:.4f} | seuil Youden = {youden_threshold:.4f}")
            combined_roc_data.append({"model_name": model_name, "augmentation_name": model_best_aug[model_name], "roc_points": roc_points,
                                      "youden_threshold": youden_threshold, "avg_score": data["avg_score"], "std_score": data["std_score"]})
        if combined_roc_data:
            fig, ax = plt.subplots(figsize=(7, 7))
            for item in combined_roc_data:
                fpr, tpr, thresholds = item["roc_points"]
                ax.plot(fpr, tpr, lw=2, label=f"{item['model_name']} ({item['augmentation_name']}) - {item['avg_score']:.3f}+/-{item['std_score']:.3f}")
                if len(thresholds) > 0:
                    idx = int(np.argmin(np.abs(thresholds - item["youden_threshold"])))
                    ax.scatter(fpr[idx], tpr[idx], s=70, edgecolor="black", linewidth=0.8)
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
            ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05]); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC - meilleures augmentations ({target_col})"); ax.legend(loc="lower right"); ax.grid(False); fig.tight_layout()
            if to_save and target_save_dir is not None:
                fig.savefig(target_save_dir / "roc_curves.png", dpi=180)
                roc_payload = [{"model_name": item["model_name"], "augmentation_name": item["augmentation_name"], "avg_score": float(item["avg_score"]),
                                "std_score": float(item["std_score"]), "youden_threshold": float(item["youden_threshold"]),
                                "fpr": np.asarray(item["roc_points"][0]).tolist(), "tpr": np.asarray(item["roc_points"][1]).tolist(),
                                "thresholds": np.asarray(item["roc_points"][2]).tolist()} for item in combined_roc_data]
                with (target_save_dir / "roc_curves.json").open("w", encoding="utf-8") as handle:
                    json.dump(roc_payload, handle, indent=2)
            plt.show(); plt.close(fig)
    if to_save and target_save_dir is not None and best_result["model_name"] is not None:
        youden_value = float(best_result["youden_threshold"]) if best_result.get("youden_threshold") is not None and np.isfinite(best_result["youden_threshold"]) else None
        best_export = {"target": target_col, "model_name": best_result["model_name"], "augmentation_name": best_result["augmentation_name"],
                       "main_metric": MAIN_METRIC_NAME, "score_mean": float(best_result["score_mean"]), "score_std": float(best_result["score_std"]),
                       "best_run_score": float(best_result["score"]), "youden_threshold": youden_value,
                       "montecarlo_runs": actual_runs, "random_seed": random_seed}
        with (target_save_dir / "best_result.json").open("w", encoding="utf-8") as handle:
            json.dump(best_export, handle, indent=2)
    if write_config and best_result["model_name"] is not None:
        score_to_save = best_result["score_mean"] if np.isfinite(best_result["score_mean"]) else best_result["score"]
        threshold_to_save = best_result.get("youden_threshold")
        if threshold_to_save is None or not np.isfinite(threshold_to_save):
            threshold_to_save = base_threshold
        save_best_combo_config(target_col, best_result["model_name"], best_result["augmentation_name"], MAIN_METRIC_NAME,
                               score_to_save, threshold_to_save, random_seed=random_seed)
    elif write_config:
        print("Impossible de sauvegarder la configuration : aucun combo valide.")
    best_result["montecarlo_runs"] = actual_runs
    return best_result


def find_best_model_and_aug(df_features_clean, df_labels_fusion, target_col, MODEL_NAMES, MAIN_METRIC_NAME,
                            write_config=False, to_save=False, save_dir="artifacts/find_best",
                            random_seed=42, montecarlo=5, condition_test=None):
    target_save_dir = _resolve_target_save_dir(save_dir, target_col) if to_save else None
    random_seed = 42 if random_seed is None else random_seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    metric_entry = get_metric()[MAIN_METRIC_NAME]
    print("====", target_col, "====")

    def _split_iterator():
        df_labels_1 = df_labels_fusion[target_col].to_frame()
        for mc_idx in range(montecarlo):
            current_seed = random_seed + mc_idx
            X_train, X_test, y_train, y_test, _ = preparer_jeu_xy(df_features_clean, df_labels_1, random_state=current_seed)
            X_test_filtered, y_test_filtered = _apply_condition_test(X_test, y_test, condition_test)
            yield {"X_train": X_train, "X_test": X_test_filtered, "y_train": y_train, "y_test": y_test_filtered, "seed": current_seed}

    return _find_best_combo_from_splits(_split_iterator(), montecarlo, MODEL_NAMES, MAIN_METRIC_NAME,
                                        metric_entry["metric_fn"], metric_entry["needs_proba"], 0.11,
                                        target_col, target_save_dir, to_save, write_config, random_seed)


def find_best_model_and_aug_from_split(X_train, X_test, y_train, y_test, target_col, MODEL_NAMES, MAIN_METRIC_NAME,
                                       write_config=False, to_save=False, save_dir="artifacts/find_best",
                                       random_seed=42, montecarlo=1, condition_test=None):
    target_save_dir = _resolve_target_save_dir(save_dir, target_col) if to_save else None
    random_seed = 42 if random_seed is None else random_seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    metrics = get_metric()
    if MAIN_METRIC_NAME not in metrics:
        raise KeyError(f"Metrique inconnue: {MAIN_METRIC_NAME}")
    metric_entry = metrics[MAIN_METRIC_NAME]
    print("====", target_col, "(split fourni) ====")

    def _split_iterator():
        for mc_idx in range(montecarlo):
            current_seed = random_seed + mc_idx
            X_test_filtered, y_test_filtered = _apply_condition_test(X_test, y_test, condition_test)
            yield {"X_train": X_train, "X_test": X_test_filtered, "y_train": y_train, "y_test": y_test_filtered, "seed": current_seed}

    return _find_best_combo_from_splits(_split_iterator(), montecarlo, MODEL_NAMES, MAIN_METRIC_NAME,
                                        metric_entry["metric_fn"], metric_entry["needs_proba"], 0.11,
                                        target_col, target_save_dir, to_save, write_config, random_seed)


# ---------------------------------------------------------------------------
# Config loading and config-driven evaluation
# ---------------------------------------------------------------------------

def load_config_for_target(target_col, config_dir=None):
    base_dir = Path(config_dir) if config_dir is not None else Path.cwd() / "configs"
    config_path = base_dir / f"config_{target_col}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Fichier de configuration introuvable pour {target_col} ({config_path}).")
    config = {}
    with config_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
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

def run_config_for_target(target_col, df_features_clean, df_labels_fusion, sensibilite=False, show_importance=False,
                          show_roc=False, show_shap=False, features_sensibilite=None, type_sensi='all',
                          method_importance='native_importance', config_dir=os.getcwd() + '\\configs\\',
                          condition_test=None, text_save=None, to_save=False, save_dir=None):
    config, config_path = load_config_for_target(target_col, config_dir=config_dir)
    model_name = config.get("model")
    augmentation_name = config.get("augmentation", "No Augmentation")
    MAIN_METRIC_NAME = config.get("main_metric", "roc_auc")
    THRESHOLD = config.get("threshold", 0.5)
    random_seed = config.get("random_seed", 42)
    try:
        random_seed = 42 if random_seed is None else int(random_seed)
    except (TypeError, ValueError):
        random_seed = 42
    if features_sensibilite is None:
        features_sensibilite = ["Neutropenie", "Prophylaxis_antifungal"]
    if not model_name:
        raise ValueError(f"Le fichier {config_path} ne contient pas de modele.")
    
    df_labels_1 = df_labels_fusion[target_col].to_frame()
    X_train, X_test, y_train, y_test, _ = preparer_jeu_xy(df_features_clean, df_labels_1, random_state=random_seed)
    X_test, y_test = _apply_condition_test(X_test, y_test, condition_test)

    if condition_test is not None:
        print(f"Filtrage du jeu de test via condition_test -> {len(X_test)} echantillons.")

    all_models = get_models(y_train, use_catboost=True, imbalance_threshold=0.2, random_state=random_seed)
    augmentations = get_augmentation_methods(random_state=random_seed)
    metrics = get_metric()
    metric_entry = metrics[MAIN_METRIC_NAME]
    print(f"Chargement configuration : {config_path}")
    print(f"Modele = {model_name} | Augmentation = {augmentation_name} | Metrique = {MAIN_METRIC_NAME}")
    run_output = run_model_aug(model_name, all_models[model_name], augmentation_name, augmentations[augmentation_name],
                               X_train, X_test, y_train, y_test, MAIN_METRIC_NAME, metric_entry["metric_fn"],
                               metric_entry["needs_proba"], THRESHOLD, target_col, X_train.columns, show_roc,
                               show_importance, show_shap, method_importance, sensibilite, features_sensibilite,
                               type_sensi, True, text_save, to_save, save_dir,
                               calibration=True,
                               calibration_method = "isotonic", # isotonic or "sigmoid" / "platt",
                               show_calibration_fitness= True,
                               )
    y_pred_proba, y_pred_discrete = _compute_binary_outputs(run_output["pipe_inference"], X_test, THRESHOLD)
    run_output["metric_scores"] = _compute_metric_scores(metrics, y_test, y_pred_proba, y_pred_discrete)
    return run_output
