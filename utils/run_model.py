import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import roc_curve
from pathlib import Path
import numpy as np
import os 
import random
from collections import defaultdict
from utils.algo_prediction import preparer_jeu_xy, AutoStandardScaler
from utils.models_and_metrics import get_models, get_metric, negative_predictive_value,save_best_combo_config
from utils.visualisation import show_metrics_binary, show_roc_curve,plot_model_bars
from utils.analyse_sensibilite import analyse_sensibilite
from utils.feature_importance import  plot_top10_features_per_estimator,shap_top10,plot_top_odds_ratios
from utils.data_aug import get_augmentation_methods
import pandas as pd
import json
import csv
def _slugify(value):
        clean = "".join(char if char.isalnum() else "_" for char in (value or ""))
        clean = clean.strip("_")
        return clean or "target"


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
        if hasattr(y_test, "loc"):
            y_filtered = y_test.loc[X_filtered.index]
        else:
            y_filtered = np.asarray(y_test)[~mask_bool]
        return X_filtered, y_filtered

    mask_bool = np.asarray(mask, dtype=bool)
    if hasattr(X_test, "loc"):
        X_filtered = X_test.loc[~mask_bool]
    else:
        X_filtered = X_test[~mask_bool]

    if hasattr(y_test, "loc"):
        y_filtered = y_test.loc[X_filtered.index]
    else:
        y_filtered = np.asarray(y_test)[~mask_bool]

    return X_filtered, y_filtered


def _find_best_combo_from_splits(
    split_iterable,
    total_runs,
    MODEL_NAMES,
    MAIN_METRIC_NAME,
    metric_fn,
    needs_proba,
    base_threshold,
    target_col,
    target_save_dir,
    to_save,
    write_config,
    random_seed,
):
    actual_runs = 0
    combo_tracker = defaultdict(lambda: {"scores": [], "best_entry": None})
    best_result = {
        "score": float("-inf"),
        "score_mean": float("-inf"),
        "score_std": 0.0,
        "model_name": None,
        "augmentation_name": None,
        "pipe_train": None,
        "pipe_inference": None,
        "y_pred": None,
        "y_test": None,
        "youden_threshold": None,
        "montecarlo_runs": 0,
    }
    model_best_scores = {}
    model_best_aug = {}
    model_best_entries = {}
    combined_roc_data = []

    for split_idx, split in enumerate(split_iterable, start=1):
        X_train = split["X_train"]
        X_test = split["X_test"]
        y_train = split["y_train"]
        y_test = split["y_test"]
        current_seed = split.get("seed", random_seed)
        actual_runs = split_idx

        if hasattr(y_test, "to_numpy"):
            y_test_array = y_test.to_numpy()
        else:
            y_test_array = np.asarray(y_test)

        print(f"Monte Carlo {split_idx}/{total_runs} (seed={current_seed})")

        all_models = get_models(y_train, use_catboost=True, imbalance_threshold=0.1, random_state=current_seed)
        augmentations = get_augmentation_methods(random_state=current_seed)

        for model_name in MODEL_NAMES:
            if model_name not in all_models:
                print(f"{model_name} indisponible dans la librairie de modeles, combo ignore.")
                continue
            base_model = all_models[model_name]

            for augmentation_name, augmentation in augmentations.items():
                result = run_model_aug(
                    model_name=model_name,
                    base_model=base_model,
                    augmentation_name=augmentation_name,
                    augmentation=augmentation,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    MAIN_METRIC_NAME=MAIN_METRIC_NAME,
                    metric_fn=metric_fn,
                    needs_proba=needs_proba,
                    THRESHOLD=base_threshold,
                    target_col=target_col,
                    feature_names=X_train.columns,
                    verbose=False
                )

                score = result["score"]
                combo_key = (model_name, augmentation_name)
                entry = combo_tracker[combo_key]
                entry["scores"].append(score)

                if entry["best_entry"] is None or score > entry["best_entry"]["score"]:
                    entry["best_entry"] = {
                        "score": score,
                        "result": result,
                        "y_test": y_test_array
                    }

    if not combo_tracker:
        print("Aucun combo valide.")
        best_result["montecarlo_runs"] = actual_runs
        return best_result

    for (model_name, augmentation_name), entry in combo_tracker.items():
        scores = entry["scores"]
        avg_score = float(np.mean(scores))
        std_score = float(np.std(scores)) if len(scores) > 1 else 0.0

        if avg_score > best_result["score_mean"]:
            best_entry = entry["best_entry"]
            best_result.update({
                "score": best_entry["score"],
                "score_mean": avg_score,
                "score_std": std_score,
                "model_name": model_name,
                "augmentation_name": augmentation_name,
                "pipe_train": best_entry["result"]["pipe_train"],
                "pipe_inference": best_entry["result"]["pipe_inference"],
                "y_pred": best_entry["result"]["y_pred"],
                "y_test": best_entry["y_test"]
            })

        current_best = model_best_scores.get(model_name, float('-inf'))
        if avg_score > current_best:
            model_best_scores[model_name] = avg_score
            model_best_aug[model_name] = augmentation_name
            model_best_entries[model_name] = {
                "avg_score": avg_score,
                "std_score": std_score,
                "entry": entry["best_entry"]
            }

    best_model_name = best_result["model_name"]
    best_aug_name = best_result["augmentation_name"]

    if best_model_name is not None:
        best_avg = best_result["score_mean"]
        if not np.isfinite(best_avg):
            best_avg = best_result["score"]
        print(
            f"Meilleur combo -> Modele: {best_model_name} | Augmentation: {best_aug_name} | "
            f"{MAIN_METRIC_NAME} (moyenne MC): {best_avg:.4f} | std: {best_result['score_std']:.4f} | "
            f"meilleur run: {best_result['score']:.4f}"
        )

    best_y_pred = best_result["y_pred"]
    best_y_test = best_result["y_test"]

    if best_y_pred is not None and best_y_test is not None:
        best_threshold, best_roc_points = _compute_youden_threshold(best_y_test, best_y_pred)
        best_result["youden_threshold"] = best_threshold
        y_pred_bin = (best_y_pred > best_threshold).astype(int)
        print(
            "Negative Predictive Value (best combo, seuil Youden {:.3f}): {}".format(
                best_threshold, negative_predictive_value(best_y_test, y_pred_bin)
            )
        )

    if model_best_entries:
        labelled_scores = {
            f"{model} ({model_best_aug[model]})": (data["avg_score"], data["std_score"])
            for model, data in model_best_entries.items()
        }
        plot_model_bars(labelled_scores, title=f"Meilleure augmentation par modele - {target_col}")

        if to_save and target_save_dir is not None:
            score_records = []
            for model_name, data in model_best_entries.items():
                entry = data["entry"]
                youden_threshold, _ = _compute_youden_threshold(entry["y_test"], entry["result"]["y_pred"])
                score_records.append({
                    "target": target_col,
                    "model_name": model_name,
                    "augmentation_name": model_best_aug[model_name],
                    "avg_score": float(data["avg_score"]),
                    "std_score": float(data["std_score"]),
                    "best_run_score": float(entry["score"]),
                    "youden_threshold": float(youden_threshold),
                    "main_metric": MAIN_METRIC_NAME,
                    "montecarlo_runs": actual_runs,
                    "random_seed": random_seed
                })

            if score_records:
                csv_path = target_save_dir / "scores_summary.csv"
                fieldnames = [
                    "target",
                    "model_name",
                    "augmentation_name",
                    "avg_score",
                    "std_score",
                    "best_run_score",
                    "youden_threshold",
                    "main_metric",
                    "montecarlo_runs",
                    "random_seed"
                ]
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(score_records)

                json_path = target_save_dir / "scores_summary.json"
                with json_path.open("w", encoding="utf-8") as handle:
                    json.dump(score_records, handle, indent=2)

                bars_path = target_save_dir / "scores_bar.png"
                _export_model_bars(
                    labelled_scores,
                    title=f"Meilleure augmentation par modele - {target_col}",
                    output_path=bars_path
                )

        print("\nCourbes ROC des meilleures augmentations par modele :")
        for model_name, data in model_best_entries.items():
            augmentation_name = model_best_aug[model_name]
            avg_score = data["avg_score"]
            std_score = data["std_score"]
            entry = data["entry"]
            youden_threshold, roc_points = _compute_youden_threshold(entry["y_test"], entry["result"]["y_pred"])
            print(
                f"- {model_name} ({augmentation_name}) â€” {MAIN_METRIC_NAME} moyen = {avg_score:.4f} Â± {std_score:.4f} "
                f"| seuil Youden = {youden_threshold:.4f}"
            )
            combined_roc_data.append({
                "model_name": model_name,
                "augmentation_name": augmentation_name,
                "roc_points": roc_points,
                "youden_threshold": youden_threshold,
                "avg_score": avg_score,
                "std_score": std_score
            })

        if combined_roc_data:
            fig, ax = plt.subplots(figsize=(7, 7))
            for item in combined_roc_data:
                fpr, tpr, thresholds = item["roc_points"]
                label = (
                    f"{item['model_name']} ({item['augmentation_name']}) - "
                    f"{item['avg_score']:.3f}Â±{item['std_score']:.3f}"
                )
                ax.plot(fpr, tpr, lw=2, label=label)

                threshold = item["youden_threshold"]
                if len(thresholds) > 0:
                    highlight_idx = int(np.argmin(np.abs(thresholds - threshold)))
                    ax.scatter(
                        fpr[highlight_idx],
                        tpr[highlight_idx],
                        s=70,
                        edgecolor="black",
                        linewidth=0.8
                    )

            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC - meilleures augmentations ({target_col})")
            ax.legend(loc="lower right")
            ax.grid(False)
            fig.tight_layout()

            if to_save and target_save_dir is not None:
                roc_fig_path = target_save_dir / "roc_curves.png"
                fig.savefig(roc_fig_path, dpi=180)

                roc_payload = []
                for roc_item in combined_roc_data:
                    fpr, tpr, thresholds = roc_item["roc_points"]
                    roc_payload.append({
                        "model_name": roc_item["model_name"],
                        "augmentation_name": roc_item["augmentation_name"],
                        "avg_score": float(roc_item["avg_score"]),
                        "std_score": float(roc_item["std_score"]),
                        "youden_threshold": float(roc_item["youden_threshold"]),
                        "fpr": np.asarray(fpr).tolist(),
                        "tpr": np.asarray(tpr).tolist(),
                        "thresholds": np.asarray(thresholds).tolist()
                    })
                roc_json_path = target_save_dir / "roc_curves.json"
                with roc_json_path.open("w", encoding="utf-8") as handle:
                    json.dump(roc_payload, handle, indent=2)

            plt.show()
            plt.close(fig)

    if to_save and target_save_dir is not None and best_result["model_name"] is not None:
        youden_value = best_result.get("youden_threshold")
        if youden_value is not None and np.isfinite(youden_value):
            youden_value = float(youden_value)
        else:
            youden_value = None
        best_export = {
            "target": target_col,
            "model_name": best_result["model_name"],
            "augmentation_name": best_result["augmentation_name"],
            "main_metric": MAIN_METRIC_NAME,
            "score_mean": float(best_result["score_mean"]),
            "score_std": float(best_result["score_std"]),
            "best_run_score": float(best_result["score"]),
            "youden_threshold": youden_value,
            "montecarlo_runs": actual_runs,
            "random_seed": random_seed
        }
        best_summary_path = target_save_dir / "best_result.json"
        with best_summary_path.open("w", encoding="utf-8") as handle:
            json.dump(best_export, handle, indent=2)

    if write_config and best_result["model_name"] is not None:
        score_to_save = best_result["score_mean"]
        if not np.isfinite(score_to_save):
            score_to_save = best_result["score"]
        threshold_to_save = best_result.get("youden_threshold")
        if threshold_to_save is None or not np.isfinite(threshold_to_save):
            threshold_to_save = base_threshold
        save_best_combo_config(
            target_col,
            best_result["model_name"],
            best_result["augmentation_name"],
            MAIN_METRIC_NAME,
            score_to_save,
            threshold_to_save,
            random_seed=random_seed
        )
    elif write_config:
        print('Impossible de sauvegarder la configuration : aucun combo valide.')

    best_result["montecarlo_runs"] = actual_runs
    return best_result


def _export_model_bars(scores_dict, title, output_path):
    model_names = list(scores_dict.keys())
    raw_values = list(scores_dict.values())

    means = []
    stds = []
    for value in raw_values:
        mean = value
        std = 0.0
        if isinstance(value, dict):
            mean = value.get("mean", value.get("avg_score", 0.0))
            std = value.get("std", value.get("std_score", 0.0))
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            mean, std = value[0], value[1]
        means.append(float(mean))
        stds.append(abs(float(std)))

    has_errors = any(std > 0 for std in stds)

    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        model_names,
        means,
        yerr=stds if has_errors else None,
        capsize=5 if has_errors else None
    )
    plt.title(title)
    plt.xlabel("ModÃ¨les")
    plt.ylabel("Score")
    plt.xticks(rotation=45)

    for idx, bar in enumerate(bars):
        height = bar.get_height()
        if stds[idx] > 0:
            label = f"{height:.3f} Â± {stds[idx]:.3f}"
        else:
            label = f"{height:.3f}"
        plt.text(bar.get_x() + bar.get_width()/2, height, label, ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _compute_youden_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, (fpr, tpr, thresholds)
    j_scores = tpr - fpr
    idx = int(np.argmax(j_scores))
    return float(thresholds[idx]), (fpr, tpr, thresholds)


def find_best_model_and_aug(df_features_clean, 
                   df_labels_fusion, 
                   target_col,
                   MODEL_NAMES,
                   MAIN_METRIC_NAME,
                   write_config = False,
                   to_save = False,
                   save_dir = "artifacts/find_best",
                   random_seed = 42,
                   montecarlo = 5,
                   condition_test = None):
    """
    Lance l'experience pour une cible (diagnostic) en moyenne sur plusieurs tirages Monte Carlo.

    Parameters
    ----------
    df_features_clean : pd.DataFrame
        Features nettoyees.
    df_labels_fusion : pd.DataFrame
        Labels.
    target_col : str
        Colonne cible.
    MODEL_NAMES : list[str]
        Modeles a evaluer.
    MAIN_METRIC_NAME : str
        Nom de la metrique a optimiser.
    write_config : bool
        Sauvegarder la configuration gagnante.
    to_save : bool, default=False
        Sauvegarder les scores (CSV/JSON) et les graphes de performance dans un dossier.
    save_dir : str or Path
        Dossier parent pour stocker les exports (un sous-dossier par cible).
    random_seed : int
        Graine initiale.
    montecarlo : int, default=5
        Nombre d'iterations Monte Carlo (split/train/test) utilisees pour moyenner les scores.
    condition_test : callable or None
        Filtre booleen pour exclure certaines lignes de X_test/y_test avant evaluation.


    Returns
    -------
    dict contenant les informations du meilleur combo modele/augmentation.
    Si write_config=True, sauvegarde config_{diagnosis}.yaml.
    """

    target_save_dir = None
    if to_save:
        base_dir = Path(save_dir).expanduser()
        target_save_dir = base_dir / _slugify(target_col)
        target_save_dir.mkdir(parents=True, exist_ok=True)

    if random_seed is None:
        random_seed = 42

    np.random.seed(random_seed)
    random.seed(random_seed)
    base_threshold = 0.11

    metrics = get_metric()
    metric_entry = metrics[MAIN_METRIC_NAME]
    metric_fn = metric_entry["metric_fn"]
    needs_proba = metric_entry["needs_proba"]

    df_features_to_use = df_features_clean
    df_labels_1 = df_labels_fusion[target_col].to_frame()
    print("====", target_col, "====")

    def _split_iterator():
        for mc_idx in range(montecarlo):
            current_seed = random_seed + mc_idx
            X_train, X_test, y_train, y_test, _ = preparer_jeu_xy(
                df_features_to_use,
                df_labels_1,
                random_state=current_seed
            )
            X_test_filtered, y_test_filtered = _apply_condition_test(X_test, y_test, condition_test)
            yield {
                "X_train": X_train,
                "X_test": X_test_filtered,
                "y_train": y_train,
                "y_test": y_test_filtered,
                "seed": current_seed
            }

    return _find_best_combo_from_splits(
        split_iterable=_split_iterator(),
        total_runs=montecarlo,
        MODEL_NAMES=MODEL_NAMES,
        MAIN_METRIC_NAME=MAIN_METRIC_NAME,
        metric_fn=metric_fn,
        needs_proba=needs_proba,
        base_threshold=base_threshold,
        target_col=target_col,
        target_save_dir=target_save_dir,
        to_save=to_save,
        write_config=write_config,
        random_seed=random_seed
    )


def find_best_model_and_aug_from_split(
    X_train,
    X_test,
    y_train,
    y_test,
    target_col,
    MODEL_NAMES,
    MAIN_METRIC_NAME,
    write_config=False,
    to_save=False,
    save_dir="artifacts/find_best",
    random_seed=42,
    montecarlo=1,
    condition_test=None,
):
    """
    Variante de find_best_model_and_aug utilisant des jeux train/test deja separes.
    """
    target_save_dir = None
    if to_save:
        base_dir = Path(save_dir).expanduser()
        target_save_dir = base_dir / _slugify(target_col)
        target_save_dir.mkdir(parents=True, exist_ok=True)

    if random_seed is None:
        random_seed = 42

    np.random.seed(random_seed)
    random.seed(random_seed)
    base_threshold = 0.11

    metrics = get_metric()
    if MAIN_METRIC_NAME not in metrics:
        raise KeyError(f"Metrique inconnue: {MAIN_METRIC_NAME}")
    metric_entry = metrics[MAIN_METRIC_NAME]
    metric_fn = metric_entry["metric_fn"]
    needs_proba = metric_entry["needs_proba"]

    print("====", target_col, "(split fourni) ====")

    def _split_iterator():
        for mc_idx in range(montecarlo):
            current_seed = random_seed + mc_idx
            X_test_filtered, y_test_filtered = _apply_condition_test(X_test, y_test, condition_test)
            yield {
                "X_train": X_train,
                "X_test": X_test_filtered,
                "y_train": y_train,
                "y_test": y_test_filtered,
                "seed": current_seed
            }

    return _find_best_combo_from_splits(
        split_iterable=_split_iterator(),
        total_runs=montecarlo,
        MODEL_NAMES=MODEL_NAMES,
        MAIN_METRIC_NAME=MAIN_METRIC_NAME,
        metric_fn=metric_fn,
        needs_proba=needs_proba,
        base_threshold=base_threshold,
        target_col=target_col,
        target_save_dir=target_save_dir,
        to_save=to_save,
        write_config=write_config,
        random_seed=random_seed
    )


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
                  verbose = False,
                  text_save = None,
                  to_save = False,
                  save_dir = None):
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

    target_save_dir = None
    if to_save:

        if not save_dir:
            raise ValueError("save_dir doit être fourni lorsque to_save=True.")
        if text_save is not None:
            target_save_dir = save_dir + _slugify(target_col) + text_save
        else:
            target_save_dir = save_dir + _slugify(target_col)
        target_save_dir = Path(target_save_dir)
        target_save_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    j_scores = tpr - fpr
    idx = np.argmax(j_scores)
    youden_threshold = thresholds[idx]
    if show_roc:
        
        show_roc_curve(
            y_test,
            y_pred,
            roc_points=(fpr, tpr, thresholds),
            highlight_threshold=youden_threshold,
            highlight_label=f"Youden = {youden_threshold:.2f}",
            highlight_color="crimson",
            save_path=str((target_save_dir / "roc_curve.png")) if target_save_dir is not None else None
        )
        y_pred_bin_roc = (y_pred > youden_threshold).astype(int)
        print("Negative Predictive Value youden:", negative_predictive_value(y_test, y_pred_bin_roc), 'threshold ',youden_threshold)

    if show_importance:
        odds_dir = str(target_save_dir / "odds_ratios") if target_save_dir is not None else ""
        plot_top_odds_ratios(
            X_test,
            y_test,
            feature_names=feature_names,
            top_n=10,
            ridge_alpha=1.0,
            n_bootstrap=500,
            random_state=None,
            to_save=to_save,
            dir_save=odds_dir,
            title=f"Top 10 odds ratios for {target_col}",
        )
        # plot_top10_features_per_estimator(
        #     pipe_train.named_steps["model"],
        #     feature_names=feature_names,
        #     col_names=[target_col],
        #     method=method_importance,
        #     X_test=X_test,
        #     y_test=y_test,
        #     to_save=False,
        #     dir_save='D:/graphs_bdd/importance'
        # )

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
            sensi_path = str(target_save_dir / "sensibilite_drop.png") if target_save_dir is not None else None
            plot_model_bars(
                drop_scores,
                title=f"Chute {MAIN_METRIC_NAME.upper()} lorsqu'on drop certaines colonnes ",
                save_path=sensi_path,
            )
        else:
            sensi_path = str(target_save_dir / f"sensibilite_{type_sensi}.png") if target_save_dir is not None else None
            analyse_sensibilite(
                pipe_inference,
                X_test,
                features_sensibilite,
                type_sensi=type_sensi,
                save_path=sensi_path,
            )

    return {
        "score": score,
        "pipe_train": pipe_train,
        "pipe_inference": pipe_inference,
        "y_pred": y_pred,
        "Youden_threshold" : youden_threshold
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
                          condition_test = None,
                          text_save = None,
                          to_save = False,
                          save_dir = None):
    """
    Recharge la configuration enregistrée pour target_col puis relance run_model_aug.
    condition_test : bool mask / index / callable (X_test, y_test) -> mask pour filtrer le set de test.
    to_save : bool, si True sauvegarde les graphiques ROC et odds ratios.
    save_dir : str ou Path, répertoire de sauvegarde des graphiques.
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


    all_models = get_models(y_train, use_catboost=True, imbalance_threshold=0.2, random_state=random_seed)
    augmentations = get_augmentation_methods(random_state=random_seed)
    metrics = get_metric()



    base_model = all_models[model_name]
    augmentation = augmentations[augmentation_name]
    metric_entry = metrics[MAIN_METRIC_NAME]

    print(f"Chargement configuration : {config_path}")
    print(
        f"Modèle = {model_name} | Augmentation = {augmentation_name} | Métrique = {MAIN_METRIC_NAME}"
    )

    run_output = run_model_aug(
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
        verbose=True,
        text_save = text_save,
        to_save=to_save,
        save_dir=save_dir
    )

    pipe_inference = run_output["pipe_inference"]
    y_test_array = np.asarray(y_test)
    y_pred_proba = None
    try:
        proba_raw = pipe_inference.predict_proba(X_test)
    except AttributeError:
        proba_raw = None

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

    if y_pred_proba is not None:
        y_pred_discrete = (y_pred_proba > THRESHOLD).astype(int)
    else:
        y_pred_discrete = pipe_inference.predict(X_test)

    metric_scores = {}
    for metric_name, metric_info in metrics.items():
        use_proba = metric_info["needs_proba"]
        if use_proba and y_pred_proba is None:
            print(f"Impossible de calculer {metric_name} (predict_proba indisponible).")
            continue
        preds_input = y_pred_proba if use_proba else y_pred_discrete
        try:
            metric_scores[metric_name] = metric_info["metric_fn"](y_test_array, preds_input)
        except Exception as exc:
            print(f"Impossible de calculer {metric_name} : {exc}")

    run_output["metric_scores"] = metric_scores
    return run_output
