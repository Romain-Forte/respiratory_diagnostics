import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from utils.models_and_metrics import get_metric

def feature_influence_sign(model, X, feature_names, epsilon=1e-4):
    signs = {}

    baseline_pred = model.predict(X).mean()

    for f in feature_names:
        X_mod = X.copy()
        X_mod[f] = X_mod[f] + epsilon

        new_pred = model.predict(X_mod).mean()

        if new_pred > baseline_pred:
            signs[f] = 1
        else:
            signs[f] = -1

    return signs

def _get_importances(estimator, feature_names):
    """
    Récupère un vecteur d'importance de features depuis différents types de modèles.
    - coef_ (modèles linéaires)
    - feature_importances_ (arbres / xgboost)
    """
    if hasattr(estimator, "coef_"):
        coefs = np.ravel(estimator.coef_)
    elif hasattr(estimator, "feature_importances_"):
        coefs = np.ravel(estimator.feature_importances_)
    else:
        raise AttributeError("L'estimateur ne possède ni coef_ ni feature_importances_.")
    feature_names = np.asarray(feature_names)
    return feature_names, coefs


def plot_top_odds_ratios(
    X,
    y,
    feature_names=None,
    top_n=10,
    ridge_alpha=1.0,
    n_bootstrap=200,
    random_state=None,
    to_save=False,
    dir_save="",
    title=None,
):
    """
    Ajuste une rÃ©gression logistique ridge (L2) sur X, y puis affiche les top features par odds ratio.
    Fournit un intervalle de confiance bootstrap Ã  95% sur les odds ratios.

    ParamÃ¨tres
    ----------
    X : DataFrame ou array-like (n_samples, n_features).
    y : array-like (n_samples,), labels binaires {0,1}.
    feature_names : list-like ou None, noms des colonnes si X n'est pas un DataFrame.
    top_n : nombre de features Ã  afficher.
    ridge_alpha : coefficient de ridge (alpha = 1/C). Doit Ãªtre >= 0.
    n_bootstrap : nombre d'Ã©chantillons bootstrap pour les intervalles.
    random_state : int ou None pour contrÃ´ler le bootstrap.
    to_save : bool, sauvegarde la figure si True.
    dir_save : dossier de sauvegarde.
    title : titre personnalisÃ©.

    Retour
    ------
    pandas.DataFrame avec feature, coef, odds_ratio, ci_lower, ci_upper.
    """
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha doit Ãªtre positif ou nul.")

    if hasattr(X, "values"):
        X_matrix = np.asarray(X.values, dtype=float)
        inferred_names = np.asarray(X.columns)
    else:
        X_matrix = np.asarray(X, dtype=float)
        inferred_names = np.array([f"Feature {i}" for i in range(X_matrix.shape[1])])

    if feature_names is None:
        feature_names = inferred_names
    else:
        feature_names = np.asarray(feature_names)

    if feature_names.shape[0] != X_matrix.shape[1]:
        raise ValueError("feature_names doit correspondre au nombre de colonnes de X.")

    y_array = np.asarray(y).ravel()
    unique_labels = np.unique(y_array)
    if unique_labels.size != 2:
        raise ValueError("plot_top_odds_ratios requiert un problÃ¨me binaire (2 classes).")

    C_value = 1e6 if ridge_alpha == 0 else 1.0 / ridge_alpha
    base_model = LogisticRegression(
        penalty="l2",
        C=C_value,
        solver="lbfgs",
        max_iter=2000,
    )
    base_model.fit(X_matrix, y_array)
    coefs = base_model.coef_.ravel()
    odds_ratios = np.exp(coefs)

    metrics = get_metric()
    roc_auc_info = metrics.get("roc_auc")
    roc_auc_value = None
    if roc_auc_info is not None:
        if roc_auc_info["needs_proba"]:
            y_pred_scores = base_model.predict_proba(X_matrix)[:, 1]
        else:
            y_pred_scores = base_model.predict(X_matrix)
        roc_auc_value = roc_auc_info["metric_fn"](y_array, y_pred_scores)

    # Bootstrap pour intervalles de confiance
    rng = np.random.default_rng(random_state)
    boot_or = []
    n_samples = X_matrix.shape[0]
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_samples, size=n_samples)
        X_boot = X_matrix[indices]
        y_boot = y_array[indices]

        if np.unique(y_boot).size < 2:
            continue

        boot_model = LogisticRegression(
            penalty="l2",
            C=C_value,
            solver="lbfgs",
            max_iter=1000,
        )
        boot_model.fit(X_boot, y_boot)
        boot_or.append(np.exp(boot_model.coef_.ravel()))

    if boot_or:
        boot_or = np.vstack(boot_or)
        ci_lower = np.percentile(boot_or, 2.5, axis=0)
        ci_upper = np.percentile(boot_or, 97.5, axis=0)
    else:
        ci_lower = np.full_like(odds_ratios, np.nan, dtype=float)
        ci_upper = np.full_like(odds_ratios, np.nan, dtype=float)

    top_n = min(top_n, X_matrix.shape[1])
    order = np.argsort(np.abs(coefs))[::-1][:top_n]
    top_features = feature_names[order]
    top_coefs = coefs[order]
    top_odds = odds_ratios[order]
    top_ci_lower = ci_lower[order]
    top_ci_upper = ci_upper[order]

    colors = np.where(top_odds >= 1.0, "tab:blue", "tab:red")

    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    y_pos = np.arange(top_n)
    ax.barh(y_pos, top_odds[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features[::-1])
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    lower_err = np.clip(top_odds - top_ci_lower, a_min=0, a_max=None)
    upper_err = np.clip(top_ci_upper - top_odds, a_min=0, a_max=None)
    xerr = np.vstack((lower_err[::-1], upper_err[::-1]))
    ax.errorbar(
        top_odds[::-1],
        y_pos,
        xerr=xerr,
        fmt="none",
        ecolor="black",
        capsize=4,
    )

    ax.set_xscale("log")
    positive_values = np.concatenate([
        top_odds[np.isfinite(top_odds) & (top_odds > 0)],
        top_ci_lower[np.isfinite(top_ci_lower) & (top_ci_lower > 0)],
        top_ci_upper[np.isfinite(top_ci_upper) & (top_ci_upper > 0)],
    ])
    if positive_values.size:
        lower_bound = positive_values.min()
        upper_bound = positive_values.max()
        margin = 1.15
        lower_bound /= margin
        upper_bound *= margin
        symmetric_factor = max(upper_bound, 1.0 / max(lower_bound, 1e-12))
        if symmetric_factor <= 1.0:
            symmetric_factor = 1.5
        ax.set_xlim(1.0 / symmetric_factor, symmetric_factor)
    else:
        ax.set_xlim(0.25, 4.0)

    ticks_candidates = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0])
    x_min, x_max = ax.get_xlim()
    log_min, log_max = np.log10(x_min), np.log10(x_max)
    ticks = [tick for tick in ticks_candidates if x_min <= tick <= x_max]
    if 1.0 not in ticks and x_min < 1.0 < x_max:
        ticks.append(1.0)
    ticks = sorted(set(ticks))
    if len(ticks) < 5:
        extra_ticks = np.logspace(log_min, log_max, num=5)
        ticks = sorted(set(ticks + [tick for tick in extra_ticks if x_min <= tick <= x_max]))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:.2f}" for tick in ticks])


    plt.xlabel("Odds ratio")
    if title is None:
        title = f"Top {top_n} odds ratios (LogReg ridge)"
    plt.title(title)
    plt.tight_layout()

    if to_save:
        os.makedirs(dir_save, exist_ok=True)
        plt.savefig(os.path.join(dir_save, "odds_ratios_ridge.png"))
        plt.close()
    else:
        plt.show()

    return pd.DataFrame({
        "feature": top_features,
        "coef": top_coefs,
        "odds_ratio": top_odds,
        "ci_lower": top_ci_lower,
        "ci_upper": top_ci_upper
    })


def plot_top10_features_per_estimator(model, 
                                      feature_names, 
                                      col_names,
                                      method='native_importance', 
                                      top_n=10, 
                                      to_save=False, 
                                      dir_save="",
                                      X_test = None,
                                      y_test = None,
                                      signed_importance = True):
    """
    Affiche (ou sauvegarde) les top features pour chaque estimateur.
    Compatible modèles OneVsRest (.estimators_) et modèles simples (ex: XGBClassifier).
    """

    estimators = [model]
    for idx, clf in enumerate(estimators):
        if method == 'native_importance':
            feat_names, coefs = _get_importances(clf, feature_names)
        elif method == 'permutation':
            if X_test is None or y_test is None:
                raise ValueError("X_test et y_test doivent être fournis pour l'importance par permutation.")
            result = permutation_importance(
                    clf, X_test, y_test, n_repeats=10, random_state=42
                )
            coefs = result.importances_mean
            feat_names = np.asarray(X_test.columns)

        else:
            raise KeyError('Method not supported, only permutation and native_importance are supported currently')
        
        
        abs_coefs = np.abs(coefs)
        top_idx = np.argsort(abs_coefs)[::-1][:top_n]
        top_features = feat_names[top_idx]
        top_values = coefs[top_idx]
        if signed_importance:
            is_any_neg = any(x < 0 for x in top_values)
            if  not is_any_neg:
                signs = feature_influence_sign(model, X_test, feature_names, epsilon=1e-4)
                

        
        colors = np.where(top_values >= 0, "tab:blue", "tab:red")

        plt.figure(figsize=(8, 4))
        plt.barh(top_features[::-1], top_values[::-1], color=colors[::-1])
        plt.title(f"Top {top_n} features - Estimateur {col_names[idx]}")
        plt.xlabel("Importance (mean decrease of impurity)")
        plt.tight_layout()

        if to_save:
            os.makedirs(dir_save, exist_ok=True)
            plt.gcf().canvas.draw()
            plt.savefig(os.path.join(dir_save, f"{col_names[idx]}_summary_bar.png"))
            plt.close()
        else:
            plt.show()


def shap_top10(model,
               X_test,
               col_names,
               top_n=10,
               to_save=False,
               dir_save="",
               max_display=15):
    """
    Affiche les top features selon SHAP pour chaque sortie du modèle.
    Flux simplifié : shap.Explainer -> bar chart + summary plot par classe.
    """
    feature_names = np.asarray(X_test.columns)
    features_matrix = np.asarray(X_test)

    try:
        explainer = shap.Explainer(model, X_test)
    except Exception:
        explainer = shap.TreeExplainer(model)

    try:
        shap_values = explainer(X_test)
        values = np.asarray(shap_values.values)
        data = np.asarray(shap_values.data)
    except Exception:
        values = np.asarray(explainer.shap_values(X_test))
        data = features_matrix

    if values.ndim == 2:
        value_mats = [values]
    elif values.ndim == 3:
        value_mats = [values[:, i, :] for i in range(values.shape[1])]
    else:
        value_mats = [values.reshape(values.shape[0], -1)]

    if isinstance(col_names, (str, int)):
        col_names = [col_names]
    col_names = list(col_names)
    if len(col_names) < len(value_mats):
        col_names += [f"Classe {i}" for i in range(len(col_names), len(value_mats))]

    if to_save:
        os.makedirs(dir_save, exist_ok=True)

    for idx, shap_matrix in enumerate(value_mats):
        mean_signed = shap_matrix.mean(axis=0)
        order = np.argsort(np.abs(mean_signed))[::-1][:top_n]
        top_features = feature_names[order]
        top_values = mean_signed[order]
        colors = ["tab:blue" if val >= 0 else "tab:red" for val in top_values]

        plt.figure(figsize=(8, 4))
        plt.barh(top_features[::-1], top_values[::-1], color=colors[::-1])
        plt.title(f"Top {top_n} SHAP - {col_names[idx]}")
        plt.xlabel("SHAP moyen")
        plt.tight_layout()
        if to_save:
            plt.savefig(os.path.join(dir_save, f"{col_names[idx]}_shap_top{top_n}.png"))
            plt.close()
        else:
            plt.show()

        summary_features = (
            data[:, idx, :]
            if data.ndim == 3 and data.shape[1] == len(value_mats)
            else features_matrix
        )
        plt.figure()
        shap.summary_plot(
            shap_matrix,
            summary_features,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        if to_save:
            plt.savefig(os.path.join(dir_save, f"{col_names[idx]}_shap_summary.png"))
            plt.close()
        else:
            plt.show()
