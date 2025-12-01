import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score
def show_metrics_binary(y_true, y_pred, threshold=0.5):
    """
    Affiche F1-score, Accuracy et la matrice de confusion pour un problème binaire.
    y_true et y_pred sont forcés en numpy arrays ; si y_pred contient des
    probabilités continues, il est binarisé avec `threshold`.
    """
    # Forcer en numpy float
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Si y_pred est 2D (ex: predict_proba), prendre la colonne de la classe 1
    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        y_pred = y_pred[:, 1]

    # Détecter si y_true n'est pas binaire (ex: floats continus) -> binariser
    unique_true = np.unique(y_true)
    if not np.all(np.isin(unique_true, [0.0, 1.0])):
        y_true = (y_true >= threshold).astype(int)
    else:
        y_true = y_true.astype(int)

    # Détecter si y_pred contient valeurs non binaires -> binariser
    unique_pred = np.unique(y_pred)
    if not np.all(np.isin(unique_pred, [0.0, 1.0])):
        y_pred = (y_pred >= threshold).astype(int)
    else:
        y_pred = y_pred.astype(int)

    # --- Calcul des métriques ---

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # --- Affichage ---
    print(f"Utilisation du seuil = {threshold}")
    print(f"🎯 F1-score : {f1:.4f}")
    print(f"🔎 Accuracy : {acc:.4f}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Prédit 0", "Prédit 1"],
        yticklabels=["Réel 0", "Réel 1"]
    )
    plt.title("Matrice de confusion")
    plt.ylabel("Classe réelle")
    plt.xlabel("Classe prédite")
    plt.show()

def show_roc_curve(y_true, y_score, threshold=0.5, pos_label=1, to_print=True):
    """
    Trace la courbe ROC et retourne la ROC AUC.
    - y_score : probabilités/scores (si 2D, prend la colonne de la classe 1)
    - y_true  : si non binaire, est binarisé avec `threshold`
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    # Forcer en numpy float
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    # Si y_score est 2D (ex: predict_proba), prendre la colonne de la classe 1
    if y_score.ndim == 2 and y_score.shape[1] > 1:
        y_score = y_score[:, 1]

    # Binariser y_true si nécessaire
    unique_true = np.unique(y_true)
    if not np.all(np.isin(unique_true, [0.0, 1.0])):
        y_true = (y_true >= threshold).astype(int)
    else:
        y_true = y_true.astype(int)

    # Calcul ROC
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = roc_auc_score(y_true, y_score)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    if to_print:
        print(f"ROC AUC = {roc_auc:.4f}")
    return roc_auc


def multilabel_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    average: str = "macro",
    plot: bool = True,
    label_names=None,
    per_label_max: int = 10,  # nombre max de courbes par label à afficher
    show_micro: bool = True,
    show_macro: bool = True,
    return_fig_ax: bool = False,
):
    """
    Calcule AUC globaux (micro/macro/weighted), AUC par label et courbes ROC.
    Optionnellement, trace les courbes ROC (par label + micro + macro).

    Paramètres
    ----------
    y_true : array (n_samples, n_labels), binaire
    y_score: array (n_samples, n_labels), scores/probas
    average: str, ignoré ici pour les retours; présent pour compat API
    plot : bool, si True génère une figure matplotlib
    label_names : list[str] ou None, noms de colonnes; sinon indices 0..L-1
    per_label_max : int, nombre max de courbes de labels à dessiner
    show_micro : bool, trace la courbe micro-moyennée si disponible
    show_macro : bool, trace la courbe macro (moyenne des interp. par label valides)
    return_fig_ax : bool, si True retourne aussi (fig, ax)

    Retour
    ------
    dict avec:
      - auc_micro, auc_macro, auc_weighted (floats ou NaN)
      - auc_per_label : array (n_labels) avec NaN pour labels invalides
      - roc_per_label : dict[label] -> (fpr, tpr) ou None
      - roc_micro : (fpr, tpr, auc) ou None
      - roc_macro : (fpr, tpr, auc) ou None
      - plotted_labels : liste des labels effectivement dessinés
    (+ éventuellement fig, ax si return_fig_ax=True)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    assert y_true.shape == y_score.shape and y_true.ndim == 2, "Shapes attendues: (n_samples, n_labels)"

    n_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f"Label {i}" for i in range(n_labels)]

    out = {}

    # AUC globaux
    for avg in ["micro", "macro", "weighted"]:
        try:
            out[f"auc_{avg}"] = roc_auc_score(y_true, y_score, average=avg)
        except ValueError:
            out[f"auc_{avg}"] = np.nan

    # Par label
    per_label_auc = np.full(n_labels, np.nan, dtype=float)
    roc_per_label = {}

    valid_mask = np.zeros(n_labels, dtype=bool)
    for k in range(n_labels):
        yk = y_true[:, k]
        sk = y_score[:, k]
        # besoin d'au moins un positif et un négatif
        if yk.max() == 0 or yk.min() == 1:
            roc_per_label[k] = None
            continue
        fpr_k, tpr_k, _ = roc_curve(yk, sk)
        roc_per_label[k] = (fpr_k, tpr_k)
        per_label_auc[k] = auc(fpr_k, tpr_k)
        valid_mask[k] = True




    out["auc_per_label"] = per_label_auc
    out["roc_per_label"] = roc_per_label

    # Micro
    if valid_mask.any():
        yt = y_true[:, valid_mask].ravel()
        ys = y_score[:, valid_mask].ravel()
        fpr_m, tpr_m, _ = roc_curve(yt, ys)
        out["roc_micro"] = (fpr_m, tpr_m, auc(fpr_m, tpr_m))
    else:
        out["roc_micro"] = None

    # Macro: interpolation sur une grille commune, puis moyenne
    if valid_mask.any():
        # grille FPR commune
        all_fpr = np.unique(np.concatenate([roc_per_label[k][0] for k in np.where(valid_mask)[0]]))
        mean_tpr = np.zeros_like(all_fpr)
        for k in np.where(valid_mask)[0]:
            fpr_k, tpr_k = roc_per_label[k]
            mean_tpr += np.interp(all_fpr, fpr_k, tpr_k)
        mean_tpr /= valid_mask.sum()
        auc_macro_curve = auc(all_fpr, mean_tpr)
        out["roc_macro"] = (all_fpr, mean_tpr, auc_macro_curve)
    else:
        out["roc_macro"] = None

    plotted_labels = []
    fig = ax = None

    if plot:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        # micro
        if show_micro and out["roc_micro"] is not None:
            fpr_m, tpr_m, auc_m = out["roc_micro"]
            ax.plot(fpr_m, tpr_m, label=f"micro-average (AUC={auc_m:.3f})", linewidth=2)

        # macro
        if show_macro and out["roc_macro"] is not None:
            fpr_g, tpr_g, auc_g = out["roc_macro"]
            ax.plot(fpr_g, tpr_g, label=f"macro-average (AUC={auc_g:.3f})", linewidth=2, linestyle="--")

        # choisir jusqu'à per_label_max meilleurs AUC pour lisibilité
        valid_indices = np.where(valid_mask)[0]
        if valid_indices.size:
            # trier par AUC décroissante
            order = valid_indices[np.argsort(per_label_auc[valid_indices])[::-1]]
            to_plot = order[:per_label_max]
            for k in to_plot:
                fpr_k, tpr_k = roc_per_label[k]
                ax.plot(fpr_k, tpr_k, label=f"{label_names[k]} (AUC={per_label_auc[k]:.3f})", alpha=0.85)
            plotted_labels = [label_names[k] for k in to_plot]

        # diagonale aléatoire
        ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1)

        ax.set_xlabel("FPR (False Positive Rate)")
        ax.set_ylabel("TPR (True Positive Rate)")
        ax.set_title("ROC multi-label")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=":", linewidth=0.7)

        plt.tight_layout()

    out["plotted_labels"] = plotted_labels

    if return_fig_ax:
        return out, (fig, ax)
    return out