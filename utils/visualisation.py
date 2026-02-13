import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score
import pandas as pd
from matplotlib import cm

def plot_multilabel_cooccurrence(
    df_labels: pd.DataFrame,
    min_support: float = 0.0,
    top_k: int | None = None,
    normalize: str = "percent",
    annot: bool = False,
    figsize=(10, 8),
    cmap: str = "magma",
    mask_upper: bool = True,
):
    """
    Visualise la co-occurrence des labels (df_cat_clean, one-hot) sous forme de heatmap.

    Paramètres
    ----------
    df_labels : DataFrame binaire (shape = [n_samples, n_labels])
    min_support : float, proportion minimale (0-1) pour garder un label
    top_k : int ou None, conserve les top_k labels les plus fréquents (après min_support)
    normalize : "percent" (par défaut) ou "count"
        - percent : affiche le pourcentage de co-occurrence sur le total des lignes
        - count   : affiche le nombre de co-occurrences
    annot : bool, affiche les valeurs dans chaque case
    figsize : tuple, taille de la figure
    cmap : palette pour la heatmap
    mask_upper : si True, masque la partie supérieure de la matrice pour lisibilité

    Retour
    ------
    fig, ax : objets matplotlib
    """
    if not isinstance(df_labels, pd.DataFrame):
        raise TypeError("df_labels doit être un DataFrame pandas avec des colonnes binaires.")

    n_rows = len(df_labels)
    if n_rows == 0:
        raise ValueError("df_labels est vide.")

    # Binariser par sécurité
    df_bin = (df_labels > 0).astype(int)

    # Fréquence de chaque label
    support = df_bin.mean().sort_values(ascending=False)
    if min_support > 0:
        support = support[support >= min_support]
    if top_k is not None:
        support = support.head(top_k)

    if support.empty:
        raise ValueError("Aucun label après filtrage min_support/top_k.")

    df_bin = df_bin[support.index]

    # Matrice de co-occurrence
    cooc_counts = df_bin.T.dot(df_bin)
    if normalize == "percent":
        cooc = cooc_counts / n_rows * 100
        fmt = ".1f"
        cbar_label = "% des patients"
    elif normalize == "count":
        cooc = cooc_counts
        fmt = ".0f"
        cbar_label = "Nombre de co-occurrences"
    else:
        raise ValueError("normalize doit être 'percent' ou 'count'.")

    # Option pour ne garder que le triangle inférieur (visuellement plus lisible)
    if mask_upper:
        mask = np.triu(np.ones_like(cooc, dtype=bool))
    else:
        mask = None

    # Ne pas saturer la diagonale
    np.fill_diagonal(cooc.values, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cooc,
        mask=mask,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        square=True,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_title("Co-occurrence des labels (df_cat_clean)")
    plt.tight_layout()
    return fig, ax

def plot_multilabel_network_matplotlib(
    df_labels: pd.DataFrame,
    min_support: float = 0.0,
    top_k: int | None = None,
    normalize: str = "count",  # "count" ou "percent"
    edge_threshold: float | None = None,  # seuil pour dessiner une arête
    edge_scale: tuple[float, float] = (0.5, 6.0),  # épaisseur min/max
    node_scale: tuple[float, float] = (300.0, 1800.0),  # taille min/max des nœuds
    figsize=(10, 8),
    annot_edges: bool = False,
    annot_nodes: bool = True,
    cmap: str = "magma",
):
    """
    Graphe de co-occurrence sans networkx : nodes sur un cercle, arêtes pondérées par co-occurrence.

    Paramètres
    ----------
    df_labels : DataFrame binaire (n_samples, n_labels) — ex: df_cat_clean
    min_support : proportion minimale (0-1) pour conserver un label
    top_k : conserve les top_k labels restants
    normalize : "count" ou "percent" pour le poids des arêtes
    edge_threshold : si défini, seules les arêtes dont le poids >= seuil sont tracées
    edge_scale : (min, max) largeur des arêtes
    node_scale : (min, max) taille des nœuds (fonction du support)
    annot_edges : affiche le poids sur les arêtes
    annot_nodes : affiche le nom des labels
    """
    if not isinstance(df_labels, pd.DataFrame):
        raise TypeError("df_labels doit être un DataFrame pandas.")
    if len(df_labels) == 0:
        raise ValueError("df_labels est vide.")

    df_bin = (df_labels > 0).astype(int)
    support = df_bin.mean().sort_values(ascending=False)
    if min_support > 0:
        support = support[support >= min_support]
    if top_k is not None:
        support = support.head(top_k)
    if support.empty:
        raise ValueError("Aucun label après filtrage min_support/top_k.")

    df_bin = df_bin[support.index]
    n_rows = len(df_bin)

    cooc_counts = df_bin.T.dot(df_bin)
    np.fill_diagonal(cooc_counts.values, 0)

    if normalize == "percent":
        cooc = cooc_counts / n_rows * 100.0
        label_fmt = lambda w: f"{w:.1f}%"
    elif normalize == "count":
        cooc = cooc_counts
        label_fmt = lambda w: f"{w:.0f}"
    else:
        raise ValueError("normalize doit être 'count' ou 'percent'.")

    # Construire la liste d'arêtes
    edges = []
    for i, src in enumerate(cooc.index):
        for j, dst in enumerate(cooc.columns):
            if j <= i:
                continue
            w = cooc.iloc[i, j]
            if w <= 0:
                continue
            if edge_threshold is not None and w < edge_threshold:
                continue
            edges.append((src, dst, float(w)))

    if not edges:
        raise ValueError("Aucune arête à dessiner après filtrage/seuil.")

    weights = np.array([w for _, _, w in edges], dtype=float)
    w_min, w_max = weights.min(), weights.max()
    # Échelle des arêtes
    if w_max == w_min:
        widths = np.full_like(weights, edge_scale[1])
    else:
        widths = edge_scale[0] + (weights - w_min) / (w_max - w_min) * (edge_scale[1] - edge_scale[0])

    # Échelle des nœuds (support)
    s_min, s_max = support.min(), support.max()
    if s_max == s_min:
        node_sizes = pd.Series(node_scale[1], index=support.index)
    else:
        node_sizes = node_scale[0] + (support - s_min) / (s_max - s_min) * (node_scale[1] - node_scale[0])

    # Placement des nœuds sur un cercle
    n_nodes = len(support)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    coords = {name: (np.cos(a), np.sin(a)) for name, a in zip(support.index, angles)}

    fig, ax = plt.subplots(figsize=figsize)

    # Arêtes
    norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
    cmap_obj = cm.get_cmap(cmap)
    for (u, v, w), width in zip(edges, widths):
        x1, y1 = coords[u]
        x2, y2 = coords[v]
        ax.plot([x1, x2], [y1, y2], linewidth=width, color=cmap_obj(norm(w)), alpha=0.8)
        if annot_edges:
            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(xm, ym, label_fmt(w), fontsize=8, ha="center", va="center")

    # Nœuds
    for name, (x, y) in coords.items():
        ax.scatter(x, y, s=node_sizes[name], color="skyblue", edgecolors="k", zorder=3)
        if annot_nodes:
            ax.text(x, y, name, fontsize=9, ha="center", va="center", zorder=4)

    sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Co-occurrence" + (" (%)" if normalize == "percent" else " (count)"))

    ax.set_title("Graphe de co-occurrence des labels")
    ax.set_axis_off()
    plt.tight_layout()
    return fig, ax
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
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
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

    # Annoter quelques points avec leur threshold associé
    if len(fpr) > 0:
        nb_labels = min(10, len(fpr))
        idxs = np.linspace(0, len(fpr) - 1, nb_labels, dtype=int)
        for idx in idxs:
            thr = thresholds[idx]
            thr_label = "inf" if np.isinf(thr) else f"{thr:.2f}"
            plt.scatter(fpr[idx], tpr[idx], color="darkorange", s=20)
            plt.text(
                fpr[idx],
                tpr[idx],
                f"t={thr_label}",
                fontsize=8,
                ha="left",
                va="bottom",
                color="black",
            )

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
