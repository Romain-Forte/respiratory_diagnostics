import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

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
