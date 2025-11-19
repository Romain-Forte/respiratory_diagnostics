import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def show_metrics_binary(y_true, y_pred):
    """
    Affiche F1-score, Accuracy et la matrice de confusion pour un problème binaire.
    """

    # --- Calcul des métriques ---
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # --- Affichage ---
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
