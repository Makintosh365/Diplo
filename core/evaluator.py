import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from typing import List


def evaluate_predictions(true_labels: List[str], predicted_labels: List[str]) -> None:
    """
    Оценивает точность классификации и выводит отчёт.

    :param true_labels: Список правильных меток.
    :param predicted_labels: Список предсказанных меток.
    """
    acc = accuracy_score(true_labels, predicted_labels)
    print(f"🔎 Accuracy: {acc:.4f}")
    print("\n🧾 Classification Report:")
    print(classification_report(true_labels, predicted_labels))

    cm = confusion_matrix(true_labels, predicted_labels, labels=sorted(set(true_labels)))
    plot_confusion_matrix(cm, labels=sorted(set(true_labels)))


def plot_confusion_matrix(cm, labels):
    """
    Строит визуализацию матрицы ошибок.

    :param cm: Матрица ошибок.
    :param labels: Список классов.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Предсказанные метки")
    plt.ylabel("Истинные метки")
    plt.title("Матрица ошибок")
    plt.tight_layout()
    plt.show()
