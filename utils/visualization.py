"""
==========================================================
RetinaSense

Visualization Utility

Description:
Creates training graphs and confusion matrix.

==========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay

from core.paths import REPORTS


# =====================================================
# Create Report Folder
# =====================================================

REPORT_FOLDER = REPORTS / "training"

REPORT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# Plot Loss Curve
# =====================================================

def plot_loss_curve(history_csv):

    history = pd.read_csv(history_csv)

    plt.figure(figsize=(8,5))

    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Train Loss",
        linewidth=2
    )

    plt.plot(
        history["epoch"],
        history["validation_loss"],
        label="Validation Loss",
        linewidth=2
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        REPORT_FOLDER / "loss_curve.png",
        dpi=300
    )

    plt.close()


# =====================================================
# Plot Accuracy Curve
# =====================================================

def plot_accuracy_curve(history_csv):

    history = pd.read_csv(history_csv)

    plt.figure(figsize=(8,5))

    plt.plot(
        history["epoch"],
        history["train_accuracy"],
        label="Train Accuracy",
        linewidth=2
    )

    plt.plot(
        history["epoch"],
        history["validation_accuracy"],
        label="Validation Accuracy",
        linewidth=2
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.title("Training Accuracy")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        REPORT_FOLDER / "accuracy_curve.png",
        dpi=300
    )

    plt.close()


# =====================================================
# Plot Confusion Matrix
# =====================================================

def plot_confusion_matrix(
    confusion_matrix,
    class_names
):

    fig, ax = plt.subplots(
        figsize=(8,8)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=class_names
    )

    display.plot(
        cmap="Blues",
        ax=ax,
        colorbar=False
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_FOLDER /
        "confusion_matrix.png",
        dpi=300
    )

    plt.close()


from pathlib import Path

# =====================================================
# Plot Everything
# =====================================================

def plot_training_history(history_csv):

    history_csv = Path(history_csv)

    if not history_csv.exists():

        print("=" * 60)
        print("History file not found.")
        print("Please train the model first.")
        print("=" * 60)

        return

    plot_loss_curve(history_csv)

    plot_accuracy_curve(history_csv)

    print("=" * 60)
    print("Training graphs saved successfully.")
    print("=" * 60)