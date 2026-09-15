"""
==========================================================
RetinaSense

Model Evaluation

Description:
Evaluates the trained RetinaSense model
on the test dataset.

==========================================================
"""

import torch
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from models.efficientnet_b3 import build_model
from utils.dataloader import get_dataloaders
from core.constants import INDEX_TO_DISEASE


# ======================================================
# Main
# ======================================================

def main():

    print("=" * 70)
    print("RetinaSense Model Evaluation")
    print("=" * 70)

    # ------------------------------------------
    # Device
    # ------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device : {device}")

    # ------------------------------------------
    # Dataset
    # ------------------------------------------

    dataset_root = "datasets/RetinaSense_Dataset"

    train_loader, validation_loader, test_loader = (
        get_dataloaders(dataset_root)
    )

    print("\nTest Dataset Loaded")
    print(f"Test Images : {len(test_loader.dataset)}")

    # ------------------------------------------
    # Build Model
    # ------------------------------------------

    model = build_model(
        pretrained=False
    )

    # ------------------------------------------
    # Load Best Model
    # ------------------------------------------

    model_path = "saved_models/best_model.pth"

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model = model.to(device)

    model.eval()

    print("\n✅ Best model loaded successfully.")

    # ------------------------------------------
    # Prediction
    # ------------------------------------------

    actual_labels = []
    predicted_labels = []

    print("\nRunning Test Evaluation...")

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            actual_labels.extend(
                labels.numpy()
            )

            predicted_labels.extend(
                predictions.cpu().numpy()
            )

    # ------------------------------------------
    # Accuracy
    # ------------------------------------------

    accuracy = accuracy_score(
        actual_labels,
        predicted_labels
    )

    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    print(
        f"Test Accuracy : {accuracy * 100:.2f}%"
    )

    # ------------------------------------------
    # Classification Report
    # ------------------------------------------

    target_names = [
        INDEX_TO_DISEASE[0],
        INDEX_TO_DISEASE[1],
        INDEX_TO_DISEASE[2],
        INDEX_TO_DISEASE[3]
    ]

    report = classification_report(
        actual_labels,
        predicted_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    print("\nClassification Report")
    print("-" * 70)

    print(
        report_df.round(4)
    )

    # ------------------------------------------
    # Save Classification Report
    # ------------------------------------------

    report_path = (
        "results/classification_report.csv"
    )

    report_df.to_csv(
        report_path
    )

    print(
        f"\nClassification report saved to:"
        f" {report_path}"
    )

    # ------------------------------------------
    # Confusion Matrix
    # ------------------------------------------

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=target_names,
        columns=target_names
    )

    print("\nConfusion Matrix")
    print("-" * 70)

    print(matrix_df)

    # ------------------------------------------
    # Save Confusion Matrix
    # ------------------------------------------

    matrix_path = (
        "results/confusion_matrix.csv"
    )

    matrix_df.to_csv(
        matrix_path
    )

    print(
        f"\nConfusion matrix saved to:"
        f" {matrix_path}"
    )

    # ------------------------------------------
    # Finished
    # ------------------------------------------

    print("\n" + "=" * 70)
    print("Evaluation Completed Successfully")
    print("=" * 70)


# ======================================================
# Run
# ======================================================

if __name__ == "__main__":

    main()