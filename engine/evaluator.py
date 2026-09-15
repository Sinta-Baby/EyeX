"""
==========================================================
RetinaSense

Evaluation

Description:
Evaluates the trained RetinaSense model.

==========================================================
"""

import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class Evaluator:

    def __init__(
        self,
        model,
        test_loader,
        device
    ):

        self.model = model

        self.test_loader = test_loader

        self.device = device

        self.model.to(device)

    def evaluate(self):

        self.model.eval()

        predictions = []

        labels = []

        with torch.no_grad():

            for images, targets in self.test_loader:

                images = images.to(self.device)

                outputs = self.model(images)

                predicted = outputs.argmax(dim=1)

                predictions.extend(
                    predicted.cpu().numpy()
                )

                labels.extend(
                    targets.numpy()
                )

        print("=" * 70)
        print("Evaluation Results")
        print("=" * 70)

        print(
            f"Accuracy : {accuracy_score(labels, predictions):.4f}"
        )

        print(
            f"Precision : {precision_score(labels, predictions, average='weighted'):.4f}"
        )

        print(
            f"Recall : {recall_score(labels, predictions, average='weighted'):.4f}"
        )

        print(
            f"F1 Score : {f1_score(labels, predictions, average='weighted'):.4f}"
        )

        print("\nConfusion Matrix\n")

        print(
            confusion_matrix(
                labels,
                predictions
            )
        )

        print("\nClassification Report\n")

        print(
            classification_report(
                labels,
                predictions
            )
        )