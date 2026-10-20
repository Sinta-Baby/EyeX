
"""
EyeX - Evaluation and Comparison of Three Classification Models
Evaluates all models on the same RetinaSense test dataset.
"""

import os
import time
import torch
import pandas as pd
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
)

from utils.dataloader import get_dataloaders
from core.constants import INDEX_TO_DISEASE
from inference.classification_models import (
    load_model,
    MODEL_CONFIGS,
    EFFICIENTNET_CLASS_NAMES,
    NEW_MODEL_CLASS_NAMES,
    DEVICE,
)


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Canonical order used by the RetinaSense test dataset.
CLASS_NAMES = [INDEX_TO_DISEASE[i] for i in range(4)]
NUM_CLASSES = len(CLASS_NAMES)

# Output-index-to-class-name mappings for each model.
MODEL_CLASS_NAMES = {
    "EfficientNet-B3": EFFICIENTNET_CLASS_NAMES,
    "ConvNeXt-Tiny": NEW_MODEL_CLASS_NAMES,
    "Swin-Tiny": NEW_MODEL_CLASS_NAMES,
}


def calculate_specificity(y_true, y_pred):
    """Calculate one-vs-rest specificity for each class."""
    matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(NUM_CLASSES))
    )

    specificities = {}

    for i, class_name in enumerate(CLASS_NAMES):
        true_negative = matrix.sum() - (
            matrix[i, :].sum()
            + matrix[:, i].sum()
            - matrix[i, i]
        )
        false_positive = matrix[:, i].sum() - matrix[i, i]

        denominator = true_negative + false_positive

        specificities[class_name] = (
            true_negative / denominator
            if denominator else 0.0
        )

    return specificities


def evaluate_model(model_name, test_loader):
    print(f"\nEvaluating {model_name}...")

    model = load_model(model_name)
    output_names = MODEL_CLASS_NAMES[model_name]

    # Translate output classes to the test dataset's label indices.
    canonical_index = {
        name: index for index, name in enumerate(CLASS_NAMES)
    }

    try:
        output_to_canonical = [
            canonical_index[name] for name in output_names
        ]
    except KeyError as exc:
        raise ValueError(
            f"Class mapping mismatch for {model_name}: {exc}. "
            f"Dataset classes: {CLASS_NAMES}; "
            f"model classes: {output_names}"
        ) from exc

    actual_labels = []
    predicted_labels = []
    total_inference_seconds = 0.0
    total_images = 0

    model.eval()

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            input_size = MODEL_CONFIGS[model_name]["input_size"]

            # Test loader supplies the existing RetinaSense transforms.
            # Resize normalized tensors for the 224x224 models.
            if images.shape[-2:] != (input_size, input_size):
                images = F.interpolate(
                    images,
                    size=(input_size, input_size),
                    mode="bilinear",
                    align_corners=False,
                )

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            outputs = model(images)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            total_inference_seconds += time.perf_counter() - start

            predicted_output_indices = outputs.argmax(dim=1).cpu().tolist()
            predicted_canonical = [
                output_to_canonical[i]
                for i in predicted_output_indices
            ]

            actual_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predicted_canonical)
            total_images += labels.size(0)

    if total_images == 0:
        raise ValueError("The test dataset is empty.")

    accuracy = accuracy_score(actual_labels, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(
        actual_labels, predicted_labels
    )
    macro_precision = precision_score(
        actual_labels, predicted_labels,
        labels=list(range(NUM_CLASSES)),
        average="macro", zero_division=0
    )
    macro_recall = recall_score(
        actual_labels, predicted_labels,
        labels=list(range(NUM_CLASSES)),
        average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        actual_labels, predicted_labels,
        labels=list(range(NUM_CLASSES)),
        average="macro", zero_division=0
    )
    weighted_precision = precision_score(
        actual_labels, predicted_labels,
        average="weighted", zero_division=0
    )
    weighted_recall = recall_score(
        actual_labels, predicted_labels,
        average="weighted", zero_division=0
    )
    weighted_f1 = f1_score(
        actual_labels, predicted_labels,
        average="weighted", zero_division=0
    )
    mcc = matthews_corrcoef(actual_labels, predicted_labels)

    specificities = calculate_specificity(
        actual_labels, predicted_labels
    )
    macro_specificity = sum(specificities.values()) / NUM_CLASSES

    average_inference_ms = (
        total_inference_seconds / total_images
    ) * 1000

    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=list(range(NUM_CLASSES)),
    )

    # Save detailed report and confusion matrix separately per model.
    safe_name = model_name.lower().replace("-", "_")

    pd.DataFrame(report).transpose().to_csv(
        os.path.join(RESULTS_DIR, f"{safe_name}_report.csv")
    )

    pd.DataFrame(
        matrix, index=CLASS_NAMES, columns=CLASS_NAMES
    ).to_csv(
        os.path.join(RESULTS_DIR, f"{safe_name}_confusion_matrix.csv")
    )

    pd.DataFrame(
        [{"class": name, "specificity": value}
         for name, value in specificities.items()]
    ).to_csv(
        os.path.join(RESULTS_DIR, f"{safe_name}_specificity.csv"),
        index=False,
    )

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "Model": model_name,
        "Test Images": total_images,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted Precision": weighted_precision,
        "Weighted Recall": weighted_recall,
        "Weighted F1": weighted_f1,
        "Macro Specificity": macro_specificity,
        "MCC": mcc,
        "Avg Inference (ms/image)": average_inference_ms,
    }


def main():
    print("=" * 75)
    print("EyeX - Three-Model Evaluation")
    print("=" * 75)
    print(f"Device: {DEVICE}")
    print(f"Class order: {CLASS_NAMES}")

    _, _, test_loader = get_dataloaders(
        "datasets/RetinaSense_Dataset"
    )

    print(f"Test images: {len(test_loader.dataset)}")

    results = []

    for model_name in MODEL_CONFIGS:
        try:
            results.append(evaluate_model(model_name, test_loader))
        except Exception as exc:
            print(f"\nERROR evaluating {model_name}: {exc}")
            raise

    comparison = pd.DataFrame(results)
    comparison.to_csv(
        os.path.join(RESULTS_DIR, "model_comparison.csv"),
        index=False,
    )

    print("\n" + "=" * 75)
    print("MODEL COMPARISON")
    print("=" * 75)
    print(comparison.round(4).to_string(index=False))
    print(f"\nSaved comparison to {RESULTS_DIR}/model_comparison.csv")
    print("Evaluation completed.")


if __name__ == "__main__":
    main()
