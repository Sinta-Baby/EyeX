import sys
from pathlib import Path


# ============================================================
# ADD EYEX PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORT EYEX CLASSIFICATION MODULE
# ============================================================

from inference.classification_models import (
    load_classification_models,
    predict_all_models,
    calculate_model_agreement
)


# ============================================================
# TEST IMAGE
# ============================================================

IMAGE_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "RetinaSense_Dataset"
    / "test"
    / "AMD"
    / "ARMD_AMD_1016_img_1255.jpg"
)


# ============================================================
# CHECK TEST IMAGE
# ============================================================

print("=" * 70)
print("EYEX — THREE MODEL PREDICTION TEST")
print("=" * 70)

print(
    "\nTest image:"
)

print(
    IMAGE_PATH
)

print(
    "\nImage exists:",
    IMAGE_PATH.exists()
)

assert IMAGE_PATH.exists(), (
    f"Test image not found: {IMAGE_PATH}"
)


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading classification models...")

models = load_classification_models()

print("Models loaded ✓")


# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\nRunning predictions...")

results = predict_all_models(
    str(IMAGE_PATH),
    models
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

for model_name, result in results.items():

    print("\n" + "-" * 70)

    print(model_name)

    print("-" * 70)

    print(
        "Prediction :",
        result["prediction"]
    )

    print(
        "Confidence :",
        f"{result['confidence'] * 100:.2f}%"
    )

    print("\nClass probabilities:")

    for class_name, probability in (
        result["probabilities"].items()
    ):

        print(
            f"  {class_name:<10}: "
            f"{probability * 100:.2f}%"
        )


# ============================================================
# MODEL AGREEMENT
# ============================================================

agreement = calculate_model_agreement(
    results
)


print("\n" + "=" * 70)
print("MODEL AGREEMENT")
print("=" * 70)

print(
    "Final prediction :",
    agreement["prediction"]
)

print(
    "Agreement        :",
    f"{agreement['agreement_count']}/"
    f"{agreement['total_models']}"
)

print(
    "Agreement ratio  :",
    f"{agreement['agreement_ratio'] * 100:.2f}%"
)

print("=" * 70)

print(
    "\nTHREE MODEL PREDICTION TEST PASSED ✓"
)