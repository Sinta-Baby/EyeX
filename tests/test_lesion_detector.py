import sys
from pathlib import Path

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.lesion_detector import IDRiDLesionDetector


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "best_idrid_unet_final.pth"
)

IMAGE_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "IDRiD"
    / "IDRiD_Segmentation"
    / "1. Original Images"
    / "a. Training Set"
    / "IDRiD_01.jpg"
)


# ---------------------------------------------------------
# Main test
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("EyeX - IDRiD Lesion Detection Test")
    print("=" * 60)

    print("\nModel path:")
    print(MODEL_PATH)

    print("\nImage path:")
    print(IMAGE_PATH)

    # -----------------------------------------------------
    # Check files
    # -----------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_PATH}"
        )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"\nTest image not found:\n{IMAGE_PATH}"
        )

    # -----------------------------------------------------
    # Load trained model
    # -----------------------------------------------------

    print("\nLoading trained IDRiD U-Net...")

    detector = IDRiDLesionDetector(
        model_path=MODEL_PATH
    )

    # -----------------------------------------------------
    # Run prediction
    # -----------------------------------------------------

    print("\nRunning lesion detection...")

    result = detector.predict(
        IMAGE_PATH,
        threshold=0.5
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("LESION DETECTION RESULTS")
    print("=" * 60)

    for lesion_name, stats in result["statistics"].items():

        print(f"\n{lesion_name}")
        print("-" * 30)

        print(
            f"Detected       : {stats['detected']}"
        )

        print(
            f"Lesion pixels  : {stats['lesion_pixels']}"
        )

        print(
            f"Area (%)       : {stats['percentage']:.4f}%"
        )

        print(
            f"Regions        : {stats['regions']}"
        )

    # -----------------------------------------------------
    # Save prediction results
    # -----------------------------------------------------

    saved = detector.save_results(result)

    print("\n" + "=" * 60)
    print("RESULT FILES")
    print("=" * 60)

    print(
        f"\nOriginal:\n{saved['original']}"
    )

    for lesion, path in saved["masks"].items():

        print(
            f"\n{lesion} mask:\n{path}"
        )

    print(
        f"\nOverlay:\n{saved['overlay']}"
    )

    # -----------------------------------------------------
    # Success
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()