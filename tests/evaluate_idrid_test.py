"""
EyeX - IDRiD U-Net Test-Set Evaluation

Evaluates the frozen trained IDRiD lesion segmentation model on the
27 untouched IDRiD testing images.

Metrics:
- Dice
- IoU
- Precision
- Recall

Missing Soft Exudates (SE) ground-truth files are treated as unannotated
and are excluded from the SE metric denominator.

Run from the EyeX project root:
    python tests/evaluate_idrid_test.py

If this file is saved directly as evaluate_idrid_test.py in the project root,
the same command is:
    python evaluate_idrid_test.py
"""

from pathlib import Path
import sys
import csv

import numpy as np
from PIL import Image

# Project root = parent of tests/ when this file is placed in tests/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.lesion_detector import IDRiDLesionDetector


DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "IDRiD"
    / "IDRiD_Segmentation"
)

TEST_IMAGE_DIR = (
    DATASET_ROOT
    / "1. Original Images"
    / "b. Testing Set"
)

TEST_MASK_ROOT = (
    DATASET_ROOT
    / "2. All Segmentation Groundtruths"
    / "b. Testing Set"
)

RESULT_DIR = PROJECT_ROOT / "results" / "idrid"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

LESIONS = [
    ("MA", "Microaneurysms", "1. Microaneurysms"),
    ("HE", "Haemorrhages", "2. Haemorrhages"),
    ("EX", "Hard Exudates", "3. Hard Exudates"),
    ("SE", "Soft Exudates", "4. Soft Exudates"),
]


def load_binary_mask(path: Path):
    """Load an IDRiD mask and convert every annotated pixel to 1."""
    mask = np.array(Image.open(path).convert("L"))
    return (mask > 0).astype(np.uint8)


def find_mask(mask_dir: Path, image_id: str):
    """
    Find the ground-truth mask for one image ID.

    Handles common IDRiD naming variations such as:
    IDRiD_01.tif
    IDRiD_01_MA.tif
    """
    if not mask_dir.exists():
        return None

    candidates = sorted(mask_dir.glob(f"{image_id}*"))
    if not candidates:
        return None

    # Prefer files whose stem starts exactly with the image ID.
    return candidates[0]


def metric_values(pred, gt):
    """Calculate Dice, IoU, precision and recall."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum(dtype=np.int64)
    fp = np.logical_and(pred, ~gt).sum(dtype=np.int64)
    fn = np.logical_and(~pred, gt).sum(dtype=np.int64)

    dice_den = (2 * tp + fp + fn)
    iou_den = (tp + fp + fn)
    precision_den = tp + fp
    recall_den = tp + fn

    dice = (2 * tp / dice_den) if dice_den else 1.0
    iou = (tp / iou_den) if iou_den else 1.0
    precision = (tp / precision_den) if precision_den else 0.0
    recall = (tp / recall_den) if recall_den else 0.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def main():
    print("=" * 72)
    print("EyeX - IDRiD U-Net TEST-SET EVALUATION")
    print("=" * 72)

    if not TEST_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"IDRiD test image directory not found:\n{TEST_IMAGE_DIR}"
        )

    if not TEST_MASK_ROOT.exists():
        raise FileNotFoundError(
            f"IDRiD test mask directory not found:\n{TEST_MASK_ROOT}"
        )

    detector = IDRiDLesionDetector()

    image_paths = sorted(TEST_IMAGE_DIR.glob("*"))
    image_paths = [
        p for p in image_paths
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    ]

    print(f"\nTest images found: {len(image_paths)}")
    if len(image_paths) != 27:
        print("WARNING: Expected 27 IDRiD testing images.")

    all_rows = []
    sums = {
        short_name: {
            "count": 0,
            "dice": [],
            "iou": [],
            "precision": [],
            "recall": [],
        }
        for short_name, _, _ in LESIONS
    }

    for index, image_path in enumerate(image_paths, start=1):
        image_id = image_path.stem
        print(f"\n[{index}/{len(image_paths)}] {image_id}")

        result = detector.predict(
            image_path,
            threshold=0.5
        )

        for channel, (short_name, lesion_name, mask_folder) in enumerate(LESIONS):
            mask_dir = TEST_MASK_ROOT / mask_folder
            mask_path = find_mask(mask_dir, image_id)

            # SE has only 14 annotated testing masks in IDRiD.
            if mask_path is None:
                print(f"  {lesion_name}: unannotated -> skipped")
                continue

            gt = load_binary_mask(mask_path)

            pred = result["binary_masks"][channel]
            if pred.shape != gt.shape:
                raise ValueError(
                    f"Shape mismatch for {image_id} {lesion_name}: "
                    f"prediction={pred.shape}, ground_truth={gt.shape}"
                )

            metrics = metric_values(pred, gt)

            sums[short_name]["count"] += 1
            for key in ("dice", "iou", "precision", "recall"):
                sums[short_name][key].append(metrics[key])

            all_rows.append({
                "image_id": image_id,
                "lesion": short_name,
                "lesion_name": lesion_name,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            })

            print(
                f"  {short_name}: "
                f"Dice={metrics['dice']:.4f}  "
                f"IoU={metrics['iou']:.4f}  "
                f"Precision={metrics['precision']:.4f}  "
                f"Recall={metrics['recall']:.4f}"
            )

    print("\n" + "=" * 72)
    print("TEST-SET SUMMARY")
    print("=" * 72)

    summary_rows = []

    for short_name, lesion_name, _ in LESIONS:
        count = sums[short_name]["count"]

        if count == 0:
            print(f"{lesion_name}: no annotated test images")
            continue

        row = {
            "lesion": short_name,
            "lesion_name": lesion_name,
            "annotated_images": count,
            "dice": float(np.mean(sums[short_name]["dice"])),
            "iou": float(np.mean(sums[short_name]["iou"])),
            "precision": float(np.mean(sums[short_name]["precision"])),
            "recall": float(np.mean(sums[short_name]["recall"])),
        }
        summary_rows.append(row)

        print(
            f"{lesion_name:18s} | "
            f"N={count:2d} | "
            f"Dice={row['dice']:.4f} | "
            f"IoU={row['iou']:.4f} | "
            f"Precision={row['precision']:.4f} | "
            f"Recall={row['recall']:.4f}"
        )

    # Macro-average across lesion classes using the per-class means.
    if summary_rows:
        macro = {
            "dice": float(np.mean([r["dice"] for r in summary_rows])),
            "iou": float(np.mean([r["iou"] for r in summary_rows])),
            "precision": float(np.mean([r["precision"] for r in summary_rows])),
            "recall": float(np.mean([r["recall"] for r in summary_rows])),
        }

        print("-" * 72)
        print(
            f"{'MACRO AVERAGE':18s} | "
            f"Dice={macro['dice']:.4f} | "
            f"IoU={macro['iou']:.4f} | "
            f"Precision={macro['precision']:.4f} | "
            f"Recall={macro['recall']:.4f}"
        )

    detail_path = RESULT_DIR / "idrid_test_metrics.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "lesion",
                "lesion_name",
                "dice",
                "iou",
                "precision",
                "recall",
                "tp",
                "fp",
                "fn",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = RESULT_DIR / "idrid_test_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lesion",
                "lesion_name",
                "annotated_images",
                "dice",
                "iou",
                "precision",
                "recall",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSaved:")
    print(detail_path)
    print(summary_path)
    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
