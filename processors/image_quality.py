"""
==========================================================
EyeX

Image Quality Assessment

Description:
Checks whether a retinal fundus image is suitable
for further AI analysis.

The module checks:
- Image resolution
- Sharpness
- Brightness
- Contrast

==========================================================
"""

from pathlib import Path

import cv2
import numpy as np


# ======================================================
# Quality Thresholds
# ======================================================

MIN_WIDTH = 200
MIN_HEIGHT = 200

MIN_SHARPNESS = 50.0

MIN_BRIGHTNESS = 30.0
MAX_BRIGHTNESS = 220.0

MIN_CONTRAST = 20.0


# ======================================================
# Calculate Image Quality
# ======================================================

def assess_image_quality(image_path):
    """
    Assess the quality of a fundus image.

    Returns a dictionary containing the individual
    quality measurements and an overall quality result.
    """

    image_path = Path(image_path)

    # ------------------------------------------
    # Check image path
    # ------------------------------------------

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # ------------------------------------------
    # Read image
    # ------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            f"Unable to read image:\n{image_path}"
        )

    # ------------------------------------------
    # Image dimensions
    # ------------------------------------------

    height, width = image.shape[:2]

    resolution_ok = (
        width >= MIN_WIDTH
        and
        height >= MIN_HEIGHT
    )

    # ------------------------------------------
    # Convert to grayscale
    # ------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # ------------------------------------------
    # Sharpness
    # ------------------------------------------

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    sharpness_ok = (
        sharpness >= MIN_SHARPNESS
    )

    # ------------------------------------------
    # Brightness
    # ------------------------------------------

    brightness = float(
        np.mean(gray)
    )

    brightness_ok = (
        MIN_BRIGHTNESS
        <= brightness
        <= MAX_BRIGHTNESS
    )

    # ------------------------------------------
    # Contrast
    # ------------------------------------------

    contrast = float(
        np.std(gray)
    )

    contrast_ok = (
        contrast >= MIN_CONTRAST
    )

    # ------------------------------------------
    # Overall quality
    # ------------------------------------------

    checks = [
        resolution_ok,
        sharpness_ok,
        brightness_ok,
        contrast_ok
    ]

    passed_checks = sum(checks)

    if passed_checks == 4:

        quality = "Good"

    elif passed_checks >= 2:

        quality = "Fair"

    else:

        quality = "Poor"

    # ------------------------------------------
    # Return result
    # ------------------------------------------

    return {

        "quality": quality,

        "width": width,

        "height": height,

        "sharpness": sharpness,

        "brightness": brightness,

        "contrast": contrast,

        "resolution_ok": resolution_ok,

        "sharpness_ok": sharpness_ok,

        "brightness_ok": brightness_ok,

        "contrast_ok": contrast_ok,

        "passed_checks": passed_checks

    }


# ======================================================
# Test
# ======================================================

if __name__ == "__main__":

    image_path = (
        "datasets/RetinaSense_Dataset/test/AMD/"
        "ARMD_AMD_1016_img_1255.jpg"
    )

    result = assess_image_quality(
        image_path
    )

    print("=" * 60)
    print("EyeX Image Quality Assessment")
    print("=" * 60)

    print(
        f"Image Size : "
        f"{result['width']} × "
        f"{result['height']}"
    )

    print(
        f"Sharpness  : "
        f"{result['sharpness']:.2f}"
    )

    print(
        f"Brightness : "
        f"{result['brightness']:.2f}"
    )

    print(
        f"Contrast   : "
        f"{result['contrast']:.2f}"
    )

    print(
        f"Quality    : "
        f"{result['quality']}"
    )

    print("=" * 60)