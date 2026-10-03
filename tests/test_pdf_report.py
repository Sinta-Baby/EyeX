import sys
import os


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


# ============================================================
# EYEX IMPORTS
# ============================================================

from inference.predict import predict_image
from reports.pdf_report import generate_pdf_report


# ============================================================
# TEST IMAGE
# ============================================================

IMAGE_PATH = (
    r"C:\EyeX\datasets\RetinaSense_Dataset"
    r"\test\AMD\ARMD_AMD_1016_img_1255.jpg"
)


# ============================================================
# TEST PDF OUTPUT
# ============================================================

OUTPUT_PATH = (
    r"C:\EyeX\reports\test_EyeX_report.pdf"
)


# ============================================================
# RUN CLASSIFICATION
# ============================================================

print("Running EyeX classification...")

analysis_result = predict_image(
    IMAGE_PATH
)

print("Classification completed.")

print(
    "Prediction:",
    analysis_result.get("disease")
)

print(
    "Confidence:",
    analysis_result.get("confidence")
)


# ============================================================
# GENERATE PDF
# ============================================================

print("\nGenerating PDF...")

generate_pdf_report(
    output_path=OUTPUT_PATH,
    image_path=IMAGE_PATH,
    analysis_result=analysis_result,
)


# ============================================================
# SUCCESS
# ============================================================

print("\nPDF generated successfully!")

print(
    "Saved to:",
    OUTPUT_PATH
)