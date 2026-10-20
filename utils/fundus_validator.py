
"""EyeX fundus image input validation.

This is a separate input gate. It does not modify or retrain
the existing EyeX disease-classification models.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from PIL import Image, ImageStat


FUNDUS_PROMPT = (
    "a retinal fundus photograph of the inside of a human eye, "
    "showing retinal blood vessels and the optic disc"
)

OTHER_PROMPTS = [
    "a photograph of a human face",
    "a photograph of a dog",
    "a photograph of a truck or car",
    "a photograph of a landscape",
    "a photograph of an everyday object",
    "a medical image that is not a retinal fundus photograph",
]

PROMPTS = [FUNDUS_PROMPT, *OTHER_PROMPTS]


@dataclass
class ValidationResult:
    status: str
    message: str
    fundus_score: Optional[float] = None
    best_other_score: Optional[float] = None


@lru_cache(maxsize=1)
def load_validator_model():
    """Load a separate pretrained image-recognition model once."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-base-patch32"

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()

    return torch, processor, model


def decide_from_scores(
    scores,
    accept_threshold=0.30,
    reject_margin=0.08,
):
    """Convert model scores into valid, invalid, or uncertain."""
    if len(scores) != len(PROMPTS):
        raise ValueError(
            f"Expected {len(PROMPTS)} scores, got {len(scores)}"
        )

    fundus_score = float(scores[0])
    best_other_score = max(float(x) for x in scores[1:])

    if (
        fundus_score >= accept_threshold
        and fundus_score > best_other_score
    ):
        return ValidationResult(
            status="valid",
            message="The image appears to be a retinal fundus image.",
            fundus_score=fundus_score,
            best_other_score=best_other_score,
        )

    if best_other_score >= fundus_score + reject_margin:
        return ValidationResult(
            status="invalid",
            message=(
                "This does not appear to be a retinal fundus image. "
                "EyeX analyzes retinal fundus images only. "
                "Please upload a valid retinal image."
            ),
            fundus_score=fundus_score,
            best_other_score=best_other_score,
        )

    return ValidationResult(
        status="uncertain",
        message=(
            "EyeX could not confidently verify this image as a "
            "retinal fundus image. Please upload a clear, centered "
            "retinal fundus photograph."
        ),
        fundus_score=fundus_score,
        best_other_score=best_other_score,
    )


def validate_fundus_image(image: Image.Image) -> ValidationResult:
    """Validate a PIL image before it enters the EyeX pipeline."""

    if not isinstance(image, Image.Image):
        return ValidationResult(
            "invalid",
            "The uploaded file could not be read as an image.",
        )

    if image.width < 64 or image.height < 64:
        return ValidationResult(
            "invalid",
            "The image is too small. Please upload a clearer retinal image.",
        )

    # Catch blank or nearly uniform images.
    if max(ImageStat.Stat(image.convert("RGB")).stddev) < 1.0:
        return ValidationResult(
            "invalid",
            "The image appears blank. Please upload a retinal fundus image.",
        )

    try:
        torch, processor, model = load_validator_model()

        inputs = processor(
            text=PROMPTS,
            images=image.convert("RGB"),
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            scores = outputs.logits_per_image.softmax(dim=1)[0].tolist()

        return decide_from_scores(scores)

    except (ImportError, OSError, RuntimeError, ValueError):
        # Do not silently run disease prediction when validation fails.
        return ValidationResult(
            "uncertain",
            "EyeX could not verify the image type. "
            "Check the validator dependencies and model download, "
            "then try again.",
        )
