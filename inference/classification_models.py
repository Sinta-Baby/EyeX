import torch
import cv2
import numpy as np

from models.efficientnet_b3 import build_model as build_efficientnet
from models.convnext_tiny import build_model as build_convnext
from models.swin_tiny import build_model as build_swin

from utils.transforms import get_test_transforms


# ============================================================
# CLASS MAPPINGS
# ============================================================

# Existing RetinaSense EfficientNet-B3 mapping
EFFICIENTNET_CLASS_NAMES = [
    "Healthy",
    "DR",
    "Glaucoma",
    "AMD"
]

# ConvNeXt-Tiny and Swin-Tiny were trained using ImageFolder.
# ImageFolder sorts folders alphabetically:
# AMD -> 0
# DR -> 1
# Glaucoma -> 2
# Healthy -> 3
NEW_MODEL_CLASS_NAMES = [
    "AMD",
    "DR",
    "Glaucoma",
    "Healthy"
]


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIGS = {
    "EfficientNet-B3": {
        "builder": build_efficientnet,
        "checkpoint": "saved_models/best_model.pth",
        "input_size": 300,
    },

    "ConvNeXt-Tiny": {
        "builder": build_convnext,
        "checkpoint": "saved_models/best_convnext_tiny.pth",
        "input_size": 224,
    },

    "Swin-Tiny": {
        "builder": build_swin,
        "checkpoint": "saved_models/best_swin_tiny.pth",
        "input_size": 224,
    },
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_name):

    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    config = MODEL_CONFIGS[model_name]

    model = config["builder"](pretrained=False)

    checkpoint = torch.load(
        config["checkpoint"],
        map_location=DEVICE
    )

    # Handle both direct state_dict and checkpoint dictionaries
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image_path, model_name):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # EfficientNet-B3
    # IMPORTANT:
    # Use the ORIGINAL RetinaSense preprocessing.
    # --------------------------------------------------------

    if model_name == "EfficientNet-B3":

        transform = get_test_transforms()

        transformed = transform(
            image=image_rgb
        )

        tensor = transformed["image"]

    # --------------------------------------------------------
    # ConvNeXt-Tiny / Swin-Tiny
    # These models were trained using ImageFolder with
    # 224x224 ImageNet preprocessing.
    # --------------------------------------------------------

    else:

        resized = cv2.resize(
            image_rgb,
            (224, 224)
        )

        resized = resized.astype(
            np.float32
        ) / 255.0

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        normalized = (
            resized - mean
        ) / std

        tensor = torch.from_numpy(
            normalized
        ).permute(2, 0, 1).float()

    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# PREDICT USING ONE MODEL
# ============================================================

def predict_single_model(image_path, model_name):

    model = load_model(model_name)

    image_tensor = preprocess_image(
        image_path,
        model_name
    )

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    predicted_index = torch.argmax(
        probabilities
    ).item()

    # --------------------------------------------------------
    # IMPORTANT:
    # Use the correct class mapping for each model.
    # --------------------------------------------------------

    if model_name == "EfficientNet-B3":

        class_names = EFFICIENTNET_CLASS_NAMES

    else:

        class_names = NEW_MODEL_CLASS_NAMES

    predicted_class = class_names[
        predicted_index
    ]

    confidence = probabilities[
        predicted_index
    ].item()

    probability_dict = {
        class_names[i]: probabilities[i].item()
        for i in range(len(class_names))
    }

    return {
        "model": model_name,
        "prediction": predicted_class,
        "confidence": confidence,
        "probabilities": probability_dict
    }


# ============================================================
# RUN ALL THREE MODELS
# ============================================================

def predict_all_models(image_path):

    results = {}

    for model_name in MODEL_CONFIGS:

        results[model_name] = predict_single_model(
            image_path,
            model_name
        )

    return results


# ============================================================
# MODEL AGREEMENT
# ============================================================

def calculate_model_agreement(results):

    predictions = [
        result["prediction"]
        for result in results.values()
    ]

    prediction_counts = {}

    for prediction in predictions:

        prediction_counts[prediction] = (
            prediction_counts.get(prediction, 0) + 1
        )

    final_prediction = max(
        prediction_counts,
        key=prediction_counts.get
    )

    agreement_count = prediction_counts[
        final_prediction
    ]

    total_models = len(predictions)

    agreement_ratio = (
        agreement_count / total_models
    )

    return {
        "final_prediction": final_prediction,
        "agreement_count": agreement_count,
        "total_models": total_models,
        "agreement_ratio": agreement_ratio,
    }