"""
==========================================================
RetinaSense

Grad-CAM Visualization

Description:
Generates Grad-CAM visualization for a retinal
fundus image using the trained EfficientNet-B3 model.

==========================================================
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.efficientnet_b3 import build_model

from core.constants import (
    INDEX_TO_DISEASE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD
)

from utils.transforms import get_test_transforms


# ======================================================
# Grad-CAM Class
# ======================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_hook = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def generate(self, image_tensor, class_index):

        self.model.zero_grad()

        output = self.model(
            image_tensor
        )

        score = output[0, class_index]

        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(dim=1)

        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        cam = cv2.resize(
            cam,
            (
                IMAGE_SIZE[1],
                IMAGE_SIZE[0]
            )
        )

        cam -= cam.min()

        if cam.max() != 0:

            cam /= cam.max()

        return cam

    def close(self):

        self.forward_hook.remove()
        self.backward_hook.remove()


# ======================================================
# Main
# ======================================================

def main():

    print("=" * 70)
    print("RetinaSense Grad-CAM")
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
    # Find Test Image
    # ------------------------------------------

    test_root = Path(
        "datasets/RetinaSense_Dataset/test"
    )

    image_files = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.tif",
        "*.tiff"
    ]:

        image_files.extend(
            test_root.rglob(extension)
        )

    if not image_files:

        raise FileNotFoundError(
            "No test images found."
        )

    image_path = image_files[0]

    print(
        f"Image : {image_path}"
    )

    # ------------------------------------------
    # Read Image
    # ------------------------------------------

    original_image = cv2.imread(
        str(image_path)
    )

    if original_image is None:

        raise ValueError(
            "Unable to read image."
        )

    original_image = cv2.cvtColor(
        original_image,
        cv2.COLOR_BGR2RGB
    )

    # ------------------------------------------
    # Preprocessing
    # ------------------------------------------

    transforms = get_test_transforms()

    transformed = transforms(
        image=original_image
    )

    image_tensor = transformed["image"]

    image_tensor = image_tensor.unsqueeze(
        0
    ).to(device)

    # ------------------------------------------
    # Build Model
    # ------------------------------------------

    model = build_model(
        pretrained=False
    )

    model_path = (
        "saved_models/best_model.pth"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model = model.to(device)

    model.eval()

    print(
        "✅ Trained model loaded."
    )

    # ------------------------------------------
    # Prediction
    # ------------------------------------------

    output = model(
        image_tensor
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()

    disease = INDEX_TO_DISEASE[
        predicted_class
    ]

    print("\n" + "=" * 70)

    print(
        f"Prediction : {disease}"
    )

    print(
        f"Confidence : {confidence * 100:.2f}%"
    )

    print("=" * 70)

    # ------------------------------------------
    # Grad-CAM Target Layer
    # ------------------------------------------

    target_layer = model.features[-1]

    gradcam = GradCAM(
        model,
        target_layer
    )

    # ------------------------------------------
    # Generate Grad-CAM
    # ------------------------------------------

    cam = gradcam.generate(
        image_tensor,
        predicted_class
    )

    gradcam.close()

    # ------------------------------------------
    # Create Heatmap
    # ------------------------------------------

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # ------------------------------------------
    # Resize Original Image
    # ------------------------------------------

    display_image = cv2.resize(
        original_image,
        (
            IMAGE_SIZE[1],
            IMAGE_SIZE[0]
        )
    )

    # ------------------------------------------
    # Overlay
    # ------------------------------------------

    overlay = cv2.addWeighted(
        display_image,
        0.6,
        heatmap,
        0.4,
        0
    )

    # ------------------------------------------
    # Create Results Folder
    # ------------------------------------------

    output_folder = Path(
        "results/gradcam"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------
    # Save Image
    # ------------------------------------------

    output_path = (
        output_folder /
        f"gradcam_{disease}.png"
    )

    plt.figure(
        figsize=(12, 4)
    )

    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        display_image
    )

    plt.title(
        "Original Image"
    )

    plt.axis("off")

    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        heatmap
    )

    plt.title(
        "Grad-CAM"
    )

    plt.axis("off")

    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        overlay
    )

    plt.title(
        f"{disease} - "
        f"{confidence * 100:.1f}%"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\n✅ Grad-CAM saved to:"
        f" {output_path}"
    )

    print("\n" + "=" * 70)
    print("Grad-CAM Completed Successfully")
    print("=" * 70)


# ======================================================
# Run
# ======================================================

if __name__ == "__main__":

    main()