import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.idrid_unet import IDRiDUNet


class IDRiDLesionDetector:
    """
    IDRiD U-Net inference engine.

    Detects four retinal lesion types:
        0 - Microaneurysms (MA)
        1 - Haemorrhages (HE)
        2 - Hard Exudates (EX)
        3 - Soft Exudates (SE)

    The trained model works on 512x512 patches.
    For a complete fundus image, overlapping patches are
    predicted and then stitched together.
    """

    LESION_NAMES = [
        "Microaneurysms",
        "Haemorrhages",
        "Hard Exudates",
        "Soft Exudates",
    ]

    SHORT_NAMES = [
        "MA",
        "HE",
        "EX",
        "SE",
    ]

    PATCH_SIZE = 512
    STRIDE = 384
    NUM_CLASSES = 4

    def __init__(self, model_path=None, device=None):
        """
        Load the trained IDRiD U-Net model.
        """

        if model_path is None:
            model_path = (
                PROJECT_ROOT
                / "saved_models"
                / "best_idrid_unet_final.pth"
            )

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found:\n{self.model_path}"
            )

        # Select device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Build the exact architecture used during training
        self.model = IDRiDUNet()

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device
        )

        # Support both a raw state_dict and a checkpoint dictionary
        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            else:
                state_dict = checkpoint

        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        print("IDRiD lesion model loaded successfully.")
        print(f"Model : {self.model_path}")
        print(f"Device: {self.device}")

    # ---------------------------------------------------------
    # Image preparation
    # ---------------------------------------------------------

    def load_image(self, image_path):
        """
        Load a fundus image as RGB numpy array.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        return image

    def image_to_tensor(self, image):
        """
        Convert an RGB numpy image in [0,255] to
        a PyTorch tensor in [0,1].
        """

        image = image.astype(np.float32) / 255.0

        tensor = torch.from_numpy(
            image.transpose(2, 0, 1)
        )

        return tensor

    # ---------------------------------------------------------
    # Patch extraction
    # ---------------------------------------------------------

    def get_positions(self, length):
        """
        Generate overlapping patch positions.

        Ensures the final patch reaches the image boundary.
        """

        if length <= self.PATCH_SIZE:
            return [0]

        positions = list(
            range(
                0,
                length - self.PATCH_SIZE + 1,
                self.STRIDE
            )
        )

        last_position = length - self.PATCH_SIZE

        if positions[-1] != last_position:
            positions.append(last_position)

        return positions

    def extract_patches(self, image):
        """
        Extract overlapping 512x512 patches.

        Returns:
            patches
            positions
            original_height
            original_width
        """

        height, width = image.shape[:2]

        # Pad if image is smaller than patch size
        padded_height = max(height, self.PATCH_SIZE)
        padded_width = max(width, self.PATCH_SIZE)

        if padded_height != height or padded_width != width:

            padded_image = np.zeros(
                (padded_height, padded_width, 3),
                dtype=np.uint8
            )

            padded_image[:height, :width] = image

            image = padded_image

        y_positions = self.get_positions(padded_height)
        x_positions = self.get_positions(padded_width)

        patches = []
        positions = []

        for y in y_positions:
            for x in x_positions:

                patch = image[
                    y:y + self.PATCH_SIZE,
                    x:x + self.PATCH_SIZE
                ]

                patches.append(patch)
                positions.append((x, y))

        return (
            patches,
            positions,
            height,
            width
        )

    # ---------------------------------------------------------
    # Model prediction
    # ---------------------------------------------------------

    @torch.no_grad()
    def predict_patches(self, patches):
        """
        Run the trained U-Net on all patches.
        """

        predictions = []

        for i in range(0, len(patches), 4):

            batch_patches = patches[i:i + 4]

            batch = torch.stack(
                [
                    self.image_to_tensor(patch)
                    for patch in batch_patches
                ]
            )

            batch = batch.to(self.device)

            logits = self.model(batch)

            probabilities = torch.sigmoid(logits)

            probabilities = (
                probabilities
                .cpu()
                .numpy()
            )

            predictions.append(probabilities)

        return np.concatenate(
            predictions,
            axis=0
        )

    # ---------------------------------------------------------
    # Stitch patches
    # ---------------------------------------------------------

    def stitch_predictions(
        self,
        patch_predictions,
        positions,
        height,
        width
    ):
        """
        Stitch overlapping patch predictions into
        full-resolution probability maps.
        """

        padded_height = max(height, self.PATCH_SIZE)
        padded_width = max(width, self.PATCH_SIZE)

        probability_sum = np.zeros(
            (
                self.NUM_CLASSES,
                padded_height,
                padded_width
            ),
            dtype=np.float32
        )

        count_map = np.zeros(
            (
                padded_height,
                padded_width
            ),
            dtype=np.float32
        )

        for prediction, (x, y) in zip(
            patch_predictions,
            positions
        ):

            probability_sum[
                :,
                y:y + self.PATCH_SIZE,
                x:x + self.PATCH_SIZE
            ] += prediction

            count_map[
                y:y + self.PATCH_SIZE,
                x:x + self.PATCH_SIZE
            ] += 1.0

        count_map = np.maximum(
            count_map,
            1.0
        )

        probability_maps = (
            probability_sum
            / count_map[None, :, :]
        )

        # Remove any padding
        probability_maps = probability_maps[
            :,
            :height,
            :width
        ]

        return probability_maps

    # ---------------------------------------------------------
    # Complete prediction
    # ---------------------------------------------------------

    def predict(self, image_path, threshold=0.5):
        """
        Complete lesion prediction pipeline.

        Returns:
            dictionary containing:
                original image
                probability maps
                binary masks
                lesion statistics
        """

        image = self.load_image(image_path)

        patches, positions, height, width = (
            self.extract_patches(image)
        )

        print(f"Image size : {width} x {height}")
        print(
            f"Patch size : "
            f"{self.PATCH_SIZE} x {self.PATCH_SIZE}"
        )
        print(
            f"Number of patches: {len(patches)}"
        )

        patch_predictions = self.predict_patches(
            patches
        )

        probability_maps = self.stitch_predictions(
            patch_predictions,
            positions,
            height,
            width
        )

        binary_masks = (
            probability_maps >= threshold
        ).astype(np.uint8)

        lesion_statistics = (
            self.calculate_statistics(
                binary_masks
            )
        )

        return {
            "image": image,
            "probability_maps": probability_maps,
            "binary_masks": binary_masks,
            "statistics": lesion_statistics,
            "image_path": str(image_path),
            "threshold": threshold,
        }

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def calculate_statistics(self, binary_masks):
        """
        Calculate model-detected lesion statistics.
        """

        statistics = {}

        for channel, name in enumerate(
            self.LESION_NAMES
        ):

            mask = binary_masks[channel]

            lesion_pixels = int(
                np.sum(mask)
            )

            total_pixels = (
                mask.shape[0] * mask.shape[1]
            )

            percentage = (
                lesion_pixels
                / total_pixels
                * 100.0
            )

            # Connected components
            num_labels, _, _, _ = (
                cv2.connectedComponentsWithStats(
                    mask,
                    connectivity=8
                )
            )

            # Subtract background component
            region_count = max(
                0,
                num_labels - 1
            )

            statistics[name] = {
                "short_name": self.SHORT_NAMES[channel],
                "detected": lesion_pixels > 0,
                "lesion_pixels": lesion_pixels,
                "percentage": percentage,
                "regions": region_count,
            }

        return statistics

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------

    def create_overlay(
        self,
        result,
        alpha=0.45
    ):
        """
        Create a lesion overlay on the original fundus image.

        Different lesion classes are represented by different
        channels in the output image.
        """

        image = result["image"].copy()

        masks = result["binary_masks"]

        overlay = image.astype(
            np.float32
        )

        # BGR-style colors for OpenCV
        # MA -> red
        # HE -> green
        # EX -> blue
        # SE -> yellow

        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
        ]

        for channel, color in enumerate(
            colors
        ):

            mask = masks[channel] > 0

            if not np.any(mask):
                continue

            color_array = np.array(
                color,
                dtype=np.float32
            )

            overlay[mask] = (
                (1.0 - alpha)
                * overlay[mask]
                + alpha
                * color_array
            )

        overlay = np.clip(
            overlay,
            0,
            255
        ).astype(np.uint8)

        return overlay

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    def save_results(
        self,
        result,
        output_dir=None
    ):
        """
        Save original image, individual lesion masks,
        and combined overlay.
        """

        if output_dir is None:
            output_dir = (
                PROJECT_ROOT
                / "results"
                / "idrid"
                / "predictions"
            )

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        image_path = Path(
            result["image_path"]
        )

        image_id = image_path.stem

        # Save original
        original_path = (
            output_dir
            / f"{image_id}_original.png"
        )

        Image.fromarray(
            result["image"]
        ).save(original_path)

        # Save individual masks
        mask_paths = {}

        for channel, short_name in enumerate(
            self.SHORT_NAMES
        ):

            mask_path = (
                output_dir
                / f"{image_id}_{short_name}_mask.png"
            )

            mask = (
                result["binary_masks"][channel]
                * 255
            ).astype(np.uint8)

            Image.fromarray(
                mask
            ).save(mask_path)

            mask_paths[short_name] = str(
                mask_path
            )

        # Save overlay
        overlay = self.create_overlay(
            result
        )

        overlay_path = (
            output_dir
            / f"{image_id}_lesion_overlay.png"
        )

        Image.fromarray(
            overlay
        ).save(overlay_path)

        return {
            "original": str(original_path),
            "masks": mask_paths,
            "overlay": str(overlay_path),
        }