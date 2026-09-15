"""
==========================================================
RetinaSense

Checkpoint Manager

Description:
Handles saving and loading model checkpoints.

==========================================================
"""

from pathlib import Path
import torch

from core.paths import SAVED_MODELS
from core.constants import BEST_MODEL_NAME

CHECKPOINT_NAME = "latest_checkpoint.pth"


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_accuracy,
    history
):
    """
    Save latest training checkpoint.
    """

    SAVED_MODELS.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_accuracy":
            best_accuracy,

        "history":
            history

    }

    torch.save(
        checkpoint,
        SAVED_MODELS / CHECKPOINT_NAME
    )


def save_best_model(model):
    """
    Save only the best model weights.
    """

    SAVED_MODELS.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        SAVED_MODELS / BEST_MODEL_NAME
    )


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    device
):
    """
    Resume training from latest checkpoint.
    """

    checkpoint_file = (
        SAVED_MODELS / CHECKPOINT_NAME
    )

    if not checkpoint_file.exists():

        print("No previous checkpoint found.")

        return (
            0,
            0.0,
            {
                "train_loss": [],
                "train_accuracy": [],
                "validation_loss": [],
                "validation_accuracy": []
            }
        )

    checkpoint = torch.load(
        checkpoint_file,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    print("=" * 70)
    print("Checkpoint Loaded")
    print("=" * 70)

    print(
        f"Resuming from Epoch {checkpoint['epoch'] + 1}"
    )

    print(
        f"Best Accuracy : {checkpoint['best_accuracy']:.2f}%"
    )

    print("=" * 70)

    return (

        checkpoint["epoch"] + 1,

        checkpoint["best_accuracy"],

        checkpoint["history"]

    )