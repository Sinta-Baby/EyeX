"""
==========================================================
RetinaSense

Main Training Script

Description:
Loads the RetinaSense dataset, builds the model,
and starts training.

==========================================================
"""

import torch

from core.constants import RANDOM_SEED
from core.paths import RETINASENSE_DATASET

from models.efficientnet_b3 import build_model
from utils.dataloader import get_dataloaders
from engine.trainer import Trainer


# =====================================================
# Main
# =====================================================

def main():

    # -------------------------------------------------
    # Reproducibility
    # -------------------------------------------------

    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # -------------------------------------------------
    # Device
    # -------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n" + "=" * 70)
    print("RetinaSense")
    print("=" * 70)

    print(f"Device : {device}")

    if torch.cuda.is_available():

        print(
            f"GPU : {torch.cuda.get_device_name(0)}"
        )

    else:

        print("GPU not available. Using CPU.")

    print("=" * 70)

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    dataset_root = str(
        RETINASENSE_DATASET
    )

    train_loader, validation_loader, test_loader = (
        get_dataloaders(
            dataset_root
        )
    )

    print("Dataset Loaded Successfully.")

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = build_model()

    print("Model Built Successfully.")

    # -------------------------------------------------
    # Trainer
    # -------------------------------------------------

    trainer = Trainer(

        model=model,

        train_loader=train_loader,

        validation_loader=validation_loader,

        device=device

    )

    # -------------------------------------------------
    # Start Training
    # -------------------------------------------------

    trainer.train()


# =====================================================
# Run
# =====================================================

if __name__ == "__main__":

    main()