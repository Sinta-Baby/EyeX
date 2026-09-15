"""
==========================================================
RetinaSense

Training Engine

Description:
Professional training engine for RetinaSense.

Features
--------
✓ GPU / CPU Support
✓ Resume Training
✓ Checkpoint Saving
✓ Early Stopping
✓ Training History
✓ Learning Rate Scheduler

==========================================================
"""

from pathlib import Path

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from core.constants import (
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BEST_MODEL_NAME,
    EARLY_STOPPING_PATIENCE
)

from core.paths import (
    SAVED_MODELS,
    TRAINING_HISTORY
)


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
        device
    ):

        self.model = model

        self.train_loader = train_loader

        self.validation_loader = validation_loader

        self.device = device

        self.model.to(device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=NUM_EPOCHS
        )

        self.best_accuracy = 0.0

        self.current_epoch = 0

        self.early_stopping_counter = 0

        self.history = {

            "epoch": [],

            "train_loss": [],

            "validation_loss": [],

            "train_accuracy": [],

            "validation_accuracy": []

        }

    # =====================================================
    # Save Checkpoint
    # =====================================================

    def save_checkpoint(
        self,
        epoch,
        best_accuracy
    ):
        SAVED_MODELS.mkdir(
            parents=True,
            exist_ok=True
        )
        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "scheduler_state_dict":
                self.scheduler.state_dict(),

            "best_accuracy":
                best_accuracy,
            
            "history":
                self.history
        }

        torch.save(
            checkpoint,
            SAVED_MODELS / "latest_checkpoint.pth"
        )

    # =====================================================
    # Resume Checkpoint
    # =====================================================

    def load_checkpoint(self):

        checkpoint_path = (
            SAVED_MODELS /
            "latest_checkpoint.pth"
        )

        if checkpoint_path.exists():

            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device
            )

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            self.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

            self.scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

            self.current_epoch = checkpoint["epoch"]

            self.best_accuracy = checkpoint["best_accuracy"]
            
            self.history = checkpoint.get(
                "history",
                self.history
            )

            print("=" * 70)
            print("Checkpoint Loaded")
            print("=" * 70)

            print(
                f"Resuming from Epoch {self.current_epoch + 1}"
            )

            print(
                f"Best Accuracy : {self.best_accuracy:.2f}%"
            )

            print("=" * 70)

        else:

            print(
                "No previous checkpoint found."
            )
            
    # =====================================================
    # Train One Epoch
    # =====================================================

    def train_epoch(self):

        self.model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        progress = tqdm(
            self.train_loader,
            desc="Training",
            leave=False
        )

        for images, labels in progress:

            images = images.to(self.device)

            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels
            )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        epoch_loss = (
            running_loss /
            len(self.train_loader)
        )

        epoch_accuracy = (
            100.0 *
            correct /
            total
        )

        return (
            epoch_loss,
            epoch_accuracy
        )

    # =====================================================
    # Validation
    # =====================================================

    def validate(self):

        self.model.eval()

        running_loss = 0.0

        correct = 0

        total = 0

        with torch.no_grad():

            progress = tqdm(
                self.validation_loader,
                desc="Validation",
                leave=False
            )

            for images, labels in progress:

                images = images.to(self.device)

                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    labels
                )

                running_loss += loss.item()

                predictions = outputs.argmax(dim=1)

                correct += (
                    predictions == labels
                ).sum().item()

                total += labels.size(0)

                progress.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

        epoch_loss = (
            running_loss /
            len(self.validation_loader)
        )

        epoch_accuracy = (
            100.0 *
            correct /
            total
        )

        return (
            epoch_loss,
            epoch_accuracy
        )

    # =====================================================
    # Save Training History
    # =====================================================

    def save_history(self):
        
        TRAINING_HISTORY.mkdir(
            parents=True,
            exist_ok=True
        )

        history = pd.DataFrame(
            self.history
        )

        history.to_csv(
            TRAINING_HISTORY / "history.csv",
            index=False
        )
        
        
    # =====================================================
    # Complete Training
    # =====================================================

    def train(self):

        self.load_checkpoint()

        print("\n" + "=" * 70)
        print("RetinaSense Training")
        print("=" * 70)

        print(f"Device : {self.device}")

        print(f"Epochs : {NUM_EPOCHS}")

        print("=" * 70)

        for epoch in range(
            self.current_epoch,
            NUM_EPOCHS
        ):

            print(
                f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
            )

            print("-" * 70)

            train_loss, train_accuracy = (
                self.train_epoch()
            )

            validation_loss, validation_accuracy = (
                self.validate()
            )

            self.scheduler.step()

            # ------------------------------------------
            # Store Training History
            # ------------------------------------------

            self.history["epoch"].append(
                epoch + 1
            )

            self.history["train_loss"].append(
                train_loss
            )

            self.history["validation_loss"].append(
                validation_loss
            )

            self.history["train_accuracy"].append(
                train_accuracy
            )

            self.history["validation_accuracy"].append(
                validation_accuracy
            )

            # ------------------------------------------
            # Print Results
            # ------------------------------------------

            print(
                f"Train Loss          : {train_loss:.4f}"
            )

            print(
                f"Train Accuracy      : {train_accuracy:.2f}%"
            )

            print(
                f"Validation Loss     : {validation_loss:.4f}"
            )

            print(
                f"Validation Accuracy : {validation_accuracy:.2f}%"
            )

            print(
                f"Learning Rate       : "
                f"{self.optimizer.param_groups[0]['lr']:.7f}"
            )

            

            # ------------------------------------------
            # Save Best Model
            # ------------------------------------------

            if validation_accuracy > self.best_accuracy:

                self.best_accuracy = validation_accuracy

                torch.save(
                    self.model.state_dict(),
                    SAVED_MODELS / BEST_MODEL_NAME
                )

                self.early_stopping_counter = 0

                print(
                    "\n✅ Best model updated."
                )

            else:

                self.early_stopping_counter += 1

                print(
                    f"\nNo improvement "
                    f"({self.early_stopping_counter}/"
                    f"{EARLY_STOPPING_PATIENCE})"
                )

                if (
                    self.early_stopping_counter >=
                    EARLY_STOPPING_PATIENCE
                ):

                    print(
                        "\n🛑 Early stopping activated."
                    )

                    break
            # ------------------------------------------
            # Save Latest Checkpoint (Every Epoch)
            # ------------------------------------------
            
            self.save_checkpoint(
                    epoch + 1,
                    self.best_accuracy
                 
            )
            # ------------------------------------------
            # Save Training History
            # ------------------------------------------

            self.save_history()

        print("\n" + "=" * 70)

        print("Training Finished")

        print(
            f"Best Validation Accuracy : "
            f"{self.best_accuracy:.2f}%"
        )

        print("=" * 70)

        from utils.visualization import plot_training_history

        plot_training_history(
            TRAINING_HISTORY / "history.csv"
        )