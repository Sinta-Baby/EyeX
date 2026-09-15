Try AI directly in your favorite apps … Use Gemini to generate drafts and refine content, plus get Gemini Pro with access to Google's next-gen AI for ₹1,950 ₹489 for 3 months
100%
import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        predictions,
        targets,
        valid
    ):

        # Always calculate loss in FP32.
        predictions = predictions.float()
        targets = targets.float()
        valid = valid.float()

        predictions = torch.sigmoid(
            predictions
        )

        dice_scores = []

        for channel in range(
            predictions.shape[1]
        ):

            valid_samples = (
                valid[:, channel] > 0
            )

            if valid_samples.sum().item() == 0:
                continue

            pred = predictions[
                valid_samples,
                channel
            ]

            target = targets[
                valid_samples,
                channel
            ]

            pred = pred.reshape(
                pred.shape[0],
                -1
            )

            target = target.reshape(
                target.shape[0],
                -1
            )

            # Only use patches containing
            # actual foreground for Dice.
            positive_samples = (
                target.sum(dim=1) > 0
            )

            if positive_samples.sum().item() == 0:
                continue

            pred = pred[
                positive_samples
            ]

            target = target[
                positive_samples
            ]

            intersection = (
                pred * target
            ).sum(dim=1)

            dice = (
                2.0 * intersection
                + self.smooth
            ) / (
                pred.sum(dim=1)
                + target.sum(dim=1)
                + self.smooth
            )

            dice_scores.append(
                dice.mean()
            )

        if len(dice_scores) == 0:

            return predictions.sum() * 0.0

        dice_score = torch.stack(
            dice_scores
        ).mean()

        return 1.0 - dice_score


class IDRiDLoss(nn.Module):

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss(
            reduction="none"
        )

        self.dice = DiceLoss()

        # Moderated positive weights.
        #
        # Order:
        # MA, HE, EX, SE

        self.register_buffer(
            "pos_weight",
            torch.tensor(
                [
                    20.0,
                    10.0,
                    10.0,
                    15.0
                ],
                dtype=torch.float32
            )
        )

    def forward(
        self,
        predictions,
        targets,
        valid
    ):

        # Always calculate loss in FP32.
        predictions = predictions.float()
        targets = targets.float()
        valid = valid.float()

        # -----------------------------------------
        # BCE
        # -----------------------------------------

        bce = self.bce(
            predictions,
            targets
        )

        weights = torch.ones_like(
            targets
        )

        for channel in range(4):

            weights[:, channel] = torch.where(
                targets[:, channel] > 0,
                self.pos_weight[channel],
                1.0
            )

        bce = bce * weights

        # -----------------------------------------
        # Ignore missing annotations
        # -----------------------------------------

        valid_pixels = valid[
            :, :, None, None
        ]

        bce = bce * valid_pixels

        valid_count = (
            valid_pixels.sum()
            * predictions.shape[2]
            * predictions.shape[3]
        )

        if valid_count.item() > 0:

            bce = (
                bce.sum()
                / valid_count
            )

        else:

            bce = predictions.sum() * 0.0

        # -----------------------------------------
        # Foreground-aware Dice
        # -----------------------------------------

        dice = self.dice(
            predictions,
            targets,
            valid
        )

        # -----------------------------------------
        # Combined loss
        # -----------------------------------------

        total_loss = (
            self.bce_weight * bce
            + self.dice_weight * dice
        )

        return total_loss