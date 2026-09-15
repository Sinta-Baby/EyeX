"""
==========================================================
RetinaSense

History Manager

Description:
Stores and saves training history.

==========================================================
"""

import pandas as pd

from core.paths import REPORTS


HISTORY_FILE = REPORTS / "history.csv"


def create_history():
    """
    Create empty training history.
    """

    return {

        "train_loss": [],

        "train_accuracy": [],

        "validation_loss": [],

        "validation_accuracy": []

    }


def update_history(

    history,

    train_loss,

    train_accuracy,

    validation_loss,

    validation_accuracy

):
    """
    Add one epoch to history.
    """

    history["train_loss"].append(train_loss)

    history["train_accuracy"].append(train_accuracy)

    history["validation_loss"].append(validation_loss)

    history["validation_accuracy"].append(
        validation_accuracy
    )

    return history


def save_history(history):
    """
    Save history.csv
    """

    REPORTS.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame(history)

    df.to_csv(
        HISTORY_FILE,
        index=False
    )

    print("History saved.")


def load_history():
    """
    Load previous history if available.
    """

    if not HISTORY_FILE.exists():

        return create_history()

    df = pd.read_csv(HISTORY_FILE)

    return {

        "train_loss":
            df["train_loss"].tolist(),

        "train_accuracy":
            df["train_accuracy"].tolist(),

        "validation_loss":
            df["validation_loss"].tolist(),

        "validation_accuracy":
            df["validation_accuracy"].tolist()

    }