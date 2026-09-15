"""
==========================================================
RetinaSense

Metrics Utility

Description:
Utility functions for evaluating classification models.

==========================================================
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =====================================================
# Accuracy
# =====================================================

def calculate_accuracy(y_true, y_pred):

    return accuracy_score(
        y_true,
        y_pred
    )


# =====================================================
# Precision
# =====================================================

def calculate_precision(y_true, y_pred):

    return precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


# =====================================================
# Recall
# =====================================================

def calculate_recall(y_true, y_pred):

    return recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


# =====================================================
# F1 Score
# =====================================================

def calculate_f1_score(y_true, y_pred):

    return f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


# =====================================================
# Confusion Matrix
# =====================================================

def calculate_confusion_matrix(y_true, y_pred):

    return confusion_matrix(
        y_true,
        y_pred
    )


# =====================================================
# Classification Report
# =====================================================

def generate_classification_report(
    y_true,
    y_pred,
    target_names
):

    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0
    )


# =====================================================
# Complete Metrics
# =====================================================

def calculate_metrics(
    y_true,
    y_pred,
    target_names
):

    return {

        "accuracy":
            calculate_accuracy(
                y_true,
                y_pred
            ),

        "precision":
            calculate_precision(
                y_true,
                y_pred
            ),

        "recall":
            calculate_recall(
                y_true,
                y_pred
            ),

        "f1_score":
            calculate_f1_score(
                y_true,
                y_pred
            ),

        "confusion_matrix":
            calculate_confusion_matrix(
                y_true,
                y_pred
            ),

        "classification_report":
            generate_classification_report(
                y_true,
                y_pred,
                target_names
            )
    }