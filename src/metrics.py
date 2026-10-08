from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metric_set(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(np.int8)
    report = classification_report(
        y,
        prediction,
        labels=[0, 1],
        target_names=["earthquake", "explosion"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "n_events": int(len(y)),
        "class_counts": {
            "earthquake": int(np.sum(y == 0)),
            "explosion": int(np.sum(y == 1)),
        },
        "threshold_from_fujian_validation": float(threshold),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_precision": float(precision_score(y, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, prediction, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc_explosion": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "per_class": {name: report[name] for name in ("earthquake", "explosion")},
    }


def bootstrap_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    replicates: int,
    seed: int,
) -> dict:
    """Event-level stratified bootstrap confidence intervals."""
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
    values = {key: [] for key in ("balanced_accuracy", "macro_f1", "roc_auc")}
    for _ in range(replicates):
        indices = np.concatenate(
            [rng.choice(pool, size=len(pool), replace=True) for pool in class_indices]
        )
        result = metric_set(y[indices], probability[indices], threshold)
        for key in values:
            values[key].append(result[key])
    return {
        key: {
            "lower_95": float(np.quantile(value, 0.025)),
            "upper_95": float(np.quantile(value, 0.975)),
        }
        for key, value in values.items()
    }
