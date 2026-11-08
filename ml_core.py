"""Shared dataset validation and evaluation functions.

This file deliberately contains no training operations. The Streamlit app uses
only these functions plus the persisted, fitted pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_COLUMNS = ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")
CLASS_NAMES = ("Benign", "Malignant")


@dataclass(frozen=True)
class Evaluation:
    metrics: dict[str, float]
    confusion: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray


def validate_test_data(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Validate a labeled CSV without inspecting or modifying model training state.

    The target MUST be coded 0=benign and 1=malignant. Any extra columns are
    ignored after they are reported to the caller. Missing, invalid or
    non-finite feature values are rejected rather than silently coerced.
    """
    if frame.empty:
        raise ValueError("The test CSV is empty. Upload at least one labeled row.")
    if not frame.columns.is_unique:
        raise ValueError("Duplicate CSV column names are not allowed.")

    required = [*feature_columns, target_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        short = ", ".join(missing[:6])
        suffix = f" (and {len(missing) - 6} more)" if len(missing) > 6 else ""
        raise ValueError(f"Missing required columns: {short}{suffix}")

    extras = [column for column in frame.columns if column not in required]
    try:
        features = frame.loc[:, feature_columns].apply(
            pd.to_numeric, errors="raise"
        ).astype("float64")
    except (ValueError, TypeError) as exc:
        raise ValueError("All 30 input features must contain numeric values.") from exc

    if features.isna().any().any():
        bad = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Missing feature values found in: {', '.join(bad[:5])}")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("The CSV contains infinite or non-finite feature values.")

    try:
        target = pd.to_numeric(frame.loc[:, target_column], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Target values must be numeric 0 (benign) or 1 (malignant).") from exc
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("Target values must be exactly 0 (benign) or 1 (malignant).")
    target = target.astype(int)
    return features, target, extras


def evaluate_fitted_model(model, features: pd.DataFrame, target: pd.Series) -> Evaluation:
    """Evaluate an already-fitted classifier; no fit/fit_transform calls here."""
    predicted = np.asarray(model.predict(features), dtype=int)
    model_classes = list(model.classes_)
    if 1 not in model_classes:
        raise ValueError("The saved model does not contain the malignant (1) class.")
    positive_col = model_classes.index(1)
    probabilities = np.asarray(model.predict_proba(features)[:, positive_col], dtype=float)

    # ROC AUC is mathematically undefined if the uploaded file has one class.
    auc = (
        float(roc_auc_score(target, probabilities))
        if target.nunique() == 2
        else float("nan")
    )
    scores = {
        "Accuracy": float(accuracy_score(target, predicted)),
        "AUC": auc,
        "Precision": float(precision_score(target, predicted, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(target, predicted, pos_label=1, zero_division=0)),
        "F1": float(f1_score(target, predicted, pos_label=1, zero_division=0)),
        "MCC": float(matthews_corrcoef(target, predicted)),
    }
    return Evaluation(
        metrics=scores,
        confusion=confusion_matrix(target, predicted, labels=[0, 1]),
        predictions=predicted,
        probabilities=probabilities,
    )
