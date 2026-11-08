"""One-time, reproducible offline model training.

Run on BITS Virtual Lab: python train_models.py
Run only on the training machine, NEVER inside the Streamlit app.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_core import METRIC_COLUMNS, evaluate_fitted_model, validate_test_data

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "model"
TEST_SIZE = 0.20
SEED = 42
TARGET = "diagnosis_malignant"


def build_classifiers() -> dict[str, Pipeline]:
    """All preprocessing stays within each cross-dataset training pipeline."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=SEED)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=5, min_samples_leaf=3, random_state=SEED
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Gaussian Naive Bayes": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("classifier", GaussianNB()),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("classifier", SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)),
            ]
        ),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    feature_columns = [name.replace(" ", "_") for name in dataset.feature_names]
    x = dataset.data.copy()
    x.columns = feature_columns
    # sklearn's raw target: malignant=0, benign=1.
    # We invert it so that POSITIVE (1) consistently means malignant.
    y = pd.Series((dataset.target == 0).astype(int), name=TARGET)

    full = x.assign(**{TARGET: y})

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    test_frame = x_test.assign(**{TARGET: y_test})
    test_frame.to_csv(ROOT / "test_data.csv", index=False)

    models = build_classifiers()
    filenames = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
        "Support Vector Machine": "svm.joblib",
    }
    records: list[dict[str, str | float | int]] = []

    for display_name, estimator in models.items():
        estimator.fit(x_train, y_train)
        model_path = MODELS_DIR / filenames[display_name]
        joblib.dump(estimator, model_path, compress=3)
        restored = joblib.load(model_path)
        expected = estimator.predict(x_test)
        if not np.array_equal(expected, restored.predict(x_test)):
            raise RuntimeError(f"Saved model validation failed for {display_name}")

        measurement = evaluate_fitted_model(restored, x_test, y_test)
        row = {"ML Model Name": display_name, **measurement.metrics}
        records.append(row)

    metrics_frame = pd.DataFrame(records, columns=["ML Model Name", *METRIC_COLUMNS])
    # Rank by MCC first, with AUC and F1 as deterministic tie breakers.
    winner = metrics_frame.sort_values(
        by=["MCC", "AUC", "F1"], ascending=[False, False, False]
    ).iloc[0]["ML Model Name"]

    metadata = {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "dataset_url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "dataset_source": "UCI; accessed through sklearn.datasets.load_breast_cancer",
        "records_total": len(full),
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "target_column": TARGET,
        "label_mapping": {"0": "Benign", "1": "Malignant"},
        "split_method": "Stratified holdout, 80% train / 20% test, random_state=42",
        "random_state": SEED,
        "model_paths": {name: f"model/{file}" for name, file in filenames.items()},
        "test_data_sha256": hashlib.sha256((ROOT / "test_data.csv").read_bytes()).hexdigest(),
        "sklearn_version": sklearn.__version__,
        "selection_criterion": "Highest test MCC; AUC and F1 as tie breakers",
        "winner_on_packaged_test_set": winner,
    }
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    # Check exported CSV and all saved model paths against the app's validator.
    reloaded_csv = pd.read_csv(ROOT / "test_data.csv")
    validated_x, validated_y, extra = validate_test_data(
        reloaded_csv, feature_columns, TARGET
    )
    assert not extra and len(validated_x) == len(validated_y) == len(x_test)
    print("Training dataset:", len(x_train), "rows; test dataset:", len(x_test), "rows")
    print(metrics_frame.round(4).to_string(index=False))
    print("Winner by MCC (AUC/F1 tie breaks):", winner)
    print("Saved five trained model pipelines, metadata.json and test_data.csv.")


if __name__ == "__main__":
    main()
