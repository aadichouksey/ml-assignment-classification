"""Interactive, read-only evaluation of five saved classification pipelines.

No model fitting or parameter tuning is performed in this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, roc_curve

from ml_core import CLASS_NAMES, METRIC_COLUMNS, evaluate_fitted_model, validate_test_data

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Classifier Observatory | AIML",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A restrained visual identity, rather than Streamlit's default demo template.
st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    h1, h2, h3 {letter-spacing: -0.025em;}
    div[data-testid="stMetric"] {
        background: #f1f7f8;
        padding: 0.85rem 1rem;
        border: 1px solid #deebed;
        border-radius: 0.7rem;
    }
    div[data-testid="stMetricLabel"] {color: #385569;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_saved_experiment():
    """Load already trained models once; never train in Streamlit."""
    with (ROOT / "model" / "metadata.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    models = {
        label: joblib.load(ROOT / relative_path)
        for label, relative_path in config["model_paths"].items()
    }
    return config, models


@st.cache_data(show_spinner=False)
def get_packaged_csv() -> pd.DataFrame:
    return pd.read_csv(ROOT / "test_data.csv")


def confusion_figure(matrix: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.1, 4.2))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    ax.set_xticks([0, 1], labels=["Benign", "Malignant"])
    ax.set_yticks([0, 1], labels=["Benign", "Malignant"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Confusion matrix (test data)", pad=15)
    pivot = (float(matrix.max()) + float(matrix.min())) / 2
    for row in range(2):
        for column in range(2):
            count = int(matrix[row, column])
            ax.text(
                column, row, str(count), ha="center", va="center", fontsize=15,
                color="white" if count > pivot else "#152b43", fontweight="bold"
            )
    fig.tight_layout()
    return fig


def roc_figure(actual: pd.Series, scores: np.ndarray, auc: float) -> plt.Figure:
    false_positive_rate, true_positive_rate, _ = roc_curve(actual, scores)
    fig, ax = plt.subplots(figsize=(5.3, 4.2))
    ax.plot(false_positive_rate, true_positive_rate, label=f"AUC = {auc:.4f}", linewidth=2.3)
    ax.plot([0, 1], [0, 1], "--", color="#7e8b99", label="Random baseline")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate (malignant)")
    ax.set_title("Receiver operating characteristic", pad=15)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def formatted_metric(value: float) -> str:
    return "N/A" if np.isnan(value) else f"{value:.4f}"


def main() -> None:
    st.title("Classification Model Observatory")
    st.caption(
        "BITS Pilani M.Tech (AIML) | Five classifiers | One fixed holdout | "
        "Saved-model evaluation only"
    )

    try:
        metadata, fitted_models = get_saved_experiment()
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Saved experiment could not be loaded: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Experiment controls")
        chosen_model = st.selectbox("Choose a fitted classifier", list(fitted_models))
        st.divider()
        st.subheader("Test dataset")
        uploaded_csv = st.file_uploader(
            "Upload labeled test CSV", type=["csv"],
            help="Include all 30 named features plus diagnosis_malignant (0=benign, 1=malignant).",
        )
        st.download_button(
            "Download example test CSV",
            data=(ROOT / "test_data.csv").read_bytes(),
            file_name="test_data.csv", mime="text/csv", use_container_width=True,
        )
        st.caption("Uploaded samples are scored only. Saved models are not retrained.")
        st.divider()
        st.markdown("**Dataset:** Wisconsin Diagnostic (UCI)")
        st.markdown(f"**Rows:** {metadata['records_total']}")
        st.markdown(f"**Input features:** {metadata['feature_count']}")
        st.markdown(f"**Training rows:** {metadata['training_rows']}")
        st.markdown(f"**Original test rows:** {metadata['test_rows']}")

    if uploaded_csv is None:
        source = "Packaged, held-out test_data.csv"
        raw = get_packaged_csv()
    else:
        source = f"Uploaded CSV: {uploaded_csv.name}"
        try:
            raw = pd.read_csv(uploaded_csv)
        except Exception as exc:
            st.error(f"Could not read the CSV file: {exc}")
            st.stop()

    try:
        features, target, extra_columns = validate_test_data(
            raw, metadata["feature_columns"], metadata["target_column"]
        )
    except ValueError as exc:
        st.error(f"The test CSV cannot be evaluated: {exc}")
        st.info("Download the example test CSV to check the expected names and label encoding.")
        st.stop()

    st.info(f"**Current data:** {source} | {len(target)} labeled records | "
            f"{int((target == 1).sum())} malignant / {int((target == 0).sum())} benign")
    if extra_columns:
        st.caption("Additional CSV columns were ignored: " + ", ".join(extra_columns))
    if target.nunique() < 2:
        st.warning(
            "Only one target class was supplied. AUC/ROC is undefined, but the other "
            "five metrics and the confusion matrix can still be computed."
        )
    if uploaded_csv is not None:
        st.caption("For an unbiased benchmark, evaluate held-out records that were not used to train the models.")

    with st.expander("Inspect uploaded / packaged test rows", expanded=False):
        st.dataframe(raw.head(12), use_container_width=True, hide_index=True)
        st.caption("Target convention: 0 = Benign; 1 = Malignant (positive class).")

    results = {
        model_name: evaluate_fitted_model(model, features, target)
        for model_name, model in fitted_models.items()
    }
    selected = results[chosen_model]
    st.subheader(f"Selected model: {chosen_model}")

    first = st.columns(3, gap="medium")
    second = st.columns(3, gap="medium")
    for container, metric_name in zip([*first, *second], METRIC_COLUMNS):
        container.metric(metric_name, formatted_metric(selected.metrics[metric_name]))

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.pyplot(confusion_figure(selected.confusion), clear_figure=True)
        tn, fp, fn, tp = selected.confusion.ravel()
        st.caption(
            f"Benign correct: {tn} | Benign flagged malignant: {fp} | "
            f"Malignant missed: {fn} | Malignant detected: {tp}"
        )
    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            target, selected.predictions, labels=[0, 1],
            target_names=["Benign (0)", "Malignant (1)"],
            output_dict=True, zero_division=0,
        )
        report_frame = pd.DataFrame(report).T
        st.dataframe(report_frame.round(4), use_container_width=True)
        st.caption("Precision, recall and F1 in the headline cards use malignant (1) as positive.")

    if target.nunique() == 2:
        with st.expander("ROC curve for the selected model", expanded=False):
            st.pyplot(
                roc_figure(target, selected.probabilities, selected.metrics["AUC"]),
                clear_figure=True,
            )

    st.divider()
    st.subheader("All five models on the same test data")
    comparison = pd.DataFrame(
        [{"ML Model Name": name, **result.metrics} for name, result in results.items()]
    )
    comparison_display = comparison.copy()
    for metric in METRIC_COLUMNS:
        comparison_display[metric] = comparison_display[metric].map(formatted_metric)
    st.dataframe(comparison_display, use_container_width=True, hide_index=True)
    st.download_button(
        "Download the six-metric comparison",
        data=comparison.to_csv(index=False).encode("utf-8"),
        mime="text/csv", file_name="evaluated_models.csv",
    )

    if target.nunique() == 2:
        ordered = comparison.sort_values(["MCC", "AUC", "F1"], ascending=False)
        best_name = ordered.iloc[0]["ML Model Name"]
        st.success(f"Top model on the current dataset, ranked by MCC: **{best_name}**")
    st.caption(
        "Educational machine learning demonstration. It must not be used for clinical diagnosis "
        "or medical decisions. Metrics for arbitrary uploads are not necessarily unbiased."
    )


if __name__ == "__main__":
    main()
