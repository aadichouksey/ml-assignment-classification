# Assignment 2: Classification Model Observatory

Machine Learning | M.Tech (Artificial Intelligence and Machine Learning) | BITS Pilani WILP

**Submission status:** Reference implementation. You must perform your own work on the BITS Virtual Lab, verify all results, personalize your analysis and UI, publish your own repository, and include a genuine Virtual Lab screenshot. Do not submit AI-generated content unchanged; the assignment explicitly prohibits direct copy-paste submissions.

## a. Problem statement

Develop and compare five supervised classification algorithms on a common public dataset with at least 500 instances and 12 input features. Report accuracy, ROC AUC, precision, recall, F1 and Matthews correlation coefficient (MCC) on the same held-out test set. Provide a Streamlit interface that accepts labeled CSV test data, selects any saved model, computes its metrics without retraining, and visualizes its confusion matrix and classification report. Publish the project to GitHub and deploy the application on Streamlit Community Cloud.

## b. Dataset description (1 mark)

- **Name:** Breast Cancer Wisconsin (Diagnostic), abbreviated WDBC.
- **Repository:** UCI Machine Learning Repository, [official dataset page](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
- **Local loader:** `sklearn.datasets.load_breast_cancer(as_frame=True)`, used directly by `train_models.py`. The full training dataset does not need to be included in the deployed repository.
- **Task:** Binary classification of digitized-cell-nucleus measurements from fine needle aspirates.
- **Number of instances:** 569; **input features:** 30 numerical variables; **classes:** 2.
- **Feature examples:** mean_radius, mean_texture, mean_perimeter, mean_area, worst_concavity. The complete list of 30 features is in `model/metadata.json`.
- **Target name:** `diagnosis_malignant`, with **0 = benign and 1 = malignant**. The original scikit-learn encoding is reversed (0 = malignant, 1 = benign), so the training script explicitly inverts it to make malignant the positive class.
- **Missing values:** None in the provided scikit-learn copy.
- **Experimental split:** 80% training (455 records) and 20% held-out test (114 records), stratified by diagnosis, using `random_state=42`. Each classifier uses the identical test set. No test records are used in model fitting or hyperparameter search.
- **Preprocessing:** Median imputation resides within every saved pipeline. Standardization resides within the Logistic Regression, KNN and SVM pipelines; the scaler and imputer are fitted using training data only.

## c. GitHub repository link (1 mark)

**Required public repository:** [GITHUB_REPOSITORY_URL](https://github.com/aadichouksey/ml-assignment-classification)

**Live interactive application:** [STREAMLIT_URL]VE(https://ml-assignment-classification.streamlit.app/)

Replace both placeholders with your own working, publicly accessible HTTPS URLs before submission. Upload all files shown in the repository layout below, including `model/`, `test_data.csv`, `README.md` and `requirements.txt`. Verify the links in an incognito browser session.

## d. Models used and evaluation comparison (5 marks)

The five classifiers were fitted once on the same 455 training records, saved using `joblib`, reloaded from disk and evaluated on the identical 114 test records. ROC AUC uses predicted probabilities (`predict_proba`) rather than hard class predictions. Precision, recall and F1 treat **malignant (1)** as the positive class. A higher value is better for all six metrics.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9035 | 0.9471 | 0.9429 | 0.7857 | 0.8571 | 0.7926 |
| KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

**Parameter choices:** Logistic Regression uses `max_iter=2000`; Decision Tree uses `max_depth=5` and `min_samples_leaf=3`; KNN uses `n_neighbors=7`; GaussianNB uses its library defaults; RBF SVM uses `C=1.0`, `gamma='scale'` and `probability=True`. All randomness in applicable estimators is seeded with 42. These are fixed reference settings, not parameters selected by evaluating the held-out test data repeatedly.

**Metric definitions:** Accuracy is the overall fraction correct; AUC measures ranking of malignant above benign examples; precision is TP/(TP+FP); recall is TP/(TP+FN); F1 is the harmonic mean of malignant precision and recall; MCC is the correlation between true and predicted binary labels. All metrics are produced by scikit-learn on the **held-out test set**, not on the training data.

**Observed confusion matrix counts** (positive means malignant):

| ML Model Name | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 71 | 1 | 3 | 39 |
| Decision Tree | 70 | 2 | 9 | 33 |
| KNN | 71 | 1 | 4 | 38 |
| Gaussian Naive Bayes | 72 | 0 | 7 | 35 |
| Support Vector Machine | 72 | 0 | 3 | 39 |

## e. Observations about model performance (3 marks)

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | At 96.49% accuracy and 0.9245 MCC, this standardized linear baseline performed strongly. Its 0.9960 AUC was the highest of all five models. It missed 3 malignant examples and incorrectly flagged 1 benign example as malignant. |
| Decision Tree | The interpretable tree produced the lowest MCC (0.7926) and recall (0.7857). It missed 9 of the 42 malignant cases; its relatively weaker holdout performance suggests that simple threshold-based splits were less successful on this data. |
| KNN | Distance-based KNN achieved 95.61% accuracy and 0.9058 MCC after scaling. It missed 4 malignant observations and produced 1 false positive. Its performance was competitive but below Logistic Regression and SVM. |
| Gaussian Naive Bayes | GaussianNB reached perfect malignant precision (1.0000) with zero false positives, but recall was only 0.8333: it missed 7 malignant cases. The algorithm's conditional-independence assumption may be restrictive for correlated cell measurements. |
| Support Vector Machine | The scaled RBF SVM was strongest by MCC (0.9442), accuracy (97.37%) and F1 (0.9630). It correctly found 39 of 42 malignant examples without labeling any benign examples malignant; its AUC (0.9947) was slightly below Logistic Regression. |
| **Overall winner for this dataset** | **Support Vector Machine** when ranking by test MCC (AUC then F1 for ties). It produced the most balanced overall confusion-matrix performance in this holdout evaluation. Note that Logistic Regression won on AUC, so the preferred model can depend on evaluation priorities. |

**Interpretation caveats:** One stratified holdout split is sufficient for this assignment, but it does not establish clinical validity or confidence intervals. A future study could use repeated stratified cross-validation on training records and external validation, without tuning against the final test set. This interface is an educational demonstration, not a medical diagnostic service.

## Repository layout

```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- train_models.py
|-- ml_core.py
`-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- gaussian_naive_bayes.joblib
    |-- svm.joblib
    `-- metadata.json
```

`train_models.py` is the original model-training source; `ml_core.py` holds the shared input-validation and evaluation functions. The saved `.joblib` files contain the fitted preprocessing pipelines (including the scalers where applicable), so there are no missing separate preprocessing artifacts.

## Reproduce and verify on the BITS Virtual Lab

1. Log in to the **BITS Virtual Lab**. Open a terminal on the lab system and put the unzipped project in your home directory. Record actual execution inside the lab for the mandatory screenshot.
2. Confirm that Python is compatible with the pinned packages (Python 3.11-3.13 recommended; this reference was produced with Python 3.13 and scikit-learn 1.8.0). Use matching versions when generating and loading `joblib` pipelines.
3. In the project folder, run these commands:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1` instead. If BITS Lab does not allow package installation or has an incompatible Python version, use the version and setup provided by the course lab; do not claim to have used the lab without executing the assignment there. For lab issues, contact the two support addresses in the provided assignment brief.

The training program exports the stratified 114-record test CSV, the five saved models and metadata. It prints the same results shown above if run with the same library versions and seed. The application only **loads** the saved `.joblib` pipelines; it does **not** call `fit` on uploaded examples.

## Streamlit interface and test procedure (4 marks)

- **CSV upload:** Use the sidebar upload widget. The CSV must contain exactly named model input features, plus the `diagnosis_malignant` target column; extra columns are ignored and reported. The bundled `test_data.csv` loads by default, and can be downloaded from the app.
- **Model selector:** Choose one of five saved models from the dropdown in the sidebar.
- **Metrics:** Six metric cards for the selected model, plus a side-by-side comparison table for **all five** models on the same data. AUC is based on probabilities. If uploaded labels contain only one class, AUC and ROC are explicitly identified as undefined.
- **Model evidence:** The app displays the confusion matrix, classification report and optional ROC curve for the selected classifier. No model retraining occurs during uploads.

**Manual evaluator smoke test:** Open the deployed URL, confirm that the app renders even before upload, switch through all five models, download the bundled test CSV, re-upload it, and confirm that metrics agree with the six-metric table in this README. Check that the confusion matrix totals 114 for the bundled set. Verify that a missing feature is rejected with a useful message, instead of silently producing incorrect results.

## GitHub publishing and Community Cloud deployment

1. Start your own GitHub repository and add this project's files from the BITS Lab environment, preserving the commit history of your actual work. Do not fabricate activity or provenance.
2. Make the repository accessible to the evaluator, and verify that `app.py`, `requirements.txt`, `README.md`, `test_data.csv`, `model/` and the source scripts appear in the GitHub web view.
3. Update `YOUR_GITHUB_REPOSITORY_URL` and `YOUR_LIVE_STREAMLIT_URL` in this README once links exist.
4. Visit [Streamlit Community Cloud](https://share.streamlit.io/), sign in with GitHub, choose the repo and the main branch, and choose `app.py` as the entrypoint. Choose a Python version compatible with your trained artifacts (this reference uses Python 3.13, if that version is offered).
5. Deploy; inspect build logs if the app fails to load. The project's dependency versions are declared in `requirements.txt`.
6. Open the app's final URL in a fresh/incognito browser, test model selection and upload, and copy the working HTTPS URL.

See the official [Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Mandatory single-PDF submission

Your submitted **single PDF must contain these items in this order**:

1. **Working GitHub repository URL** (clickable).
2. **Working live Streamlit app URL** (clickable).
3. **One genuine screenshot from BITS Virtual Lab** showing execution of the assignment.
4. **Complete README content**, including this dataset description, the comparison table and all five observations plus the winner.

After you have real links and a real screenshot, create the final PDF using a document editor. Include these four sections in the order shown above; for section 4, copy the **complete, updated contents of this README** into the PDF (not merely a link). Export to PDF and verify that the GitHub and Streamlit URLs are clickable. Do not use placeholder URLs or fabricate evidence. Submit the single PDF by **08 November 2026, 23:59**. The assignment says draft submissions are not accepted.

## Source and reproducibility notes

- Original dataset: W. H. Wolberg, W. N. Street and O. L. Mangasarian, *Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository, 1995.
- The repository includes a frozen held-out test CSV and version-pinned model files, together with `train_models.py` for regeneration.
- The test records are for evaluation; only use a separate training dataset for model fitting. Malignant is deliberately defined as the positive class in every metric and chart.
- Saved pickle/joblib artifacts should be loaded only from trusted sources. Never load arbitrary uploaded model files from visitors.
