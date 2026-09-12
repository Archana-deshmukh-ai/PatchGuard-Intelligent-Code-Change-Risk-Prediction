# PatchGuard Development Log

## Project Vision
**PatchGuard: Intelligent Code Change Risk Prediction**

The long-term goal of PatchGuard is to build a real-world software engineering bug-risk prediction system. The final architecture will analyze code changes (Git commits / Pull Requests), extract software engineering metrics, predict bug probability using Machine Learning, provide LLM-driven risk explanations, leverage RAG over repository documentation/issues/PR history, and integrate into developer CI/CD workflows to give actionable recommendations.

## Current Stage
**Step 1 Milestone Complete & Rebranded to PatchGuard**

## Completed Work
* Rebranded project to **PatchGuard: Intelligent Code Change Risk Prediction**.
* Established Git repository hygiene (`.gitignore` created to exclude bytecode, virtualenvs, IDE files, and secrets).
* Updated public [`README.md`](file:///c:/Projects/CODE_project/bug-prediction/README.md) into a living document for GitHub, clearly demarcating synthetic experiment results from real-world claims.
* Project directory structure finalized (`bug-prediction/` with `data/raw/`, `data/processed/`, `src/`, `models/`, `reports/`, `notebooks/`).
* Python environment and packages verified (`pandas`, `numpy`, `scikit-learn`, `joblib`).
* Created [`requirements.txt`](file:///c:/Projects/CODE_project/bug-prediction/requirements.txt) with baseline dependencies.
* Synthetic dataset generator ([`src/generate_dataset.py`](file:///c:/Projects/CODE_project/bug-prediction/src/generate_dataset.py)) implemented using probabilistic log-odds + Gaussian noise.
* Synthetic dataset generated ([`data/raw/synthetic_commits.csv`](file:///c:/Projects/CODE_project/bug-prediction/data/raw/synthetic_commits.csv)) containing 300 rows and 6 columns.
* Full dataset verification completed (shape, null check, class distribution, code_churn identity check).
* Implemented [`src/data_loader.py`](file:///c:/Projects/CODE_project/bug-prediction/src/data_loader.py): schema validation, null value verification, and feature/target separation ($X$ and $y$).
* Implemented [`src/preprocessing.py`](file:///c:/Projects/CODE_project/bug-prediction/src/preprocessing.py): stratified 80/20 train-test splitting and `StandardScaler` feature scaling (fitted on train set only to eliminate data leakage).
* Implemented [`src/train.py`](file:///c:/Projects/CODE_project/bug-prediction/src/train.py): trained Logistic Regression baseline on `X_train_scaled` and saved `models/logistic_regression.pkl` and `models/scaler.pkl`.
* Implemented [`src/evaluate.py`](file:///c:/Projects/CODE_project/bug-prediction/src/evaluate.py): evaluated Logistic Regression and a `DummyClassifier` (majority class baseline) on the unseen test set (60 samples), saving full results to [`reports/baseline_results.json`](file:///c:/Projects/CODE_project/bug-prediction/reports/baseline_results.json).
* Implemented [`src/predict.py`](file:///c:/Projects/CODE_project/bug-prediction/src/predict.py): single commit inference script returning prediction class, bug probability, and risk level.

## Current Dataset
* **Location**: `data/raw/synthetic_commits.csv`
* **Rows**: 300
* **Columns**: 6 (5 features, 1 target)
* **Feature Names**: `lines_added`, `lines_deleted`, `files_changed`, `functions_changed`, `code_churn`
* **Target Name**: `bug_introduced` (0 = Clean/Safe, 1 = Bug-prone)
* **Class Distribution**: 203 Clean (67.67%), 97 Bug-prone (32.33%)
* **Train/Test Split**: 240 train samples (80%), 60 test samples (20%), stratified by class.
* **Missing-Value Status**: 0 missing values across all columns.
* **Code-Churn Identity Check**: `code_churn == lines_added + lines_deleted` verified 100% True.

## Baseline Experimental Results

### 1. Model Artifacts
* **Saved Model**: `models/logistic_regression.pkl`
* **Saved Scaler**: `models/scaler.pkl`
* **Saved Report**: `reports/baseline_results.json`

### 2. Metrics Comparison (Test Set = 60 Samples)

| Metric | Dummy Baseline (Majority Class '0') | Logistic Regression Baseline |
| :--- | :---: | :---: |
| **Accuracy** | 0.6833 (68.33%) | **0.8667 (86.67%)** |
| **Precision** | 0.0000 | **0.8667 (86.67%)** |
| **Recall** | 0.0000 | **0.6842 (68.42%)** |
| **F1-Score** | 0.0000 | **0.7647 (76.47%)** |
| **ROC-AUC** | 0.5000 | **0.8806 (88.06%)** |

### 3. Confusion Matrix Breakdown (Test Set = 60 Samples)
* **Logistic Regression**:
  * **True Negatives (TN)**: 39 (Clean commits correctly identified as safe)
  * **False Positives (FP)**: 2 (Safe commits incorrectly flagged as risky)
  * **False Negatives (FN)**: 6 (Bug-inducing commits missed by model)
  * **True Positives (TP)**: 13 (Bug-inducing commits correctly caught)
* **Dummy Classifier**: `TN=41, FP=0, FN=19, TP=0` (Missed all 19 bug-inducing commits).

### 4. Feature Coefficients (Learned Log-Odds Weights)

| Feature Name | Coefficient | Interpretation |
| :--- | :---: | :--- |
| `functions_changed` | **+0.7551** | Strongest positive association with predicted bug log-odds |
| `code_churn` | **+0.2458** | Positive association with risk log-odds |
| `lines_added` | **+0.1770** | Positive association with risk log-odds |
| `lines_deleted` | **+0.1590** | Positive association with risk log-odds |
| `files_changed` | **-0.0104** | Slightly negative association (holding churn/functions constant) |
| *Intercept (bias)* | **-0.8198** | Base log-odds bias reflecting majority clean class |

### 5. Single Commit Prediction Test
* **Test Input**: `lines_added=120`, `lines_deleted=30`, `files_changed=6`, `functions_changed=10`, `code_churn=150`
* **Prediction Label**: **Bug-prone (Class 1)**
* **Bug Probability**: **0.8417**
* **Risk Level**: **HIGH**

## Important Engineering & Git Decisions
* **Branding**: Rebranded project to PatchGuard.
* **Git Hygiene**: Added `.gitignore` to prevent tracking bytecode, virtualenvs, OS junk, and IDE files.
* **Artifact Tracking Strategy**:
  * `data/raw/synthetic_commits.csv` (15 KB) & `models/*.pkl` (<5 KB) are small and included in initial commit to allow instant out-of-the-box reproduction for cloned repositories.
  * `data/processed/` files are tracked for pipeline inspection.
* **Living README**: `README.md` formatted for public GitHub viewing with clear synthetic experiment disclaimers.

## Development History
### Step 1A — Project Understanding
Completed. Established ML problem definition, baseline architecture, metrics, and limitations.

### Step 1B — Synthetic Dataset Foundation
Completed. Created project structure, requirements, README, dataset generator, and verified `data/raw/synthetic_commits.csv`.

### Step 1C — Data Loading & Preprocessing
Completed. Created `src/data_loader.py` and `src/preprocessing.py`, performed data validation, feature scaling, stratified splitting, and saved processed CSVs.

### Step 1D — Baseline Model Training & Evaluation
Completed. Implemented `src/train.py`, `src/evaluate.py`, and `src/predict.py`. Evaluated Logistic Regression vs Dummy Baseline on unseen test set, generated `reports/baseline_results.json`, and tested single commit inference.

### Step 1 Milestone — Rebranding to PatchGuard & Git Hygiene
Completed. Rebranded project to PatchGuard, set up `.gitignore`, updated `README.md` and `PROJECT_LOG.md`, and prepared first Git commit milestone.
