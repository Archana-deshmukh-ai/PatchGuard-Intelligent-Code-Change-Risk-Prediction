# PatchGuard: Intelligent Code Change Risk Prediction

Predicting defect-inducing code changes using machine learning, code churn analysis, and contextual intelligence.

---

## 1. Project Overview
**PatchGuard** is a software engineering machine learning system designed to analyze code changes (Git commits and Pull Requests) and estimate the probability that a change will introduce a defect/bug into production.

The long-term goal of PatchGuard is to serve as an intelligent assistant in developer workflows—flagging high-risk code modifications, explaining *why* a change is risky using Large Language Models (LLMs), and providing contextual recommendations grounded in repository documentation and historical issues via Retrieval-Augmented Generation (RAG).

---

## 2. Problem Statement
Software defects introduced during active development are significantly more expensive to fix after being deployed to production. Manual code reviews are critical, but human reviewers can miss subtle bug-inducing patterns in large or complex commits.

PatchGuard addresses this challenge by providing automated, data-driven bug risk scores at the moment code changes are committed, enabling developers to prioritize code reviews and focus testing on high-risk modifications.

---

## 3. Why PatchGuard
* **Early Defect Detection**: Catch potential bugs before code is merged into release branches.
* **Objective Risk Scoring**: Quantify commit risk based on code churn, file dispersion, and structural modifications.
* **Developer-Centric Design**: Designed to integrate into CI/CD pipelines without interrupting developer velocity.
* **Progressive Architecture**: Built modularly to evolve from simple linear baselines to advanced tree-based models and LLM-driven contextual explanations.

---

## 4. How It Works (System Vision)

```text
[ Git Commit / Pull Request ]
              │
              ▼
[ Repository Mining & Feature Extraction ]
              │
              ▼
[ Machine Learning Risk Model ]
              │
              ▼
[ Bug Probability & Classification ] ──► (PatchGuard Baseline Stage)
              │
              ▼
[ LLM Risk Explanation & RAG Context ] ──► (Future Stage)
              │
              ▼
[ Developer CI/CD Feedback & Report ]
```

---

## 5. Current Stage: Baseline ML Pipeline
We are currently at **Step 1 (ML Baseline)**. In this stage, PatchGuard establishes a clean, reproducible binary classification pipeline using **Logistic Regression** to predict whether a commit is bug-prone (`1`) or clean (`0`).

### Initial 5 Features
| Feature | Description | Type |
| :--- | :--- | :---: |
| `lines_added` | Total new lines of code added | Integer |
| `lines_deleted` | Total lines of code removed | Integer |
| `files_changed` | Total number of modified files | Integer |
| `functions_changed` | Total number of functions/methods modified | Integer |
| `code_churn` | Total code volatility (`lines_added + lines_deleted`) | Integer |

---

## 6. Dataset & Synthetic Baseline Notice
> [!IMPORTANT]
> The current dataset stored in `data/raw/synthetic_commits.csv` (300 samples) is **synthetic** and generated strictly to verify the ML pipeline engineering. It reflects plausible relationships (e.g., larger changes tend to have higher risk with realistic noise), but does **not** represent real-world repository behavior. The pipeline is designed so that this dataset can later be replaced with mined Git history (via SZZ algorithm) without changing downstream pipeline code.

---

## 7. Baseline Experiment Results

> [!NOTE]
> The metrics below reflect model performance on the **synthetic baseline test set (60 samples)**. They demonstrate pipeline functionality, not real-world defect prediction accuracy.

### Metrics Comparison (Test Set = 60 Samples)

| Metric | Dummy Baseline (Majority Class '0') | PatchGuard Baseline (Logistic Regression) |
| :--- | :---: | :---: |
| **Accuracy** | 0.6833 | **0.8667** |
| **Precision** | 0.0000 | **0.8667** |
| **Recall** | 0.0000 | **0.6842** |
| **F1-Score** | 0.0000 | **0.7647** |
| **ROC-AUC** | 0.5000 | **0.8806** |

### Confusion Matrix (Logistic Regression)
* **True Negatives (TN)**: 39 (Clean commits correctly identified)
* **False Positives (FP)**: 2 (Safe commits flagged as risky)
* **False Negatives (FN)**: 6 (Bug-inducing commits missed by model)
* **True Positives (TP)**: 13 (Bug-inducing commits correctly caught)

---

## 8. Example Prediction Output

Running inference on a new hypothetical commit (`src/predict.py`):

```bash
python src/predict.py
```

**Output**:
```text
==================================================
       SINGLE COMMIT RISK INFERENCE TEST          
==================================================
Input Code Change  : {'lines_added': 120, 'lines_deleted': 30, 'files_changed': 6, 'functions_changed': 10, 'code_churn': 150}
Prediction Label   : Bug-prone (Class 1)
Bug Probability    : 0.8417
Risk Level         : HIGH
--------------------------------------------------
Note: Risk level boundaries (0-0.33 LOW, 0.33-0.66 MEDIUM, 0.66-1.0 HIGH)
are engineering choices for pipeline testing, not calibrated thresholds.
==================================================
```

---

## 9. Installation & Reproduction Guide

### Prerequisites
* Python 3.10+

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/PatchGuard.git
cd PatchGuard/bug-prediction

# Install dependencies
pip install -r requirements.txt
```

### Reproducing the Pipeline Step-by-Step
```bash
# 1. Generate Synthetic Dataset (300 commit samples)
python src/generate_dataset.py

# 2. Test Data Loader Validation & Schema Checks
python src/data_loader.py

# 3. Perform Preprocessing (Stratified Train/Test Split & StandardScaler)
python src/preprocessing.py

# 4. Train Baseline Logistic Regression Model
python src/train.py

# 5. Evaluate Baseline against Dummy Majority Classifier
python src/evaluate.py

# 6. Test Single Commit Risk Inference
python src/predict.py
```

---

## 10. Project Structure

```text
bug-prediction/
├── data/
│   ├── raw/
│   │   └── synthetic_commits.csv       # Raw input dataset
│   └── processed/
│       ├── test_processed.csv          # Scaled test split
│       └── train_processed.csv         # Scaled train split
├── models/
│   ├── logistic_regression.pkl         # Trained model artifact
│   └── scaler.pkl                      # Fitted StandardScaler artifact
├── notebooks/                          # EDA & experimental notebooks
├── reports/
│   └── baseline_results.json           # Evaluation metrics JSON report
├── src/
│   ├── data_loader.py                  # Schema validation & data loader
│   ├── evaluate.py                     # Evaluation & baseline comparison
│   ├── generate_dataset.py             # Synthetic dataset generator
│   ├── predict.py                      # Single commit risk prediction CLI
│   ├── preprocessing.py                # Train/test split & feature scaling
│   └── train.py                        # Logistic Regression model training
├── .gitignore                          # Git exclusion rules
├── PROJECT_LOG.md                      # Internal engineering log & memory
├── README.md                           # Public repository documentation
└── requirements.txt                    # Project dependencies
```

---

## 11. Current Limitations
1. **Synthetic Data**: The baseline relies on synthetic data designed to test pipeline flow. Real software engineering distributions have significantly higher noise and non-linearity.
2. **Coarse Metadata Features**: The model currently evaluates lines touched and file counts, ignoring actual code syntax, AST nodes, and semantic changes.
3. **Linear Decision Boundary**: Logistic Regression assumes linear log-odds relationships.

---

## 12. Project Roadmap
- [x] **Step 1: Baseline ML System** (Logistic Regression, synthetic dataset, pipeline engineering, evaluation metrics).
- [ ] **Step 2: Enhanced Feature Engineering** (Process coupling, author experience, commit frequency, time metrics).
- [ ] **Step 3: Real Repository Data Mining** (Git commit extraction, issue tracker linking, SZZ defect-labeling algorithm).
- [ ] **Step 4: Tree-Based ML Models** (Random Forest, XGBoost baseline comparison).
- [ ] **Step 5: LLM Risk Explanations** (Generating human-readable explanations for flagged changes).
- [ ] **Step 6: RAG Integration** (Retrieving context from repository docs, issues, and PR history).
- [ ] **Step 7: Developer-Facing System** (CLI tool / CI/CD pipeline integration).
