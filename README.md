# PatchGuard: Intelligent Code Change Risk Prediction

Predicting defect-inducing code changes using machine learning, Git repository mining, SZZ defect labeling, code churn analysis, and contextual intelligence.

---

## 1. Project Overview
**PatchGuard** is a software engineering machine learning system designed to analyze code changes (Git commits and Pull Requests) and estimate the probability that a change will introduce a defect/bug into production.

The long-term goal of PatchGuard is to serve as an intelligent assistant in developer workflows—flagging high-risk code modifications, explaining *why* a change is risky using Large Language Models (LLMs), and providing contextual recommendations grounded in repository documentation and historical issues via Retrieval-Augmented Generation (RAG).

---

## 2. Current Project Status
**Phase 4 Complete — Real-Data ML Pipeline Completed and Approved**

PatchGuard has evolved from synthetic baseline experiments to a real-world software engineering ML pipeline trained on real Git repository history mined via the SZZ defect-labeling algorithm on the mature open-source benchmark repository `bottlepy/bottle` (1,990 commits).

---

## 3. How It Works (System Vision)

```text
[ Git Commit / Pull Request ]
              │
              ▼
[ Real Git Feature Extraction (src/git_extractor.py) ]
              │
              ▼
[ SZZ Defect-Labeling Pipeline (src/szz_labeler.py) ]
              │
              ▼
[ Temporal Train / Validation / Test Splitting ]
              │
              ▼
[ Model-Aware Feature Preprocessing (src/preprocess_real_data.py) ]
              │
              ▼
[ Machine Learning Risk Model (src/train_real_baseline.py) ]
              │
              ▼
[ Validation Model Selection & Threshold Locking (src/select_model.py) ]
              │
              ▼
[ Unbiased Final Test Evaluation (src/evaluate_final_test.py) ] ──► (Phase 4 Completed)
              │
              ▼
[ Risk Prediction Engine & Risk Levels ] ──► (Phase 5 NEXT)
              │
              ▼
[ LLM Risk Explanation & RAG Context ] ──► (Future Stage)
              │
              ▼
[ Developer CI/CD Feedback & Dashboard ]
```

---

## 4. The 10 Commit-Time Model Features

| # | Feature Name | Description | Type |
| :--- | :--- | :--- | :---: |
| 1 | `lines_added` | Total new lines of code added in the commit | Integer |
| 2 | `lines_deleted` | Total lines of code removed in the commit | Integer |
| 3 | `code_churn` | Total volatility (`lines_added + lines_deleted`) | Integer |
| 4 | `files_changed` | Total number of modified files | Integer |
| 5 | `functions_changed` | Estimated function/method signatures modified | Integer |
| 6 | `num_directories_touched` | Distinct directory paths modified | Integer |
| 7 | `is_test_file_modified` | Binary indicator (1 if test file modified, else 0) | Integer |
| 8 | `avg_lines_changed_per_file` | Average line churn per modified file | Float |
| 9 | `max_lines_changed_in_single_file` | Maximum line churn in a single file | Integer |
| 10 | `num_source_files_changed` | Total standard source code files modified | Integer |

> **Leakage Protection Rule**: Commit hashes (`commit_hash`), timestamps (`commit_timestamp`), and retrospective SZZ provenance fields (`fixing_commit_hash`, `fixing_issue_id`, `label_confidence`, `label_source`, `label_semantics`, `traced_lines_count`) are strictly excluded from ML model features.

---

## 5. Target Label Semantics (SZZ Algorithm)

Target Variable: `bug_introduced`
* `1`: **SZZ-identified bug-introducing commit** (Evidence proves modified lines were later deleted/modified in a bug-fixing commit).
* `0`: **No SZZ evidence that the commit introduced a later fixed defect** (No historical evidence found).

> [!NOTE]
> Label `0` indicates *no identified defect evidence* in historical git trace; it does **not** guarantee a commit is logically "bug-free".

---

## 6. Phase 4 Real-Data Benchmark Results (`bottlepy/bottle`)

The pipeline was evaluated on `bottlepy/bottle` using a strict chronological split:
* **TRAIN**: 1,181 oldest commits (70%) — $y=1$ rate: 45.05%
* **VALIDATION**: 253 middle commits (15%) — $y=1$ rate: 29.64%
* **TEST**: 253 newest commits (15%) — $y=1$ rate: 20.16% (Unseen chronological test period)

### Final Locked Evaluation Results (Test Set $N=253$, Locked Model: `LogisticRegression` @ Threshold `0.35`)

| Metric | Dummy Baseline (Majority Class) | Locked PatchGuard Model (`LogisticRegression` @ 0.35) |
| :--- | :---: | :---: |
| **Accuracy** | **0.7984** | **0.6522** |
| **Precision** | 0.0000 | **0.3168** |
| **Recall** | 0.0000 | **0.6275** (Catches 32 / 51 test defect commits) |
| **F1-Score** | 0.0000 | **0.4211** |
| **ROC-AUC** | 0.5000 | **0.6709** |
| **Predicted Positives** | 0 / 253 | **101 / 253** |

### Confusion Matrix (Test Set)
```text
               Predicted Negative (0)    Predicted Positive (1)
Actual (0):            133                       69            (TN=133, FP=69)
Actual (1):             19                       32            (FN=19,  TP=32)
```

> [!IMPORTANT]
> **Interpretation**: The majority baseline achieves higher accuracy (79.84%) simply by predicting 0 for all commits due to class imbalance in recent history. However, the majority baseline has **zero recall** and cannot detect any bugs. PatchGuard's locked candidate model provides non-trivial predictive signal (Recall **62.75%**, F1 **42.11%**, ROC-AUC **0.6709**), successfully detecting 32 out of 51 defect-introducing commits in unseen history.

---

## 7. Project Structure

```text
bug-prediction/
├── data/
│   ├── raw/
│   │   ├── bottle_git_commits.csv       # Extracted commit features for bottlepy/bottle
│   │   ├── patchguard_git_commits.csv   # Mined commits from PatchGuard repo
│   │   ├── synthetic_commits.csv        # Phase 1 synthetic dataset
│   │   └── szz_bottle_labeled_commits.csv # SZZ labels & audit provenance
│   └── processed/
│       ├── bottle_real_dataset.csv      # Complete joined real ML dataset (N=1,687)
│       ├── bottle_train.csv             # Chronological train split (N=1,181)
│       ├── bottle_validation.csv        # Chronological validation split (N=253)
│       └── bottle_test.csv              # Chronological unseen test split (N=253)
├── models/
│   ├── dummy_classifier.pkl             # Majority class baseline
│   ├── logistic_regression_real.pkl     # Locked real-data Logistic Regression model
│   ├── random_forest_real.pkl           # Trained real-data Random Forest model
│   ├── real_data_scaler.pkl             # StandardScaler fitted ON TRAIN ONLY
│   └── xgboost_real.pkl                 # Trained real-data XGBoost model
├── reports/
│   ├── final_test_predictions.csv       # Test commit predictions & probabilities
│   ├── final_test_results.json          # Machine-readable final test evaluation report
│   ├── real_baseline_validation_results.json # Baseline validation metrics
│   ├── threshold_analysis_validation.csv # Validation threshold sweep data
│   └── threshold_analysis_validation.json # Validation threshold sweep report
├── src/
│   ├── build_dataset.py                 # Real dataset builder (features + SZZ labels)
│   ├── data_loader.py                   # Data loader and schema validator
│   ├── evaluate_final_test.py           # Phase 4.6 final locked test evaluator
│   ├── git_extractor.py                 # Phase 2 real Git repository extractor
│   ├── predict.py                       # Single commit risk inference CLI
│   ├── preprocess_real_data.py          # Phase 4.3 model-aware feature preprocessor
│   ├── preprocessing.py                 # Baseline preprocessor
│   ├── select_model.py                  # Phase 4.5 validation threshold selector
│   ├── szz_labeler.py                   # Phase 3 SZZ defect-labeling pipeline
│   ├── temporal_split.py                # Phase 4.2 chronological dataset splitter
│   └── train_real_baseline.py           # Phase 4.4 baseline model trainer
├── .gitignore                           # Git exclusion rules
├── PROJECT_LOG.md                       # Comprehensive engineering decision log
├── README.md                            # Public repository documentation
└── requirements.txt                     # Project dependencies
```

---

## 8. Reproducing the Real-Data Pipeline Step-by-Step

```bash
# 1. Mine commit-time features from local benchmark repository
python scratch/test_bottle_extractor.py

# 2. Build joined real ML dataset (features + SZZ labels)
python src/build_dataset.py

# 3. Create chronological train/validation/test split (70/15/15)
python src/temporal_split.py

# 4. Preprocess features and fit StandardScaler on TRAIN ONLY
python src/preprocess_real_data.py

# 5. Train baseline models on TRAIN split
python src/train_real_baseline.py

# 6. Run validation threshold analysis and model selection
python src/select_model.py

# 7. Execute ONE final unbiased evaluation on unseen TEST split
python src/evaluate_final_test.py
```

---

## 9. Current Limitations & Key Learnings

1. **Shallow Feature Representation**: Commit-level diff metrics capture commit size and dispersion, but do not capture deep AST structure or semantic code logic.
2. **Detection of Small Patches**: Small, single-file bug fixes produce small diff footprints, making them harder for churn metrics alone to detect (leading to False Negatives).
3. **Refactoring Noise**: Large structural refactorings exhibit high churn and multi-file touches, increasing predicted risk scores even when no defects are introduced (leading to False Positives).
4. **Temporal Class Prevalence Shift**: Defect-introducing commit frequency decreases as codebases mature (from 45% in early history to 20% in recent commits), affecting fixed-threshold precision across multi-year timeframes.
5. **Uncalibrated Probability Scores**: Model probabilities represent discrimination scores and have not yet been formally calibrated into real-world probabilities.

---

## 10. Project Roadmap

- [x] **Phase 1: Project & ML Foundation** — Supervised binary classification framing, synthetic pipeline verification, baseline evaluation metrics.
- [x] **Phase 2: Real Git Repository Mining** — Subprocess-based `GitRepositoryExtractor` parsing 10 commit-time diff features across Git history.
- [x] **Phase 3: Defect Labeling / SZZ** — Retrospective line-blame tracing (`SZZDefectLabeler`), multi-signal confidence scoring, tested on `bottlepy/bottle`.
- [x] **Phase 4: Real-Data ML Pipeline** — Dataset construction, 70/15/15 chronological split, Train-only scaling, multi-model baseline training, validation threshold selection, final locked test evaluation.
- [ ] **Phase 5: Risk Prediction Engine** *(NEXT)* — Turning the ML risk model into a reusable prediction module (Probability $\rightarrow$ Risk Level $\rightarrow$ Signal Breakdown $\rightarrow$ Actionable Recommendation).
- [ ] **Phase 6: LLM Code-Change Analysis** — Generating human-readable risk explanations from code diffs using LLMs.
- [ ] **Phase 7: RAG / Repository Intelligence** — Retrieving context from repository docs, historical issues, and PR history.
- [ ] **Phase 8: Developer Dashboard** — Interactive UI for commit risk monitoring and audit reports.
- [ ] **Phase 9: GitHub / CI-CD Integration** — Automated PR risk bot and workflow status checks.
- [ ] **Phase 10: Productionization** — End-to-end API, containerization, and production deployment.
