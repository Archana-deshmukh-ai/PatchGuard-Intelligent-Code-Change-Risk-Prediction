# PatchGuard Development Log

## Project Vision
**PatchGuard: Intelligent Code Change Risk Prediction**

The long-term goal of PatchGuard is to build a real-world software engineering bug-risk prediction system. The final architecture will analyze code changes (Git commits / Pull Requests), predict bug probability using Machine Learning, provide LLM-driven risk explanations, leverage RAG over repository documentation/issues/PR history, and integrate into developer CI/CD workflows to give actionable recommendations.

## Current Stage
**Phase 1 Milestone Approved — Isolated Edge-Case Validation Completed**

## Phase 1 Git Repository Extractor (`src/git_extractor.py`)

Implemented a modular, reusable Git history extraction engine (`GitRepositoryExtractor`) designed to parse any local Git repository into commit-level software engineering features.

### 1. Extractor Design & Key Features
* **Native Subprocess Execution**: Uses native `git` CLI commands (`git log`, `git diff-tree --numstat -M`, `git diff-tree -U0 -M`) for zero third-party dependencies and maximum cross-platform speed.
* **Strict Commit-Time Scoping**: All features are computed strictly using diffs relative to parent commits (`commit~1` to `commit`). Zero future-leakage.
* **Chronological Order**: Commits are logged in historical sequence (`git log --reverse`).
* **Flexible Traversal**: Defaults to extracting all non-merge commits. Configurable options for `--first-parent` filtering, custom ref branches, and commit count limits.
* **Rename & Move Handling**: Uses `-M` flag in `git diff-tree` and custom path parsing to prevent pure renames from inflating line churn or corrupting file/directory counts.
* **Function Header Context Heuristic**: `functions_changed` is explicitly calculated as a heuristic count of modified function blocks by inspecting `@@ ... @@` diff hunk headers.

### 2. Extracted Dataset Schema (14 Columns)
* **Metadata**: `commit_hash` (short), `full_hash`, `commit_timestamp` (ISO 8601), `author`.
* **10 Features**: `lines_added`, `lines_deleted`, `code_churn`, `files_changed`, `functions_changed`, `num_directories_touched`, `is_test_file_modified`, `avg_lines_changed_per_file`, `max_lines_changed_in_single_file`, `num_source_files_changed`.

---

## Isolated Edge-Case Experiment Verification Report (`scratch/test_repo`)

An isolated temporary repository was created under `scratch/test_repo` to test controlled edge-case commits without contaminating PatchGuard's repository history.

| Edge Case Test | Extracted Output | Verification Result |
| :--- | :--- | :---: |
| **1. Pure File Rename** | `code_churn = 0`, `lines_added = 0`, `lines_deleted = 0` | **PASSED** (0 fake churn) |
| **2. Binary File Addition** | `files_changed = 1`, `code_churn = 0`, no crash | **PASSED** |
| **3. Empty Commit (`--allow-empty`)** | `code_churn = 0`, `files_changed = 0`, `avg_lines = 0` | **PASSED** |
| **4. Merge Commit Handling** | Skipped 1 merge commit when `include_merges=False` | **PASSED** |
| **5. Test File Detection** | `is_test_file_modified = 1` for `tests/test_core.py` | **PASSED** |
| **6. Multi-Directory Sprawl** | `num_directories_touched = 2` for `pkg/` and `ui/` | **PASSED** |
| **7. Churn Distribution (Avg & Max)**| `avg_lines = 52.5`, `max_lines = 100` (100-line single file) | **PASSED** |
| **8. Chronological Sorting** | `commit_timestamp` strictly sorted = `True` | **PASSED** |

---

## Multi-Model Comparison Baseline Summary (Previous Milestone)

Evaluated 4 baseline models on the exact same 10-feature dataset split (300 samples: 240 train / 60 unseen test):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Pos Preds (>=0.50) | Confusion Matrix [TN, FP, FN, TP] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Majority Reference** | 0.7333 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0 / 60 | `[44, 0, 16, 0]` |
| **Logistic Regression** (Scaled) | **0.7500** | **0.6667** | 0.1250 | 0.2105 | **0.6690** | 3 / 60 | `[43, 1, 14, 2]` |
| **Random Forest** (Unscaled) | 0.6500 | 0.1429 | 0.0625 | 0.0870 | 0.5540 | 7 / 60 | `[38, 6, 15, 1]` |
| **XGBoost** (Unscaled) | 0.7000 | 0.3750 | **0.1875** | **0.2500** | 0.6207 | 8 / 60 | `[39, 5, 13, 3]` |

*Technical note on Random Forest*: Random Forest performed poorly on the small synthetic dataset; additional validation would be required to determine whether overfitting contributed.

---

## Development History
### Step 1A — Project Understanding
Completed. Established ML problem definition, baseline architecture, metrics, and limitations.

### Step 1B — Synthetic Dataset Foundation
Completed. Created project structure, requirements, README, dataset generator, and verified `data/raw/synthetic_commits.csv`.

### Step 1C — Data Loading & Preprocessing
Completed. Created `src/data_loader.py` and `src/preprocessing.py`, performed data validation, feature scaling, stratified splitting, and saved processed CSVs.

### Step 1D — Baseline Model Training & Evaluation
Completed. Implemented `src/train.py`, `src/evaluate.py`, and `src/predict.py`. Evaluated Logistic Regression vs Dummy Baseline on unseen test set.

### Step 1 Milestone — Rebranding to PatchGuard & Git Hygiene
Completed. Rebranded project to PatchGuard, set up `.gitignore`, updated `README.md` and `PROJECT_LOG.md`.

### Milestone 2 — Feature Representation Expansion (10 Features) & Controlled Diagnostic
Completed. Expanded to 10 features, performed controlled diagnostic experiment, corrected technical wording on multicollinearity, and evaluated threshold sensitivity.

### Milestone 3 — Multi-Model Comparison Baseline
Completed. Added `xgboost` dependency to `requirements.txt`, trained and evaluated Logistic Regression, Random Forest, and XGBoost models on unseen test split, and analyzed Gini and Gain feature importances.

### Milestone 4 — Real Git Repository Mining (Phase 1 Extractor) & Isolated Edge-Case Validation
Completed. Implemented `src/git_extractor.py`, created an isolated test repository in `scratch/test_repo`, verified all 8 edge cases (renames, binary files, empty commits, merge filtering, directory counting, test file detection, avg/max line churn), verified chronological ordering, and updated `PROJECT_LOG.md`.
