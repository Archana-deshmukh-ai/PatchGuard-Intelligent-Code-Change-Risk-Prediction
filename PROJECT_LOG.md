# PatchGuard Development Log

## Project Vision
**PatchGuard: Intelligent Code Change Risk Prediction**

The long-term goal of PatchGuard is to build a real-world software engineering bug-risk prediction system. The final architecture will analyze code changes (Git commits / Pull Requests), predict bug probability using Machine Learning, provide LLM-driven risk explanations, leverage RAG over repository documentation/issues/PR history, and integrate into developer CI/CD workflows to give actionable recommendations.

## Current Stage
**Phase 6 Complete — LLM Code-Change Analysis & CLI Completed, Tested, and Checkpointed**

### Handoff Guide for Future AI / Developer Sessions
> [!IMPORTANT]
> **Phase 6 is COMPLETE and CHECKPOINTED**.
> **DO NOT START PHASE 7 IMPLEMENTATION YET.**
>
> When resuming development in a future session:
> 1. First inspect [`README.md`](file:///c:/Projects/CODE_project/bug-prediction/README.md), [`PROJECT_LOG.md`](file:///c:/Projects/CODE_project/bug-prediction/PROJECT_LOG.md), `git status`, and `git log`.
> 2. Understand that Phase 6 (Offline Git Diff Extraction, Analysis Schemas, Prompt Builder, Response Parser, Evidence Validator, OpenAI Provider Integration, CLI Orchestration, and Verification) is complete, tested with 81 passing unit tests, and fully checkpointed.
> 3. The next phase is **Phase 7 — RAG / Repository Intelligence**.
>
> **Long-Term Roadmap**:
> 1. Project & ML Foundation — **COMPLETE**
> 2. Real Git Repository Mining — **COMPLETE**
> 3. Defect Labeling / SZZ — **COMPLETE**
> 4. Real-Data ML Pipeline — **COMPLETE**
> 5. Risk Prediction Engine — **COMPLETE**
> 6. LLM Code-Change Analysis & CLI — **COMPLETE**
> 7. **RAG / Repository Intelligence — NEXT**
> 8. Developer Dashboard
> 9. GitHub / CI-CD Integration
> 10. Productionization

---

## Phase 6.7 Final Verification, Documentation & Milestone Checkpoint

Completed Phase 6.7 (Final Testing, Documentation, and Checkpointing), performing end-to-end verification, schema stability auditing, CLI smoke testing, offline test execution, documentation alignment, repository hygiene, and milestone commit creation for Phase 6.

### 1. Final Phase 6 Milestone Summary
* **Phase 6.2 (Git Diff Extraction)**: `GitDiffExtractor` parsing commits into structured `CommitDiff`, `FileDiff`, `DiffHunk`, and `DiffLine` domain models.
* **Phase 6.3 (Analysis Schemas & Provider Abstractions)**: `LLMAnalysisResult` schema, `LLMProvider` base contract, and deterministic `MockLLMProvider`.
* **Phase 6.4 (Prompt Builder, Analyzer, Response Parser, Evidence Validator)**: Budget-managed prompt construction (`PromptBuilder`), `LLMCodeAnalyzer` pipeline orchestrator, defensive `ResponseParser` handling markdown code-fence sanitization, and `EvidenceValidator` checking cited line ranges against structured diff lines.
* **Phase 6.5 (OpenAI LLM Provider)**: Live provider integration using official OpenAI Python SDK (`OpenAIProvider`, `max_retries` configuration, environment API key handling).
* **Phase 6.6 (CLI Integration)**: Unified CLI (`src/cli.py`) orchestrating quantitative `predict` and qualitative `analyze` subcommands with text card and JSON rendering.
* **Phase 6.7 (Final Testing & Checkpoint)**: Comprehensive test suite verification, schema validation, documentation alignment, hygiene verification, and checkpoint commit creation.

### 2. Test Verification & Suite Count
* **Full Automated Test Suite**: **81 passing unit tests** across 8 test modules (`tests/test_cli.py`, `tests/test_openai_provider.py`, `tests/test_analysis_pipeline.py`, `tests/test_analysis_schema.py`, `tests/test_diff_extractor.py`, `tests/test_prediction_engine.py`, `tests/test_git_extractor.py`, `tests/test_szz_labeler.py`).
* **Test Runtime & Safety**: Completed in ~61s 100% offline without network calls or API keys required.
* **Formatting & Hygiene**: `git diff --check` passed cleanly with 0 errors.
* **Milestone Checkpoint Commit**: `3e7a5c9` — *Complete Phase 6 LLM code-change analysis*

---

## Phase 6.6 CLI Integration Implementation & Verification

Completed and validated Phase 6.6 (CLI Integration), extending the unified command line interface (`src/cli.py`) with the `analyze` subcommand to orchestrate Phase 5 quantitative ML risk prediction and Phase 6 qualitative LLM analysis.

### 1. Key CLI Capabilities & Architectural Decoupling
* **Subcommand Orchestration**: Added `patchguard analyze` while retaining `patchguard predict`. Reuses existing `RiskPredictionEngine` and `LLMCodeAnalyzer` pipelines without duplicating extraction, prediction, parsing, or validation logic.
* **Provider Abstraction Selection**: Supports `--provider mock` (default, 100% offline using `MockLLMProvider`) and `--provider openai` (using `OpenAIProvider`). `--model` parameter allows specifying models (defaults to `gpt-4o-mini` for `openai`).
* **Format Renderers**: Provides `--format text` human-readable terminal output card with structured sections (Quantitative Defect Risk, Qualitative LLM Summary, Key Changes, Potential Risk Factors, Affected Areas, Evidence Citations) and `--format json` for machine consumption.
* **Error Handling & Exit Codes**: All domain exceptions (including `MissingAPIKeyError`, `OpenAIProviderError`, `RepositoryNotFoundError`, `CommitNotFoundError`, `UnsupportedCommitTypeError`) produce clean stderr formatting (`[PatchGuard Error] ExceptionName: message`) and return exit code 1.
* **API Key Security**: `OPENAI_API_KEY` is checked strictly from environment variables when using `--provider openai`. Never hardcoded, printed, or logged.

### 2. Verification & Test Suite (`tests/test_cli.py`)
* Comprehensive test suite containing 9 unit tests covering CLI argument parsing, default subcommands, `predict` output formatting, `analyze` mock execution (text & JSON), error handling for invalid repos/commits, missing API key error formatting, and mocked `OpenAIProvider` execution.
* Total test suite count expanded to **81 passing tests** (64 base + 8 OpenAI + 9 CLI tests).
* Verified 100% offline execution of tests.

---

## Phase 5 Risk Prediction Engine Implementation & Verification

Completed and validated Phase 5 (Risk Prediction Engine), encapsulating the locked Phase 4 ML model (`models/logistic_regression_real.pkl` @ threshold `0.35`) and preprocessor scaler (`models/real_data_scaler.pkl`) into an inference-only engine (`RiskPredictionEngine`) and CLI tool (`src/cli.py`).

### 1. Architectural Summary & Major Components Added
* **Core Inference Engine (`src/engine/predictor.py`)**: `RiskPredictionEngine` provides a clean `predict(repo_path, commit_hash)` API. Inference-only: zero retraining, zero threshold tuning, zero dataset modifications.
* **Canonical Schema & Ordering (`src/engine/schema.py`)**: Strictly enforces the canonical 10-feature schema names and exact column ordering (`FEATURE_COLUMNS`).
* **Custom Exceptions (`src/engine/exceptions.py`)**: Structured exception hierarchy under `PatchGuardError` handling invalid repos, missing commits, merge commits (`UnsupportedCommitTypeError`), and zero-diff commits (`EmptyCommitError`).
* **Targeted Commit Extractor (`src/git_extractor.py`)**: Added `extract_single_commit(commit_hash)` to extract commit-time features for single commits without full history scans.
* **Thin CLI Tool (`src/cli.py`)**: `patchguard predict` command supporting human-readable terminal text cards (`--format text`) and structured machine JSON (`--format json`).
* **Decoupled Architecture**: Intentionally decoupled from Phase 6 (zero LLM/RAG imports or prompt coupling in Phase 5).

### 2. Provenance & Feature Contribution Signals
* **Artifact Hashes**: `PredictionResult` records SHA-256 checksums of model and scaler pickle artifacts (`model_artifact_hash`, `scaler_artifact_hash`) alongside `model_id`, `model_version`, `feature_schema_version`, and ISO 8601 UTC timestamps.
* **Quantitative Model Signals**: Model signals log-odds contributions ($C_i = \beta_i \cdot z_i$) are documented strictly as mathematical feature contributions to the Logistic Regression score ($z \to p$), **not** causal explanations of bugs.
* **Presentation Risk Levels**: `LOW` ($p < 0.35$), `MEDIUM` ($0.35 \le p < 0.60$), and `HIGH` ($p \ge 0.60$) are UI presentation labels only, with explicit disclaimers attached.

### 3. Verification & Test Suite (`tests/test_prediction_engine.py`)
* Executed unit and integration test suite: **12/12 tests passed**.
* Verified invalid repository, invalid commit SHA, merge commit rejection, threshold behavior ($0.35$), probability bounds ($p \in [0.0, 1.0]$), JSON serialization, SHA-256 artifact hashes, and deterministic repeated prediction.
* CLI text and JSON execution verified on local commits.
* *Known Minor Limitation*: Dedicated unit test method for empty commits (`EmptyCommitError`) is not yet implemented in `tests/test_prediction_engine.py` (logic is present in `predictor.py`).

### 4. Phase 5 Release Recommendation
**`APPROVE PHASE 5 FOR RELEASE AND PROCEED TO PHASE 6`**  
Phase 5 (Risk Prediction Engine) is complete, validated, fully tested, inference-only, and ready for release.

---

## Phase 3 SZZ Defect Labeling Implementation (`src/szz_labeler.py`)

Implemented the SZZ (Śliwerski, Zimmermann, Zeller) defect-labeling pipeline (`SZZDefectLabeler`) to retrospectively trace bug-fixing commits back to their origin bug-introducing commits.

### 1. Label Semantics
* `1`: **SZZ-identified bug-introducing commit** (Evidence proves modified lines were later deleted/modified in a bug-fixing commit).
* `0`: **No SZZ evidence that the commit introduced a later fixed defect** (No historical evidence found).

### 2. Candidate Bug-Fix Detection Behavior
* **Multi-Signal Evidence**: Detects candidate fixing commits using issue tracker references (`GH-123`, `JIRA-456`, `#789`), fix keywords (`fix`, `bug`, `issue`, `resolve`), and revert indicators.
* **Non-Exclusion of Tests**: Does *not* hard-code test-file exclusions (test file modifications are preserved if accompanied by candidate fix evidence).
* **Cosmetic Keyword Filters**: Filters out commits matching cosmetic keywords (`typo`, `formatting`, `indentation`, `style`, `lint`).

### 3. SZZ Line Tracing & Provenance Schema
* **Line-Level Blame**: Parses unified diffs (`git diff -U0 -M`) to locate deleted/modified line ranges and runs `git blame -L <start>,<end> -M -C` on the parent commit.
* **Multi-Signal Confidence Scoring**:
  * `HIGH`: Issue ID match (`GH-42`) AND fix keyword match with successful line blame.
  * `MEDIUM`: Issue ID or fix keyword match with successful line blame.
  * `LOW`: Weak keyword match or multi-parent blame dispersion.
* **Preservation of Multiple Fixes**: Multiple fixing commits tracing to the same introducing commit are aggregated without overwriting evidence (`fixing_commit_hash="abc,def"`, `fixing_issue_id="GH-42,#88"`).

---

## Controlled SZZ Experiment Results (`scratch/test_szz_repo`)

An isolated temporary repository was created under `scratch/test_szz_repo` to test controlled bug-introducing, bug-fixing, formatting, test-only, and revert commits.

| Verification Check | Expected Result | Observed Output | Status |
| :--- | :--- | :--- | :---: |
| **1. Traced Bug-Introducing Commit 1** | Flagged as `bug_introduced = 1` | `f88d28c` -> `bug_introduced = 1` | **PASSED** |
| **2. Traced Bug-Introducing Commit 2** | Flagged as `bug_introduced = 1` | `e2ab51d` -> `bug_introduced = 1` | **PASSED** |
| **3. Issue ID Capture** | Preserved `fixing_issue_id` | Captured `GH-42` | **PASSED** |
| **4. Label Semantics** | Exact non-binary label descriptions registered | Matched requested wording | **PASSED** |
| **5. Revert Case Provenance** | Reverts tracked separately via `revert_trace` | Tracked separately | **PASSED** |

---

## Phase 1 Git Repository Extractor (`src/git_extractor.py`)

Implemented a modular, reusable Git history extraction engine (`GitRepositoryExtractor`) designed to parse any local Git repository into commit-level software engineering features.

### 1. Extractor Design & Key Features
* **Native Subprocess Execution**: Uses native `git` CLI commands (`git log`, `git diff-tree --numstat -M`, `git diff-tree -U0 -M`) for zero third-party dependencies and maximum cross-platform speed.
* **Strict Commit-Time Scoping**: All features are computed strictly using diffs relative to parent commits (`commit~1` to `commit`). Zero future-leakage.
* **Chronological Order**: Commits are logged in historical sequence (`git log --reverse`).

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
Completed. Implemented `src/git_extractor.py`, created an isolated test repository in `scratch/test_repo`, verified all 8 edge cases, verified chronological ordering, and updated `PROJECT_LOG.md`.

### Phase 3 — SZZ Defect Labeling Component Implementation & Controlled Validation
Completed. Created `src/szz_labeler.py`, implemented SZZ algorithm with multi-signal confidence scoring, tested on controlled `scratch/test_szz_repo` repository, verified audit provenance schema, and updated `PROJECT_LOG.md`.

### Phase 3 — Real-Repository SZZ Validation (`bottlepy/bottle`)
Completed. Validated SZZ defect labeler on mature open-source benchmark repository `bottlepy/bottle` (1,990 commits). Produced `data/raw/szz_bottle_labeled_commits.csv`, evaluated confidence distributions, analyzed fixing linkages/issue IDs, conducted manual sanity checks, and analyzed false-positive/noise limitations.

### Phase 4.1 — Real-Data ML Dataset Construction (`src/build_dataset.py`)
Completed. Built modular dataset-building pipeline `src/build_dataset.py` joining commit-time Git features (`bottle_git_commits.csv`) and SZZ retrospective defect labels (`szz_bottle_labeled_commits.csv`) via `commit_hash`. Generated `data/processed/bottle_real_dataset.csv`. Validated data integrity, zero feature leakage, label semantics, and chronological timestamp sorting.

### Phase 4.2 — Temporal Train / Validation / Test Split (`src/temporal_split.py`)
Completed. Implemented chronological 70% / 15% / 15% temporal splitting without shuffling or stratification. Produced `bottle_train.csv` (1,181 rows), `bottle_validation.csv` (253 rows), and `bottle_test.csv` (253 rows). Verified zero commit hash overlaps, strict non-overlapping timestamp boundaries, preserved raw un-scaled feature values, and zero feature leakage.

### Phase 4.3 — Real-Data Preprocessing & Model-Aware Scaler (`src/preprocess_real_data.py`)
Completed. Built `src/preprocess_real_data.py` to fit `StandardScaler` strictly on `X_train` and exported fitted scaler to `models/real_data_scaler.pkl`. Preserved raw CSV split files untouched for tree-based models (Random Forest, XGBoost). Audited zero missing values, high feature correlations ($|r| \ge 0.70$), right-skewed distributions, and verified scaled mean=0 / std=1 on Train while validation/test reflect natural temporal distribution shifts.

### Phase 4.4 — Real-Data Baseline Model Training (`src/train_real_baseline.py`)
Completed. Trained four baseline models (DummyClassifier, LogisticRegression, RandomForest, XGBoost) strictly on `bottle_train.csv` (1,181 rows). Evaluated performance on `bottle_validation.csv` (253 rows) ONLY. Left test set (`bottle_test.csv`) 100% untouched. Saved model artifacts to `models/` and validation report to `reports/real_baseline_validation_results.json`.

### Phase 4.5 — Validation-Based Model Selection & Threshold Analysis (`src/select_model.py`)
Completed. Evaluated threshold sweeps (0.10 to 0.90 in steps of 0.05) on VALIDATION ONLY ($N=253$). Produced `reports/threshold_analysis_validation.json` and `reports/threshold_analysis_validation.csv`. Analyzed operating points (Max F1, High Recall, Balanced, High Precision), threshold stability, false-negative risk, and tree-model overfitting. Selected `LogisticRegression` at threshold `0.35` as candidate model for final locked evaluation.

### Phase 4.6 — Final Locked Model Evaluation & Phase 4 Validation (`src/evaluate_final_test.py`)
Completed. Executed ONE final unbiased evaluation of the locked model (`LogisticRegression` @ threshold `0.35`) on the completely unseen chronological test set (`bottle_test.csv`, $N=253$). Produced `reports/final_test_results.json` and `reports/final_test_predictions.csv`. Evaluated generalization gap, baseline comparison, error analysis, and 12 leakage/reproducibility checks.

---

## Phase 4.6 Final Locked Model Evaluation Results

Executed final evaluation of locked model (`LogisticRegression` at threshold `0.35`) on unseen chronological test split (`bottle_test.csv`, $N=253$, 51 positives, 202 negatives).

### 1. Final Test Performance & Baseline Comparison

| Model / Configuration | Threshold | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Predicted Positives | Confusion Matrix |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DummyClassifier (Baseline)** | Majority | 0.7984 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0 / 253 | `[[202, 0], [51, 0]]` |
| **Locked Logistic Regression** | **0.35** | **0.6522** | **0.3168** | **0.6275** | **0.4211** | **0.6709** | **101 / 253** | `[[133, 69], [19, 32]]` |

### 2. Validation vs. Final Test Comparison (Generalization Gap)

| Metric | Locked Validation (@0.35) | Final Test (@0.35) | Difference (Test - Val) | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 0.7668 | 0.6522 | -0.1146 | Shift in class distribution |
| **Precision** | 0.5851 | 0.3168 | -0.2683 | Higher FP rate due to lower actual defect rate |
| **Recall** | 0.7333 | 0.6275 | -0.1058 | Retains strong defect detection (catches 32/51 defects) |
| **F1 Score** | 0.6509 | 0.4211 | -0.2298 | Driven by precision shift from class prevalence drop |
| **ROC-AUC** | 0.7924 | 0.6709 | -0.1215 | Maintains clear discrimination over random guessing (+0.1709) |

### 3. Error Analysis Findings
* **False Positives (69 commits)**: Primarily larger refactoring commits with high line churn and multi-file touches (e.g. `48a3e07` with 52 churn across 7 files).
* **False Negatives (19 commits)**: Small, single-file localized patches (e.g. `5a6fc77` with churn=2) whose small diff footprints result in lower risk scores.
* **True Positives (32 commits)**: Successfully flagged 62.75% of all historical defect-introducing commits in the test period.

### 4. Leakage & Reproducibility Audit
* All 12 reproducibility checks passed: exact test row count 253, 0 duplicate hashes, scaler fit on train only, model trained on train only, locked threshold 0.35, zero test threshold sweeps, zero test model selection, zero SZZ provenance in feature set, test CSV unmodified.

### 5. Final Phase 4 Recommendation
**`APPROVE PHASE 4 AND PROCEED TO PHASE 5`**  
Phase 4 (Real-Data Machine Learning Pipeline) is complete, validated, leak-free, and fully documented.


---

## Phase 4.5 Validation-Based Model Selection & Threshold Analysis Results

Evaluated decision threshold trade-offs across Logistic Regression, Random Forest, and XGBoost on VALIDATION ONLY ($N=253$, 75 positives, 178 negatives). Test set (`bottle_test.csv`) remained 100% untouched.

### 1. Operating Point Summary Across Models

| Model | Operating Point | Threshold | Accuracy | Precision | Recall | F1 Score | ROC-AUC | TP | FP | TN | FN | Predicted Positives |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **Max F1 (Selected)** | **0.35** | **0.7668** | **0.5851** | **0.7333** | **0.6509** | **0.7924** | **55** | **39** | **139** | **20** | **94 / 253** |
| Logistic Regression | High Recall | 0.30 | 0.5178 | 0.3687 | 0.8800 | 0.5197 | 0.7924 | 66 | 113 | 65 | 9 | 179 / 253 |
| Logistic Regression | Balanced | 0.40 | 0.7708 | 0.6104 | 0.6267 | 0.6184 | 0.7924 | 47 | 30 | 148 | 28 | 77 / 253 |
| Logistic Regression | High Precision | 0.75 | 0.7470 | 0.7895 | 0.2000 | 0.3191 | 0.7924 | 15 | 4 | 174 | 60 | 19 / 253 |
| **Random Forest** | Max F1 | 0.45 | 0.7431 | 0.5556 | 0.6667 | 0.6061 | 0.7481 | 50 | 40 | 138 | 25 | 90 / 253 |
| Random Forest | High Recall | 0.25 | 0.6403 | 0.4394 | 0.7733 | 0.5604 | 0.7481 | 58 | 74 | 104 | 17 | 132 / 253 |
| **XGBoost** | Max F1 | 0.45 | 0.7668 | 0.5976 | 0.6533 | 0.6242 | 0.7628 | 49 | 33 | 145 | 26 | 82 / 253 |
| XGBoost | High Recall | 0.20 | 0.5850 | 0.4013 | 0.8133 | 0.5374 | 0.7628 | 61 | 91 | 87 | 14 | 152 / 253 |

### 2. Candidate Model & Threshold Selection Rationale
* **Candidate Model Selected**: `LogisticRegression` (with standardized features from `models/real_data_scaler.pkl`).
* **Candidate Threshold Selected**: `0.35`.
* **Selection Justification**:
  1. **Highest Peak F1**: Achieves **0.6509** F1 score (outperforming XGBoost's peak of 0.6242 and RF's peak of 0.6061).
  2. **Best Ranking Discrimination**: Achieves the highest ROC-AUC (**0.7924**).
  3. **Low False-Negative Risk**: At threshold 0.35, catches **73.33% of defects (55/75)** with only **20 False Negatives** (missed defects), significantly improving over baseline threshold 0.50 (39 missed defects).
  4. **Overfitting Resilience**: Demonstrates stable generalization (Train F1 = 0.5928, Val F1 = 0.6509 @ 0.35) unlike tree models which show a *"possible overfitting signal; requires further validation"*.
  5. **Threshold Stability**: Performance remains stable in the 0.35–0.40 range ($F1 \ge 0.6184$), confirming robustness against isolated validation peaks.

### 3. Phase 4.5 Recommendation
**`APPROVE FOR FINAL LOCKED EVALUATION`**  
Candidate model `LogisticRegression` and candidate threshold `0.35` are selected based on validation evidence and locked for final test evaluation in Phase 4.6.


---

## Phase 4.4 Real-Data Baseline Model Training Results

Trained four baseline models on the real software engineering dataset and evaluated on VALIDATION ONLY ($N=253$, 75 positives, 178 negatives).

### 1. Validation Performance Summary (Threshold = 0.50)

| Model | Train F1 | Val Acc | Val Prec | Val Recall | Val F1 | Val ROC-AUC | Val Pred Positives | Confusion Matrix | Overfitting Diagnostic |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DummyClassifier** | 0.0000 | 0.7036 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0 / 253 | `[[178, 0], [75, 0]]` | No gap |
| **Logistic Regression** | 0.5928 | 0.7668 | 0.6429 | 0.4800 | 0.5496 | **0.7924** | 56 / 253 | `[[158, 20], [39, 36]]` | No extreme gap |
| **Random Forest** | 0.9124 | 0.7431 | 0.5610 | **0.6133** | 0.5860 | 0.7481 | 82 / 253 | `[[142, 36], [29, 46]]` | possible overfitting signal; requires further validation |
| **XGBoost** | 0.8825 | **0.7708** | **0.6164** | 0.6000 | **0.6081** | 0.7628 | 73 / 253 | `[[150, 28], [30, 45]]` | possible overfitting signal; requires further validation |

### 2. Validation Probability Statistics ($P(y=1)$)

* **Logistic Regression**: min = 0.2083, max = 1.0000, mean = 0.3978, median = 0.3250
* **Random Forest**: min = 0.0000, max = 0.9900, mean = 0.3644, median = 0.2700
* **XGBoost**: min = 0.0054, max = 0.9993, mean = 0.3715, median = 0.2306

### 3. Key Observations & Overfitting Diagnostics
* **Signal Discovery**: All non-dummy models outperform the naive majority baseline, demonstrating genuine predictive signal in commit-time features ($X$) for defect risk ($y$).
* **Model Differences**:
  * **XGBoost** achieves the highest Validation F1 (**0.6081**) and Accuracy (**0.7708**).
  * **Logistic Regression** achieves the highest Validation ROC-AUC (**0.7924**) and Precision (**0.6429**).
  * **Random Forest** achieves the highest Validation Recall (**0.6133**).
* **Overfitting Diagnostics**: Both Random Forest (Train F1 = 0.9124 vs Val F1 = 0.5860) and XGBoost (Train F1 = 0.8825 vs Val F1 = 0.6081) exhibit a *"possible overfitting signal; requires further validation"*.

### 4. Phase 4.4 Recommendation
**`APPROVE FOR VALIDATION-BASED MODEL SELECTION`**  
Baseline models are trained, artifacts are saved, test set remains completely untouched, and validation metrics provide a rigorous baseline for model selection and threshold analysis.


---

## Phase 4.3 Real-Data Preprocessing & Scaler Results

Implemented model-aware feature preprocessing with strict Train-only scaling isolation and saved `models/real_data_scaler.pkl`.

### 1. Preprocessing Pipeline Architecture
* **Scaler Fitting**: `StandardScaler` fitted **ONLY** on `X_train` (1,181 commits). `X_val` and `X_test` transformed using training statistics ($\mu, \sigma$).
* **Model-Aware Handling**:
  * Logistic Regression: consumes standardized arrays (`X_train_scaled`, `X_val_scaled`, `X_test_scaled`).
  * Tree-based Models (Random Forest, XGBoost): consume raw unscaled DataFrames.
* **Artifact Location**: `models/real_data_scaler.pkl` (persisted via `joblib`).
* **Raw CSV Preservation**: `bottle_train.csv`, `bottle_validation.csv`, `bottle_test.csv` remain intact and un-overwritten.

### 2. Missing-Value Audit
* **Train / Val / Test Missing Values**: **0** missing values across all 10 features.

### 3. High Correlation Audit ($|r| \ge 0.70$ on TRAIN)
* `lines_added` $\leftrightarrow$ `code_churn`: $r = 0.9379$
* `code_churn` $\leftrightarrow$ `max_lines_changed_in_single_file`: $r = 0.7896$
* `avg_lines_changed_per_file` $\leftrightarrow$ `max_lines_changed_in_single_file`: $r = 0.8445$
* *Decision*: All 10 features retained for baseline evaluation; correlated features are handled by tree models and evaluated in model comparison.

### 4. Scaled Feature Statistics
* **TRAIN Scaled**: mean $= 0.0000$, std $= 1.0000$ across all 10 features.
* **VALIDATION Scaled**: mean range $[-0.1453, +1.3394]$, std range $[0.9495, 14.3742]$.
* **TEST Scaled**: mean range $[-0.2154, +0.9243]$, std range $[0.9703, 15.2409]$.
* *Interpretation*: Non-zero scaled means/stds in Validation and Test accurately capture real-world temporal distribution shifts over time.

### 5. Final Phase 4.3 Recommendation
**`APPROVE FOR REAL-DATA MODEL TRAINING`**  
The preprocessing pipeline and scaler artifact are complete, leak-free, model-aware, and ready for model training in Phase 4.4.


---

## Phase 4.2 Temporal Train / Validation / Test Split Results

Built and validated chronological dataset splits for temporal evaluation: `bottle_train.csv`, `bottle_validation.csv`, and `bottle_test.csv`.

### 1. Split Row Counts & Proportions (70% / 15% / 15%)
* **TRAIN Set (Oldest 70%)**: 1,181 commits (70.01%)
* **VALIDATION Set (Middle 15%)**: 253 commits (15.00%)
* **TEST Set (Newest 15%)**: 253 commits (15.00%)
* **Total Sum Checksum**: 1,181 + 253 + 253 = 1,687 rows (100% exact match).

### 2. Temporal Boundary Validation
* **TRAIN Timestamp Range**: `2009-06-30T20:27:53+02:00` $\rightarrow$ `2014-03-11T15:05:46-03:00`
* **VALIDATION Timestamp Range**: `2014-03-12T00:32:55-07:00` $\rightarrow$ `2017-01-09T14:42:35+01:00`
* **TEST Timestamp Range**: `2017-01-18T18:04:32+08:00` $\rightarrow$ `2026-09-06T13:21:24+02:00`
* **Boundary Checks**: `TRAIN max timestamp < VAL min timestamp` (True) and `VAL max timestamp < TEST min timestamp` (True). Zero temporal overlap.

### 3. Natural Historical Class Distribution
* **TRAIN Set**: Positive ($y=1$) = 532 (45.05%), Negative ($y=0$) = 649 (54.95%)
* **VALIDATION Set**: Positive ($y=1$) = 75 (29.64%), Negative ($y=0$) = 178 (70.36%)
* **TEST Set**: Positive ($y=1$) = 51 (20.16%), Negative ($y=0$) = 202 (79.84%)
* **Observation**: Natural software lifecycle trend where defect-introducing commit rates decrease as codebase matures. The test set contains 51 positive examples and 202 negative examples, providing solid representation for both classes.

### 4. Integrity & Leakage Protection Audits
* **Overlaps**: 0 duplicate commit hashes across splits. Every commit appears in exactly one split.
* **Feature Consistency**: All 10 ML features present, numeric, and zero nulls across all 3 splits. `code_churn == lines_added + lines_deleted` verified on 100% of rows.
* **Preprocessing State**: Preprocessing (`StandardScaler`) has **NOT** yet occurred. Feature values remain raw.
* **Model Training State**: Model training has **NOT** yet occurred.

### 5. Final Phase 4.2 Recommendation
**`APPROVE FOR TEMPORAL PREPROCESSING`**  
The temporal split artifacts are created, leak-free, strictly chronological, and ready for Phase 4.3 feature standardization (fitting scaler on TRAIN only).


---

## Phase 4 Real-Data ML Dataset Construction Results (`bottle_real_dataset.csv`)

Built and validated the first real-world ML dataset for PatchGuard by combining commit-time features with SZZ defect labels. Output saved to `data/processed/bottle_real_dataset.csv`.

### 1. Dataset Construction Architecture & Join Strategy
* **Primary Key**: `commit_hash` (7-character short SHA matching full SHA).
* **Join Strategy**: `LEFT JOIN` from commit-time Git features (`df_features`) to SZZ labels (`df_szz`). Commit-time history is primary.
* **Join Coverage**:
  * Git feature commits: 1,687 non-merge commits.
  * SZZ labeled commits: 1,990 commits (includes 303 merge commits).
  * Feature commits missing SZZ match: **0** (100% feature commit coverage).
  * SZZ merge commits omitted from features: 303 (intentionally excluded merge commits to prevent churn duplication).

### 2. Feature Schema & Leakage Protection
* **Features Included (10 ML Features)**: `lines_added`, `lines_deleted`, `code_churn`, `files_changed`, `functions_changed`, `num_directories_touched`, `is_test_file_modified`, `avg_lines_changed_per_file`, `max_lines_changed_in_single_file`, `num_source_files_changed`.
* **Identifiers & Metadata**: `commit_hash`, `commit_timestamp`.
* **Target Label**: `bug_introduced` (`1` = SZZ-identified bug-introducing commit, `0` = No SZZ evidence that the commit introduced a later fixed defect).
* **Provenance Columns Excluded**: `fixing_commit_hash`, `fixing_issue_id`, `label_confidence`, `label_source`, `label_semantics`, `traced_lines_count` are strictly excluded from ML dataset features.

### 3. Data Integrity & Validation Checks
* **Total Dimensions**: 1,687 rows × 13 columns.
* **Duplicates**: 0 duplicate `commit_hash` records.
* **Missing Values**: 0 missing values across all feature and target columns.
* **Label Distribution**:
  * `1` (SZZ-identified defect introducing): 658 commits (39.00%)
  * `0` (No SZZ evidence): 1,029 commits (61.00%)
* **Invariants Verified**: `code_churn == lines_added + lines_deleted` across 100% of rows. All 10 ML features strictly numeric.
* **Temporal Sorting**: Timestamps sorted in chronological order (`2009-06-30` to `2026-09-06`). `is_monotonic_increasing = True`.

### 4. Phase 4 Readiness Recommendation
**`APPROVE FOR TEMPORAL ML TRAINING`**  
The real-world ML dataset is fully constructed, validated, leak-free, and chronologically ordered for temporal train/validation/test splitting in Phase 4.


---

## Real-Repository SZZ Validation Results (`bottlepy/bottle`)

Executed real-world SZZ defect extraction on `bottlepy/bottle` (`scratch/external_repo/bottle`). Output saved to `data/raw/szz_bottle_labeled_commits.csv`.

### 1. Quantitative Benchmark Statistics
* **Target Repository**: `bottlepy/bottle`
* **Total Commits Analyzed**: 1,990
* **Execution Runtime**: 1,202.27 seconds (~20 minutes)
* **Candidate Fix Commits Found**: 494 (24.82%)
* **SZZ Bug-Introducing Commits ($y=1$)**: 661 (33.22%)
* **No SZZ Evidence Commits ($y=0$)**: 1,329 (66.78%)

### 2. Confidence & Label Source Distribution
* **Confidence Breakdown**:
  * `NONE` ($y=0$): 1,329 (66.78%)
  * `MEDIUM`: 588 (29.55%)
  * `HIGH`: 73 (3.67%)
  * `LOW`: 0 (0.00%)
* **Label Source Breakdown**:
  * `unlabeled_no_evidence`: 1,329 (66.78%)
  * `szz_blame_traced`: 651 (32.71%)
  * `revert_trace`: 10 (0.50%)

### 3. Provenance & Fixing Linkages
* **Total Fixing Linkages**: 1,384 distinct introducing-to-fixing mappings.
* **Commits Blamed by Multiple Fixes**: 293 commits.
* **Unique Issue IDs Captured**: 21 explicit issue references (e.g., `Issue #12`, `issue #85`, `Issue 21`).

### 4. Noise & SZZ Limitations Analysis
* **Initial/Large Commit Over-Blaming**: Large initial commits (e.g. `4f50cec`) collect line blames from many subsequent fix commits because large portions of code originate there.
* **Added-Only Code Defect Limitation**: Bug fixes that consist entirely of adding new lines (0 deleted lines) cannot be line-traced back to an originating commit.
* **Cosmetic & Refactoring Noise**: Deleting/reformatting existing lines in a fix commit traces back to whoever last touched those lines, even if the original commit was logically correct.

### 5. Final Phase 3 Recommendation
**`APPROVE SZZ FOR REAL-DATA ML INTEGRATION`**  
The SZZ defect labeler successfully extracted defect labels and complete provenance tracking on a real Python codebase. Labeling semantics, candidate fix detection, revert handling, and multi-signal confidence scoring operated reliably.


---

## Phase 6.5 — Real LLM Provider Integration (OpenAI)

Integrated the first real LLM provider (`OpenAIProvider`) into PatchGuard's Phase 6 code change analysis architecture.

### 1. Objective & Architectural Preservation
* **Provider Abstraction**: Preserved the `LLMProvider` abstract base class in `src/analysis/providers.py`.
* **Provider Independence**: `LLMCodeAnalyzer` in `src/analysis/analyzer.py` continues depending strictly on `LLMProvider`, keeping `MockLLMProvider` and `OpenAIProvider` fully interchangeable.
* **Offline Mock Preservation**: `MockLLMProvider` remains functional and untouched.

### 2. OpenAI Provider Implementation (`src/analysis/providers.py`)
* **SDK / API Approach**: Uses official `openai` Python SDK (v1.0.0+ syntax) Chat Completions API with `OpenAI(api_key=..., timeout=...)`.
* **Model Configuration**: Configurable model via `OpenAIProvider(model="...")` with default `gpt-4o-mini`.
* **API Key Handling**: Resolves `OPENAI_API_KEY` strictly from environment variable or constructor override. Throws `MissingAPIKeyError` if key is missing/empty. Never hard-codes or logs secret keys.
* **Structured Output Mechanism**: Requests structured JSON response text using `response_format={"type": "json_object"}` parameter. Returns raw text to downstream `ResponseParser`.
* **Timeout & Error Handling**: Bounded 30.0s timeout with application-level exception mapping (`OpenAIProviderError`, `MissingAPIKeyError`).

### 3. Dependencies & Privacy Boundary
* **Dependencies**: Added `openai>=1.0.0` to `requirements.txt`.
* **Privacy Boundary**: Documented that `OpenAIProvider` transmits code diff prompt content to third-party endpoints (`api.openai.com`). Diff analysis should be evaluated against sensitivity guidelines.

### 4. Verification & Testing
* **Test Suite**: Created `tests/test_openai_provider.py` (8 test cases) using `unittest.mock`.
* **Network Independence**: 100% offline tests; zero real network requests or real API keys used.
* **Total Passing Tests**: 72/72 tests passing (`python -m unittest discover -s tests`).

### 5. Known Limitations & Next Step
* **Limitations**: Live calls require external API access and an active `OPENAI_API_KEY`.
* **Next Phase**: **Phase 6.6 — CLI Integration**
