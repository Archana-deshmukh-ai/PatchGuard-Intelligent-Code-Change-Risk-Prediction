# PatchGuard: Intelligent Code Change Risk Prediction

Predicting defect-inducing code changes using machine learning, Git repository mining, SZZ defect labeling, code churn analysis, and contextual intelligence.

---

## 1. Project Overview
**PatchGuard** is a software engineering machine learning system designed to analyze code changes (Git commits and Pull Requests) and estimate the probability that a change will introduce a defect/bug into production.

The long-term goal of PatchGuard is to serve as an intelligent assistant in developer workflows—flagging high-risk code modifications, explaining *why* a change is risky using Large Language Models (LLMs), and providing contextual recommendations grounded in repository documentation and historical issues via Retrieval-Augmented Generation (RAG).

---

## 2. Current Project Status
**Phase 7.3 Complete — Automated GitHub PR Workflow + Reporting**

PatchGuard features an automated GitHub Pull Request workflow orchestrator (`src/github/workflow.py`), HMAC-SHA256 webhook event validator (`src/github/webhook.py`), and idempotent PR comment reporter (`src/github/reporter.py`), alongside the unified CLI (`src/cli.py`).

- **Automated Webhook Workflow Architecture**:
  - `GitHubPRWorkflow`: Receives incoming raw webhook payloads and HTTP headers, verifies HMAC-SHA256 signatures against `GITHUB_WEBHOOK_SECRET` before JSON parsing, filters for supported actions (`opened`, `synchronize`, `reopened`), retrieves PR metadata via `GitHubClient`, acquires authenticated temporary workspace via `RepositoryAcquisitionManager`, runs per-commit ML defect predictions and cumulative LLM diff analysis via `PRAnalyzer`, and publishes markdown risk reports to GitHub PRs via `GitHubPRReporter`.
  - `GitHubPRReporter`: Formats markdown PR comments with stable idempotency tag `<!-- patchguard-report -->`. Searches existing PR issue comments (GET `/repos/{owner}/{repo}/issues/{number}/comments`) and updates existing comments (PATCH) instead of creating duplicate comments (POST) on `synchronize` events.
- **Security & Secret Hygiene Boundary**:
  - HMAC-SHA256 signature verification uses constant-time comparison (`hmac.compare_digest`) before JSON parsing.
  - Zero repository code execution (no setup.py, npm, build, or test scripts executed).
  - Credentials, JWTs, installation tokens, and authorization headers are sanitized and never exposed in logs, tracebacks, exception strings, or GitHub PR comments.
  - Required GitHub App Permissions: Minimum read-only repository contents (`contents:read`), pull requests (`pull_requests:read`), and issue/PR comment write access (`issues:write`).
- **Offline Test Verification**:
  - Full automated test suite (160 passing unit tests) operates 100% offline using mock HTTP fetchers and temporary Git repositories without external network dependencies.


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
[ Unbiased Final Test Evaluation (src/evaluate_final_test.py) ]
              │
              ▼
[ Risk Prediction Engine & CLI Tool (src/cli.py) ] ──► (Phase 5 Completed)
              │
              ▼
[ LLM Code-Change Analysis Pipeline (src/analysis/) ] ──► (Phase 6.5 Completed)
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

# 8. Predict defect risk for a commit via CLI (Quantitative ML prediction)
python src/cli.py predict --repo ./path/to/repo --commit HEAD

# 9. Predict defect risk for a commit via CLI with JSON output format
python src/cli.py predict --repo ./path/to/repo --commit HEAD --format json

# 10. Analyze code change risk with combined ML risk prediction and LLM analysis (Mock provider)
python src/cli.py analyze --repo ./path/to/repo --commit HEAD --provider mock --format text

# 11. Analyze code change risk with live OpenAI provider in JSON format
python src/cli.py analyze --repo ./path/to/repo --commit HEAD --provider openai --format json --model gpt-4o-mini

# 12. Analyze local Pull Request (per-commit ML predictions + cumulative BASE->HEAD LLM analysis)
python src/cli.py pr-analyze --repo ./path/to/repo --base main --head feature/my-change

# 14. Analyze GitHub Pull Request metadata via GitHub API (read-only adapter)
python src/cli.py github-pr-analyze --repo owner/repository --pr 42

# 15. Analyze GitHub Pull Request with local git repo path override and JSON format
python src/cli.py github-pr-analyze --repo owner/repository --pr 42 --local-repo ./path/to/repo --format json
```

---

## 9. Current Limitations & Key Learnings

1. **Shallow Feature Representation**: Commit-level diff metrics capture commit size and dispersion, but do not capture deep AST structure or semantic code logic.
2. **Detection of Small Patches**: Small, single-file bug fixes produce small diff footprints, making them harder for churn metrics alone to detect (leading to False Negatives).
3. **Refactoring Noise**: Large structural refactorings exhibit high churn and multi-file touches, increasing predicted risk scores even when no defects are introduced (leading to False Positives).
4. **Temporal Class Prevalence Shift**: Defect-introducing commit frequency decreases as codebases mature (from 45% in early history to 20% in recent commits), affecting fixed-threshold precision across multi-year timeframes.
5. **Uncalibrated Probability Scores**: Model probabilities represent discrimination scores and have not yet been formally calibrated into real-world probabilities.
6. **Synchronous Webhook Processing**: Webhook event processing operates synchronously within the request handler; background job queues (Redis/Celery) are intentionally out of scope for Phase 7.3.
7. **External Fork PR Acquisition**: Pull requests originating from external repository forks where head commit SHAs are not present in the base repository require fetching from fork remotes; cross-repository fork PR acquisition is currently unsupported in Phase 7.3.


---

## 10. Project Roadmap

- [x] **Phase 1: Project & ML Foundation** — Supervised binary classification framing, synthetic pipeline verification, baseline evaluation metrics.
- [x] **Phase 2: Real Git Repository Mining** — Subprocess-based `GitRepositoryExtractor` parsing 10 commit-time diff features across Git history.
- [x] **Phase 3: Defect Labeling / SZZ** — Retrospective line-blame tracing (`SZZDefectLabeler`), multi-signal confidence scoring, tested on `bottlepy/bottle`.
- [x] **Phase 4: Real-Data ML Pipeline** — Dataset construction, 70/15/15 chronological split, Train-only scaling, multi-model baseline training, validation threshold selection, final locked test evaluation.
- [x] **Phase 5: Risk Prediction Engine** — Inference-only `RiskPredictionEngine`, thin CLI (`patchguard predict`), model signals, SHA-256 artifact hashes, 10-feature schema enforcement.
- [x] **Phase 6: LLM Code-Change Analysis & CLI** — Structured Git diff extraction, prompt construction, LLM provider integration (Mock + OpenAI), JSON parsing, evidence validation, and unified CLI (`patchguard analyze`).
- [x] **Phase 7.1: Local PR Analysis Core** — `PRAnalyzer`, `PRAnalysisInput`, `PRAnalysisResult`, cumulative merge-base diff extraction, per-commit ML predictions, and `patchguard pr-analyze` CLI.
- [x] **Phase 7.2.1: GitHub Read-Only PR Retrieval + Adapter** — Read-only `GitHubClient`, RS256 JWT / token authentication (`GitHubAppAuthenticator`), PR schema mapping (`GitHubPRAdapter`), and `github-pr-analyze` CLI subcommand.
- [x] **Phase 7.2.2: Authenticated GitHub Repository Acquisition** — `RepositoryAcquisitionManager`, temporary workspace lifecycle, `-c include.path` secure token transport, SHA & origin URL validation, end-to-end `github-pr-analyze` CLI.
- [x] **Phase 7.3: Automated GitHub PR Workflow + Reporting** — Webhook HMAC-SHA256 signature verification, event filtering (`opened`, `synchronize`, `reopened`), `GitHubPRWorkflow` orchestration, and idempotent markdown PR comment reporting (`GitHubPRReporter`).
- [ ] **Phase 8: Developer Dashboard** *(NEXT)* — Interactive UI for commit risk monitoring and audit reports.
- [ ] **Phase 9: Productionization** — End-to-end API, containerization, and production deployment.
