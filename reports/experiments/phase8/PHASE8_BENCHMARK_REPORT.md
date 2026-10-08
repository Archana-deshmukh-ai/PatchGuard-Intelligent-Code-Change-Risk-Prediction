# Phase 8 ML Model Benchmarking & Risk Prediction Report

## Executive Summary
- **Experiment Version**: `v1.0.0-phase8`
- **Timestamp (UTC)**: `2026-10-08T09:30:24.177078`
- **Git Commit SHA**: `df651b50f8b82e64364e350a0b34911e4ed35fa2`
- **Random Seed**: `42`
- **Validation Winner**: **XGBoost**
- **Recommendation**: **PRESERVE_PRODUCTION_LOGISTIC_REGRESSION: Neither RF nor XGBoost provided a defensible, substantial improvement over the production LR baseline under temporal evaluation. Keep Logistic Regression as production default.**

---

## Dataset & Split Provenance
- **Total Dataset Commits**: 1687
- **Train Split**: 1181 commits (Positive rate: 45.05%)
- **Validation Split**: 253 commits (Positive rate: 29.64%)
- **Test Split**: 253 commits (Positive rate: 20.16%)
- **Strict Monotonic Temporal Splitting**: True
- **Zero Commit Hash Overlap**: True

---

## Benchmark Results Summary

| Model | Frozen Thresh | Val PR-AUC | Val ROC-AUC | Val F1 | Test PR-AUC | Test ROC-AUC | Test F1 | Test Prec | Test Rec | Winner? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Production_LogisticRegression_v1 | 0.35 | 0.6220 | 0.7924 | 0.6509 | 0.3086 | 0.6709 | 0.4211 | 0.3168 | 0.6275 | No |
| Phase8_LogisticRegression | 0.35 | 0.6273 | 0.7953 | 0.6467 | 0.3115 | 0.6732 | 0.4238 | 0.3200 | 0.6275 | No |
| Phase8_RandomForest | 0.45 | 0.6127 | 0.7909 | 0.6242 | 0.3778 | 0.7089 | 0.4211 | 0.3415 | 0.5490 | No |
| Phase8_XGBoost | 0.40 | 0.6293 | 0.7817 | 0.6258 | 0.3535 | 0.7122 | 0.4697 | 0.3827 | 0.6078 | **YES** |

---

## Detailed Candidate Model Analysis

### LogisticRegression
- **Features Used**: `scaled`
- **Best Hyperparameters**: `{"C": 10.0, "solver": "liblinear", "random_state": 42, "max_iter": 1000}`
- **Frozen Threshold**: `0.35`
- **Validation Metrics**: PR-AUC: `0.6273`, ROC-AUC: `0.7953`, F1: `0.6467`, Precision: `0.587`, Recall: `0.72`
- **Final Test Metrics**: PR-AUC: `0.3115`, ROC-AUC: `0.6732`, F1: `0.4238`, Precision: `0.32`, Recall: `0.6275`
- **Validation to Test Degradation**: PR-AUC Diff: `-0.3158`, ROC-AUC Diff: `-0.1221`, F1 Diff: `-0.2229`

### RandomForest
- **Features Used**: `raw`
- **Best Hyperparameters**: `{"n_estimators": 200, "max_depth": 5, "min_samples_split": 5, "n_jobs": 1, "random_state": 42}`
- **Frozen Threshold**: `0.45`
- **Validation Metrics**: PR-AUC: `0.6127`, ROC-AUC: `0.7909`, F1: `0.6242`, Precision: `0.5976`, Recall: `0.6533`
- **Final Test Metrics**: PR-AUC: `0.3778`, ROC-AUC: `0.7089`, F1: `0.4211`, Precision: `0.3415`, Recall: `0.549`
- **Validation to Test Degradation**: PR-AUC Diff: `-0.2349`, ROC-AUC Diff: `-0.0820`, F1 Diff: `-0.2031`

### XGBoost
- **Features Used**: `raw`
- **Best Hyperparameters**: `{"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "n_jobs": 1, "eval_metric": "logloss", "random_state": 42}`
- **Frozen Threshold**: `0.4`
- **Validation Metrics**: PR-AUC: `0.6293`, ROC-AUC: `0.7817`, F1: `0.6258`, Precision: `0.5795`, Recall: `0.68`
- **Final Test Metrics**: PR-AUC: `0.3535`, ROC-AUC: `0.7122`, F1: `0.4697`, Precision: `0.3827`, Recall: `0.6078`
- **Validation to Test Degradation**: PR-AUC Diff: `-0.2758`, ROC-AUC Diff: `-0.0695`, F1 Diff: `-0.1561`

---

## Production Artifact Safety Audit
- **`models/logistic_regression_real.pkl`**: Modified? **NO (Preserved 100%)**
- **`models/real_data_scaler.pkl`**: Modified? **NO (Preserved 100%)**
- **Production Prediction Behavior**: **UNTOUCHED**
