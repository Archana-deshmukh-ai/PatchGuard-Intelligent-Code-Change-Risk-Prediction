"""
Phase 8 Experiment Configuration and Constant Definitions.
"""

import os
from typing import Dict, Any, List

FEATURE_COLUMNS: List[str] = [
    'lines_added',
    'lines_deleted',
    'code_churn',
    'files_changed',
    'functions_changed',
    'num_directories_touched',
    'is_test_file_modified',
    'avg_lines_changed_per_file',
    'max_lines_changed_in_single_file',
    'num_source_files_changed'
]

TARGET_COLUMN: str = 'bug_introduced'
IDENTIFIER_COLUMNS: List[str] = ['commit_hash', 'commit_timestamp']

DEFAULT_SEED: int = 42
EXPERIMENT_VERSION: str = "v1.0.0-phase8"
PRODUCTION_THRESHOLD: float = 0.35

# Spectrum of decision thresholds evaluated on Validation
THRESHOLD_SWEEP: List[float] = [
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
]

# Hyperparameter Grid Definitions (Small, deliberate candidate space)
LOGISTIC_REGRESSION_GRID: List[Dict[str, Any]] = [
    {"C": 0.01, "solver": "liblinear", "random_state": DEFAULT_SEED, "max_iter": 1000},
    {"C": 0.1,  "solver": "liblinear", "random_state": DEFAULT_SEED, "max_iter": 1000},
    {"C": 1.0,  "solver": "liblinear", "random_state": DEFAULT_SEED, "max_iter": 1000},
    {"C": 10.0, "solver": "liblinear", "random_state": DEFAULT_SEED, "max_iter": 1000}
]

RANDOM_FOREST_GRID: List[Dict[str, Any]] = [
    {"n_estimators": 100, "max_depth": 3,    "min_samples_split": 5,  "n_jobs": 1, "random_state": DEFAULT_SEED},
    {"n_estimators": 100, "max_depth": 5,    "min_samples_split": 5,  "n_jobs": 1, "random_state": DEFAULT_SEED},
    {"n_estimators": 100, "max_depth": 8,    "min_samples_split": 5,  "n_jobs": 1, "random_state": DEFAULT_SEED},
    {"n_estimators": 200, "max_depth": 5,    "min_samples_split": 5,  "n_jobs": 1, "random_state": DEFAULT_SEED},
    {"n_estimators": 200, "max_depth": None, "min_samples_split": 10, "n_jobs": 1, "random_state": DEFAULT_SEED}
]

XGBOOST_GRID: List[Dict[str, Any]] = [
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "n_jobs": 1, "eval_metric": "logloss", "random_state": DEFAULT_SEED},
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "n_jobs": 1, "eval_metric": "logloss", "random_state": DEFAULT_SEED},
    {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.01, "n_jobs": 1, "eval_metric": "logloss", "random_state": DEFAULT_SEED},
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "n_jobs": 1, "eval_metric": "logloss", "random_state": DEFAULT_SEED},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.10, "n_jobs": 1, "eval_metric": "logloss", "random_state": DEFAULT_SEED}
]
