"""
Unit Test Suite for Phase 8 ML Model Benchmarking & Risk Prediction Improvement.
"""

import os
import sys
import json
import unittest
import tempfile
import shutil
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiments.config import (
    FEATURE_COLUMNS, TARGET_COLUMN, DEFAULT_SEED, PRODUCTION_THRESHOLD, THRESHOLD_SWEEP
)
from src.experiments.evaluator import (
    ModelTrainerEvaluator, compute_detailed_metrics, sweep_validation_thresholds
)
from src.experiments.runner import Phase8BenchmarkRunner

class TestPhase8Benchmarking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.project_root = project_root
        cls.data_dir = os.path.join(project_root, 'data', 'processed')
        cls.prod_models_dir = os.path.join(project_root, 'models')

    def test_1_canonical_feature_schema_preserved(self):
        """1. Enforces exact canonical 10-feature schema names and ordering."""
        self.assertEqual(len(FEATURE_COLUMNS), 10)
        expected_schema = [
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
        self.assertEqual(FEATURE_COLUMNS, expected_schema)

    def test_2_temporal_split_sizes_and_monotonicity(self):
        """2. Verifies temporal split sizes (1,181 / 253 / 253) and zero hash overlap."""
        runner = Phase8BenchmarkRunner(self.project_root)
        split_info = runner.validate_split_integrity()

        self.assertEqual(split_info["train_rows"], 1181)
        self.assertEqual(split_info["val_rows"], 253)
        self.assertEqual(split_info["test_rows"], 253)
        self.assertEqual(split_info["total_rows"], 1687)
        self.assertTrue(split_info["zero_hash_overlap"])
        self.assertTrue(split_info["strictly_chronological"])

    def test_3_preprocessing_fitted_only_on_train_data(self):
        """3. Confirms StandardScaler in evaluator is fitted ONLY on X_train."""
        df_train = pd.read_csv(os.path.join(self.data_dir, 'bottle_train.csv'))
        df_val = pd.read_csv(os.path.join(self.data_dir, 'bottle_validation.csv'))
        df_test = pd.read_csv(os.path.join(self.data_dir, 'bottle_test.csv'))

        evaluator = ModelTrainerEvaluator(df_train, df_val, df_test)

        # Independent scaler fit on train ONLY
        independent_scaler = StandardScaler()
        independent_scaler.fit(df_train[FEATURE_COLUMNS])

        np.testing.assert_array_almost_equal(evaluator.scaler.mean_, independent_scaler.mean_)
        np.testing.assert_array_almost_equal(evaluator.scaler.var_, independent_scaler.var_)

    def test_4_validation_test_data_do_not_influence_scaler(self):
        """4. Modifying validation/test feature values leaves the fitted scaler parameters unchanged."""
        df_train = pd.read_csv(os.path.join(self.data_dir, 'bottle_train.csv'))
        df_val = pd.read_csv(os.path.join(self.data_dir, 'bottle_validation.csv'))
        df_test = pd.read_csv(os.path.join(self.data_dir, 'bottle_test.csv'))

        evaluator1 = ModelTrainerEvaluator(df_train, df_val, df_test)

        # Mutate validation and test features heavily
        df_val_mutated = df_val.copy()
        df_val_mutated[FEATURE_COLUMNS] = df_val_mutated[FEATURE_COLUMNS] * 1000.0

        df_test_mutated = df_test.copy()
        df_test_mutated[FEATURE_COLUMNS] = df_test_mutated[FEATURE_COLUMNS] + 5000.0

        evaluator2 = ModelTrainerEvaluator(df_train, df_val_mutated, df_test_mutated)

        np.testing.assert_array_equal(evaluator1.scaler.mean_, evaluator2.scaler.mean_)
        np.testing.assert_array_equal(evaluator1.scaler.var_, evaluator2.scaler.var_)

    def test_5_deterministic_model_training(self):
        """5. Training models with DEFAULT_SEED (42) yields deterministic results."""
        df_train = pd.read_csv(os.path.join(self.data_dir, 'bottle_train.csv'))
        X_train = df_train[FEATURE_COLUMNS]
        y_train = df_train[TARGET_COLUMN]

        rf1 = RandomForestClassifier(n_estimators=50, random_state=DEFAULT_SEED)
        rf1.fit(X_train, y_train)

        rf2 = RandomForestClassifier(n_estimators=50, random_state=DEFAULT_SEED)
        rf2.fit(X_train, y_train)

        p1 = rf1.predict_proba(X_train)
        p2 = rf2.predict_proba(X_train)

        np.testing.assert_array_equal(p1, p2)

    def test_6_evaluator_returns_all_required_metrics(self):
        """6. compute_detailed_metrics calculates all required classification and discrimination metrics."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.4, 0.3, 0.1, 0.75])

        metrics = compute_detailed_metrics(y_true, y_prob, threshold=0.50)

        required_keys = [
            "decision_threshold", "total_samples", "positive_samples", "negative_samples",
            "positive_prevalence", "predicted_positives", "predicted_positive_rate",
            "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score",
            "confusion_matrix", "tp", "fp", "tn", "fn"
        ]
        for key in required_keys:
            self.assertIn(key, metrics)

        self.assertEqual(metrics["total_samples"], 8)
        self.assertEqual(metrics["positive_samples"], 4)
        self.assertTrue(0.0 <= metrics["pr_auc"] <= 1.0)
        self.assertTrue(0.0 <= metrics["roc_auc"] <= 1.0)
        self.assertTrue(0.0 <= metrics["brier_score"] <= 1.0)

    def test_7_threshold_selection_uses_validation_data_only(self):
        """7. sweep_validation_thresholds returns operating points across threshold sweep."""
        y_val = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
        y_val_prob = np.array([0.85, 0.15, 0.75, 0.30, 0.65, 0.20, 0.90, 0.10])

        sweep, op_dict = sweep_validation_thresholds(y_val, y_val_prob, THRESHOLD_SWEEP)

        self.assertEqual(len(sweep), len(THRESHOLD_SWEEP))
        self.assertIn("max_f1", op_dict)
        self.assertIn("high_recall", op_dict)
        self.assertIn("balanced", op_dict)
        self.assertIn("high_precision", op_dict)
        self.assertIn("production_0_35", op_dict)
        self.assertEqual(op_dict["production_0_35"]["decision_threshold"], PRODUCTION_THRESHOLD)

    def test_8_phase8_artifacts_isolated(self):
        """8. Running benchmark creates artifacts in isolated models/experiments/phase8 and reports/experiments/phase8."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Copy data directory into temp_dir
            temp_data_dir = os.path.join(temp_dir, 'data', 'processed')
            os.makedirs(temp_data_dir, exist_ok=True)
            for f in ['bottle_train.csv', 'bottle_validation.csv', 'bottle_test.csv']:
                shutil.copy(os.path.join(self.data_dir, f), os.path.join(temp_data_dir, f))

            temp_prod_models = os.path.join(temp_dir, 'models')
            os.makedirs(temp_prod_models, exist_ok=True)
            # Copy prod artifacts
            shutil.copy(os.path.join(self.prod_models_dir, 'logistic_regression_real.pkl'), os.path.join(temp_prod_models, 'logistic_regression_real.pkl'))
            shutil.copy(os.path.join(self.prod_models_dir, 'real_data_scaler.pkl'), os.path.join(temp_prod_models, 'real_data_scaler.pkl'))

            runner = Phase8BenchmarkRunner(temp_dir)
            report = runner.run_benchmark()

            exp_models = os.path.join(temp_dir, 'models', 'experiments', 'phase8')
            exp_reports = os.path.join(temp_dir, 'reports', 'experiments', 'phase8')

            self.assertTrue(os.path.exists(os.path.join(exp_models, 'logisticregression_phase8.pkl')))
            self.assertTrue(os.path.exists(os.path.join(exp_models, 'randomforest_phase8.pkl')))
            self.assertTrue(os.path.exists(os.path.join(exp_models, 'xgboost_phase8.pkl')))
            self.assertTrue(os.path.exists(os.path.join(exp_models, 'phase8_scaler.pkl')))

            self.assertTrue(os.path.exists(os.path.join(exp_reports, 'phase8_model_benchmark.json')))
            self.assertTrue(os.path.exists(os.path.join(exp_reports, 'phase8_benchmark_summary.csv')))
            self.assertTrue(os.path.exists(os.path.join(exp_reports, 'PHASE8_BENCHMARK_REPORT.md')))
        finally:
            shutil.rmtree(temp_dir)

    def test_9_existing_production_artifacts_unmodified(self):
        """9. Production artifacts (logistic_regression_real.pkl, real_data_scaler.pkl) hashes remain identical after benchmark."""
        prod_model_path = os.path.join(self.prod_models_dir, 'logistic_regression_real.pkl')
        prod_scaler_path = os.path.join(self.prod_models_dir, 'real_data_scaler.pkl')

        mtime_model_before = os.path.getmtime(prod_model_path)
        mtime_scaler_before = os.path.getmtime(prod_scaler_path)

        runner = Phase8BenchmarkRunner(self.project_root)
        runner.run_benchmark()

        mtime_model_after = os.path.getmtime(prod_model_path)
        mtime_scaler_after = os.path.getmtime(prod_scaler_path)

        self.assertEqual(mtime_model_before, mtime_model_after)
        self.assertEqual(mtime_scaler_before, mtime_scaler_after)

    def test_10_missing_split_file_raises_error(self):
        """10. Missing temporal split file raises FileNotFoundError."""
        temp_dir = tempfile.mkdtemp()
        try:
            with self.assertRaises(FileNotFoundError):
                Phase8BenchmarkRunner(temp_dir)
        finally:
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    unittest.main()
