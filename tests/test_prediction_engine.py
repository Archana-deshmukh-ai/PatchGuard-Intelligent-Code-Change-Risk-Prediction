"""
Unit and Integration Test Suite for PatchGuard Risk Prediction Engine (Phase 5).
"""

import os
import sys
import json
import unittest
import tempfile
import shutil
import subprocess

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.engine import (
    RiskPredictionEngine,
    PredictionResult,
    FEATURE_COLUMNS,
    RepositoryNotFoundError,
    InvalidGitRepositoryError,
    CommitNotFoundError,
    UnsupportedCommitTypeError,
    EmptyCommitError,
    ModelArtifactNotFoundError,
    IncompatibleSchemaError
)

class TestRiskPredictionEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.repo_root = project_root
        cls.models_dir = os.path.join(project_root, 'models')
        cls.engine = RiskPredictionEngine(models_dir=cls.models_dir)

    def test_1_valid_commit_prediction(self):
        """1. End-to-end prediction on HEAD of current repository."""
        res = self.engine.predict(repo_path=self.repo_root, commit_hash="HEAD")
        
        self.assertIsInstance(res, PredictionResult)
        self.assertEqual(len(res.commit_hash), 7)
        self.assertEqual(len(res.full_hash), 40)
        self.assertGreaterEqual(res.raw_probability, 0.0)
        self.assertLessEqual(res.raw_probability, 1.0)
        self.assertEqual(res.decision_threshold, 0.35)
        self.assertIn(res.risk_level, ["LOW", "MEDIUM", "HIGH"])
        self.assertIn(res.prediction_label, ["Elevated Defect Risk", "Standard Risk Change"])
        
        # Check all 10 features present
        for col in FEATURE_COLUMNS:
            self.assertIn(col, res.features)

    def test_2_invalid_repository_path(self):
        """2. Non-existent repository path raises RepositoryNotFoundError."""
        bad_path = os.path.join(self.repo_root, "non_existent_directory_12345")
        with self.assertRaises(RepositoryNotFoundError):
            self.engine.predict(repo_path=bad_path, commit_hash="HEAD")

    def test_3_invalid_git_repository(self):
        """3. Directory that is not a git repo raises InvalidGitRepositoryError."""
        temp_dir = tempfile.mkdtemp()
        try:
            with self.assertRaises(InvalidGitRepositoryError):
                self.engine.predict(repo_path=temp_dir, commit_hash="HEAD")
        finally:
            shutil.rmtree(temp_dir)

    def test_4_invalid_commit_hash(self):
        """4. Invalid commit SHA raises CommitNotFoundError."""
        with self.assertRaises(CommitNotFoundError):
            self.engine.predict(repo_path=self.repo_root, commit_hash="invalid_sha_123456789")

    def test_5_feature_schema_canonical_ordering_mismatch(self):
        """5. Feature schema ordering check enforces exact FEATURE_COLUMNS."""
        self.assertEqual(len(FEATURE_COLUMNS), 10)
        self.assertEqual(FEATURE_COLUMNS[0], 'lines_added')
        self.assertEqual(FEATURE_COLUMNS[1], 'lines_deleted')
        self.assertEqual(FEATURE_COLUMNS[2], 'code_churn')
        self.assertEqual(FEATURE_COLUMNS[3], 'files_changed')
        self.assertEqual(FEATURE_COLUMNS[4], 'functions_changed')
        self.assertEqual(FEATURE_COLUMNS[5], 'num_directories_touched')
        self.assertEqual(FEATURE_COLUMNS[6], 'is_test_file_modified')
        self.assertEqual(FEATURE_COLUMNS[7], 'avg_lines_changed_per_file')
        self.assertEqual(FEATURE_COLUMNS[8], 'max_lines_changed_in_single_file')
        self.assertEqual(FEATURE_COLUMNS[9], 'num_source_files_changed')

    def test_6_model_and_scaler_loading_and_hashes(self):
        """6. Artifact loading produces non-empty SHA-256 hashes for model and scaler."""
        self.assertEqual(len(self.engine.model_artifact_hash), 64)
        self.assertEqual(len(self.engine.scaler_artifact_hash), 64)

    def test_7_missing_model_artifact(self):
        """7. Missing model directory raises ModelArtifactNotFoundError."""
        temp_dir = tempfile.mkdtemp()
        try:
            with self.assertRaises(ModelArtifactNotFoundError):
                RiskPredictionEngine(models_dir=temp_dir)
        finally:
            shutil.rmtree(temp_dir)

    def test_8_probability_bounds(self):
        """8. Model raw probability is strictly bounded in [0.0, 1.0]."""
        res = self.engine.predict(repo_path=self.repo_root, commit_hash="HEAD")
        self.assertTrue(0.0 <= res.raw_probability <= 1.0)

    def test_9_threshold_decision_behavior(self):
        """9. Decision threshold behavior check at high and low thresholds."""
        res_low = self.engine.predict(repo_path=self.repo_root, commit_hash="HEAD", threshold=0.00)
        self.assertTrue(res_low.is_above_threshold)
        
        res_high = self.engine.predict(repo_path=self.repo_root, commit_hash="HEAD", threshold=1.01)
        self.assertFalse(res_high.is_above_threshold)

    def test_10_deterministic_repeated_prediction(self):
        """10. Repeated predictions on the exact same commit yield identical results."""
        res1 = self.engine.predict(repo_path=self.repo_root, commit_hash="HEAD")
        res2 = self.engine.predict(repo_path=self.repo_root, commit_hash="HEAD")
        
        self.assertEqual(res1.full_hash, res2.full_hash)
        self.assertEqual(res1.raw_probability, res2.raw_probability)
        self.assertEqual(res1.is_above_threshold, res2.is_above_threshold)
        self.assertEqual(res1.risk_level, res2.risk_level)
        self.assertEqual(res1.features, res2.features)
        self.assertEqual(res1.model_signals.log_odds_contributions, res2.model_signals.log_odds_contributions)
        self.assertEqual(res1.model_artifact_hash, res2.model_artifact_hash)
        self.assertEqual(res1.scaler_artifact_hash, res2.scaler_artifact_hash)

    def test_11_prediction_result_serialization(self):
        """11. PredictionResult serializes cleanly to dict and JSON without errors."""
        res = self.engine.predict(repo_path=self.repo_root, commit_hash="HEAD")
        d = res.to_dict()
        
        self.assertIsInstance(d, dict)
        self.assertEqual(d["commit_hash"], res.commit_hash)
        
        # Verify JSON serializability
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)
        self.assertIn(res.commit_hash, json_str)

    def test_12_merge_commit_handling(self):
        """12. Attempting to predict on a merge commit raises UnsupportedCommitTypeError."""
        # Find a merge commit in local repo history if available, or test exception logic
        cmd = ["git", "log", "--merges", "-1", "--format=%H"]
        proc = subprocess.run(cmd, cwd=self.repo_root, stdout=subprocess.PIPE, text=True)
        merge_hash = proc.stdout.strip()
        
        if merge_hash:
            with self.assertRaises(UnsupportedCommitTypeError):
                self.engine.predict(repo_path=self.repo_root, commit_hash=merge_hash)
        else:
            self.skipTest("No merge commits found in target repository to test merge commit error.")

if __name__ == '__main__':
    unittest.main()
