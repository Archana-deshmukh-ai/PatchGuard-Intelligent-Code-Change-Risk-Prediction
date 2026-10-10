"""
Unit and integration test suite for PatchGuard REST API (Phase 10.2).
Tests API endpoints, schemas, authentication, path traversal guards,
readiness checks, error handling, and production governance without external network access.
"""

import os
import sys
import shutil
import tempfile
import subprocess
import unittest
from typing import List, Optional

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi.testclient import TestClient
from src.api.config import APIConfig
from src.api.service import (
    PredictionService,
    ServiceNotReadyError,
    EXPECTED_MODEL_HASH,
    EXPECTED_SCALER_HASH,
    PRODUCTION_THRESHOLD
)
from src.api.app import create_app
from src.engine import RiskPredictionEngine
from src.engine.predictor import compute_file_sha256


class TestPatchGuardAPI(unittest.TestCase):
    """Integration and unit tests for PatchGuard REST API."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = project_root
        cls.models_dir = os.path.join(project_root, 'models')
        cls.model_path = os.path.join(cls.models_dir, 'logistic_regression_real.pkl')
        cls.scaler_path = os.path.join(cls.models_dir, 'real_data_scaler.pkl')

        # Snapshot initial artifact hashes to guarantee zero mutation
        cls.initial_model_hash = compute_file_sha256(cls.model_path)
        cls.initial_scaler_hash = compute_file_sha256(cls.scaler_path)

    def setUp(self):
        # Create isolated workspace for permitted repositories
        self.workspace_dir = tempfile.mkdtemp(prefix="test_api_workspace_")
        self.allowed_repos_root = os.path.abspath(self.workspace_dir)

        # Create a test Git repository inside the workspace
        self.repo_name = "test-repo"
        self.repo_path = os.path.join(self.allowed_repos_root, self.repo_name)
        os.makedirs(self.repo_path, exist_ok=True)
        self._init_test_git_repo(self.repo_path)

        # Create APIConfig with test workspace and real production models
        self.config = APIConfig(
            host="127.0.0.1",
            port=8000,
            api_key=None,
            allowed_repos_root=self.allowed_repos_root,
            models_dir=self.models_dir,
            default_threshold=PRODUCTION_THRESHOLD
        )

        self.service = PredictionService(config=self.config)
        self.app = create_app(config=self.config, service=self.service)
        self.client = TestClient(self.app)

    def tearDown(self):
        if os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir, ignore_errors=True)

    def _run_git(self, repo_dir: str, args: List[str]) -> subprocess.CompletedProcess:
        git_executable = shutil.which("git") or "git"
        cmd = [git_executable] + args
        env = os.environ.copy()
        env["GIT_AUTHOR_NAME"] = "Test Author"
        env["GIT_AUTHOR_EMAIL"] = "test@example.com"
        env["GIT_COMMITTER_NAME"] = "Test Author"
        env["GIT_COMMITTER_EMAIL"] = "test@example.com"
        return subprocess.run(
            cmd,
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            env=env
        )

    def _init_test_git_repo(self, repo_dir: str):
        self._run_git(repo_dir, ["init", "-b", "main"])
        self._run_git(repo_dir, ["config", "user.name", "Test Author"])
        self._run_git(repo_dir, ["config", "user.email", "test@example.com"])

        # Initial commit
        file_path = os.path.join(repo_dir, "initial.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# Initial commit file\ndef initial():\n    return True\n")
        self._run_git(repo_dir, ["add", "initial.py"])
        self._run_git(repo_dir, ["commit", "-m", "Initial commit"])

        # Second commit (feature change)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write("def feature():\n    return 42\n")
        self._run_git(repo_dir, ["add", "initial.py"])
        self._run_git(repo_dir, ["commit", "-m", "Feature commit"])

    # -------------------------------------------------------------------------
    # 1. Health Endpoint Tests
    # -------------------------------------------------------------------------
    def test_health_endpoint_liveness_only(self):
        """GET /health reports process liveness without claiming model readiness."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "pass")
        self.assertEqual(data["service"], "patchguard-api")
        self.assertEqual(data["version"], "1.0.0")

        # Must not expose model readiness details or artifact hashes
        self.assertNotIn("model_id", data)
        self.assertNotIn("decision_threshold", data)
        self.assertNotIn("model_artifact_hash", data)

    # -------------------------------------------------------------------------
    # 2. Readiness Endpoint Tests
    # -------------------------------------------------------------------------
    def test_ready_endpoint_success(self):
        """GET /ready succeeds when real production model and scaler are loaded."""
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["model_id"], "logistic_regression_real")
        self.assertEqual(data["model_version"], "v1.0.0-phase4")
        self.assertEqual(data["feature_schema_version"], "v1.0-10features")
        self.assertEqual(data["decision_threshold"], 0.35)
        self.assertEqual(data["model_artifact_hash"].lower(), EXPECTED_MODEL_HASH.lower())
        self.assertEqual(data["scaler_artifact_hash"].lower(), EXPECTED_SCALER_HASH.lower())

    def test_ready_endpoint_unready_503(self):
        """GET /ready returns HTTP 503 when the engine fails initialization or readiness."""
        # Inject an unready service
        unready_config = APIConfig(
            allowed_repos_root=self.allowed_repos_root,
            models_dir="/non/existent/models/directory"
        )
        unready_service = PredictionService(config=unready_config)
        unready_app = create_app(config=unready_config, service=unready_service)
        unready_client = TestClient(unready_app)

        response = unready_client.get("/ready")
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "SERVICE_NOT_READY")

    # -------------------------------------------------------------------------
    # 3. Commit Prediction Endpoint Tests
    # -------------------------------------------------------------------------
    def test_predict_commit_success(self):
        """POST /v1/predictions/commit succeeds on valid commit with full contract payload."""
        payload = {
            "repository": self.repo_name,
            "commit_hash": "HEAD"
        }
        response = self.client.post("/v1/predictions/commit", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # 1. Commit metadata
        self.assertIn("commit", data)
        self.assertEqual(data["commit"]["repository"], self.repo_name)
        self.assertTrue(len(data["commit"]["full_hash"]) == 40)
        self.assertTrue(len(data["commit"]["commit_hash"]) >= 7)

        # 2. Prediction data & locked threshold 0.35
        self.assertIn("prediction", data)
        pred = data["prediction"]
        self.assertTrue(0.0 <= pred["raw_probability"] <= 1.0)
        self.assertEqual(pred["decision_threshold"], 0.35)
        self.assertIn(pred["prediction_label"], ["Standard Risk Change", "Elevated Defect Risk"])
        self.assertIn(pred["risk_level"], ["LOW", "MEDIUM", "HIGH"])
        self.assertIn("presentation_disclaimer", pred)

        # 3. Quantitative model signals
        self.assertIn("model_signals", data)
        sig = data["model_signals"]
        self.assertIn("top_positive_factors", sig)
        self.assertIn("top_negative_factors", sig)
        self.assertIn("log_odds_contributions", sig)

        # 4. Canonical 10 features
        self.assertIn("features", data)
        features = data["features"]
        self.assertEqual(len(features), 10)
        expected_features = [
            'lines_added', 'lines_deleted', 'code_churn', 'files_changed',
            'functions_changed', 'num_directories_touched', 'is_test_file_modified',
            'avg_lines_changed_per_file', 'max_lines_changed_in_single_file',
            'num_source_files_changed'
        ]
        for ef in expected_features:
            self.assertIn(ef, features)

        # 5. Provenance metadata
        self.assertIn("provenance", data)
        prov = data["provenance"]
        self.assertEqual(prov["model_id"], "logistic_regression_real")
        self.assertEqual(prov["model_artifact_hash"].lower(), EXPECTED_MODEL_HASH.lower())
        self.assertEqual(prov["scaler_artifact_hash"].lower(), EXPECTED_SCALER_HASH.lower())

    def test_predict_commit_client_threshold_override_rejected(self):
        """POST /v1/predictions/commit rejects client-supplied threshold override with 422."""
        payload = {
            "repository": self.repo_name,
            "commit_hash": "HEAD",
            "threshold": 0.50  # Client override forbidden in Phase 10.2
        }
        response = self.client.post("/v1/predictions/commit", json=payload)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")
        self.assertIn("extra", data["error"]["message"].lower())

    def test_predict_commit_missing_required_fields(self):
        """POST /v1/predictions/commit rejects empty or missing body with 422."""
        response = self.client.post("/v1/predictions/commit", json={})
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")

    # -------------------------------------------------------------------------
    # 4. Path Traversal & Security Boundary Tests
    # -------------------------------------------------------------------------
    def test_predict_commit_path_traversal_rejected(self):
        """Rejects directory traversal sequences with 422 or 403."""
        # Traversal with path separators violates regex schema -> 422
        bad_repos = ["../outside", "../../etc", "repo/sub", "..\\windows", "repo\\..\\outside"]
        for bad_repo in bad_repos:
            payload = {"repository": bad_repo, "commit_hash": "HEAD"}
            response = self.client.post("/v1/predictions/commit", json=payload)
            self.assertIn(response.status_code, [403, 422])
            data = response.json()
            self.assertIn("error", data)

    def test_predict_commit_repository_not_found(self):
        """POST /v1/predictions/commit returns 404 when repository does not exist."""
        payload = {
            "repository": "nonexistent-repo",
            "commit_hash": "HEAD"
        }
        response = self.client.post("/v1/predictions/commit", json=payload)
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "REPOSITORY_NOT_FOUND")

    def test_predict_commit_commit_not_found(self):
        """POST /v1/predictions/commit returns 404 when commit reference cannot be resolved."""
        payload = {
            "repository": self.repo_name,
            "commit_hash": "0123456789abcdef0123456789abcdef01234567"
        }
        response = self.client.post("/v1/predictions/commit", json=payload)
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "COMMIT_NOT_FOUND")

    # -------------------------------------------------------------------------
    # 5. Unsupported Commit Semantics Tests
    # -------------------------------------------------------------------------
    def test_predict_commit_empty_commit_returns_400(self):
        """POST /v1/predictions/commit returns 400 EMPTY_COMMIT on zero-file changes."""
        self._run_git(self.repo_path, ["commit", "--allow-empty", "-m", "Empty commit"])
        payload = {
            "repository": self.repo_name,
            "commit_hash": "HEAD"
        }
        response = self.client.post("/v1/predictions/commit", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "EMPTY_COMMIT")

    def test_predict_commit_merge_commit_returns_400(self):
        """POST /v1/predictions/commit returns 400 UNSUPPORTED_COMMIT_TYPE on merge commits."""
        # Create a branch and a commit
        self._run_git(self.repo_path, ["checkout", "-b", "feature-branch"])
        branch_file = os.path.join(self.repo_path, "feature.py")
        with open(branch_file, "w", encoding="utf-8") as f:
            f.write("def feature_branch(): pass\n")
        self._run_git(self.repo_path, ["add", "feature.py"])
        self._run_git(self.repo_path, ["commit", "-m", "Branch feature"])

        # Switch back to main, add another commit, and merge
        self._run_git(self.repo_path, ["checkout", "main"])
        main_file = os.path.join(self.repo_path, "main_diverge.py")
        with open(main_file, "w", encoding="utf-8") as f:
            f.write("def main_diverge(): pass\n")
        self._run_git(self.repo_path, ["add", "main_diverge.py"])
        self._run_git(self.repo_path, ["commit", "-m", "Main diverge"])

        self._run_git(self.repo_path, ["merge", "--no-ff", "feature-branch", "-m", "Merge commit"])

        payload = {
            "repository": self.repo_name,
            "commit_hash": "HEAD"
        }
        response = self.client.post("/v1/predictions/commit", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "UNSUPPORTED_COMMIT_TYPE")

    # -------------------------------------------------------------------------
    # 6. Authentication Tests
    # -------------------------------------------------------------------------
    def test_authentication_workflow(self):
        """Verifies API key enforcement, constant-time validation, and 401 handling."""
        auth_config = APIConfig(
            api_key="secret-key-12345",
            allowed_repos_root=self.allowed_repos_root,
            models_dir=self.models_dir
        )
        auth_app = create_app(config=auth_config, service=self.service)
        auth_client = TestClient(auth_app)

        # 1. Health and Ready probes remain accessible without auth
        health_resp = auth_client.get("/health")
        self.assertEqual(health_resp.status_code, 200)

        ready_resp = auth_client.get("/ready")
        self.assertEqual(ready_resp.status_code, 200)

        # 2. Prediction endpoint without header returns 401
        payload = {"repository": self.repo_name, "commit_hash": "HEAD"}
        unauth_resp = auth_client.post("/v1/predictions/commit", json=payload)
        self.assertEqual(unauth_resp.status_code, 401)
        self.assertEqual(unauth_resp.json()["error"]["code"], "AUTHENTICATION_REQUIRED")

        # 3. Prediction endpoint with invalid header returns 401
        wrong_auth_resp = auth_client.post(
            "/v1/predictions/commit",
            json=payload,
            headers={"X-API-Key": "wrong-secret"}
        )
        self.assertEqual(wrong_auth_resp.status_code, 401)
        self.assertEqual(wrong_auth_resp.json()["error"]["code"], "AUTHENTICATION_REQUIRED")

        # 4. Prediction endpoint with valid header succeeds 200
        valid_auth_resp = auth_client.post(
            "/v1/predictions/commit",
            json=payload,
            headers={"X-API-Key": "secret-key-12345"}
        )
        self.assertEqual(valid_auth_resp.status_code, 200)
        self.assertEqual(valid_auth_resp.json()["commit"]["repository"], self.repo_name)

    def test_predict_commit_relative_dot_reference_rejected(self):
        """POST /v1/predictions/commit rejects '.' reference as repository identifier."""
        payload = {
            "repository": ".",
            "commit_hash": "HEAD"
        }
        response = self.client.post("/v1/predictions/commit", json=payload)
        self.assertEqual(response.status_code, 403)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "FORBIDDEN_REPOSITORY_ACCESS")

    def test_predict_commit_symlink_escape_rejected(self):
        """Rejects repository symlinks pointing outside permitted root."""
        # Create an outside directory
        outside_dir = tempfile.mkdtemp(prefix="outside_repo_")
        try:
            symlink_name = "escaped-symlink"
            symlink_path = os.path.join(self.allowed_repos_root, symlink_name)
            try:
                os.symlink(outside_dir, symlink_path)
                can_symlink = True
            except (OSError, NotImplementedError):
                can_symlink = False

            if can_symlink:
                payload = {"repository": symlink_name, "commit_hash": "HEAD"}
                response = self.client.post("/v1/predictions/commit", json=payload)
                self.assertEqual(response.status_code, 403)
                self.assertEqual(response.json()["error"]["code"], "FORBIDDEN_REPOSITORY_ACCESS")
            else:
                # If OS does not allow unprivileged symlinks, verify service logic with mock
                from unittest.mock import patch
                from src.api.service import ForbiddenRepositoryAccessError
                with patch("os.path.realpath") as mock_realpath:
                    # Return path outside canonical root
                    mock_realpath.side_effect = lambda p: "/outside/escaped/path" if symlink_name in p else p
                    with self.assertRaises(ForbiddenRepositoryAccessError):
                        self.service.resolve_repository_path(symlink_name)
        finally:
            shutil.rmtree(outside_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 7. Sanitized Error Presentation & Artifact Protection
    # -------------------------------------------------------------------------
    def test_sanitized_error_suppresses_internal_paths(self):
        """Verifies that error messages suppress absolute local filesystem paths."""
        payload = {
            "repository": "nonexistent-repo",
            "commit_hash": "HEAD"
        }
        response = self.client.post("/v1/predictions/commit", json=payload)
        self.assertEqual(response.status_code, 404)
        error_msg = response.json()["error"]["message"]
        # Ensure raw drive letters like C:\ or Unix paths like /var/ are not leaked
        self.assertNotIn("C:\\", error_msg)
        self.assertNotIn("C:/", error_msg)
        self.assertNotIn("/repositories/", error_msg)

    def test_sanitize_message_utility_removes_container_paths(self):
        """Verifies that _sanitize_message filters Linux and container paths."""
        from src.api.error_handlers import _sanitize_message
        raw_msg = "Failed at /repositories/bottle/src or /app/models/model.pkl or C:\\Projects\\repo"
        sanitized = _sanitize_message(raw_msg)
        self.assertNotIn("/repositories/", sanitized)
        self.assertNotIn("/app/", sanitized)
        self.assertNotIn("C:\\", sanitized)
        self.assertIn("[protected-path]", sanitized)

    def test_production_artifacts_unmodified_after_test_suite(self):
        """Verifies production model and scaler artifact SHA-256 hashes are 100% identical."""
        current_model_hash = compute_file_sha256(self.model_path)
        current_scaler_hash = compute_file_sha256(self.scaler_path)

        self.assertEqual(current_model_hash, self.initial_model_hash)
        self.assertEqual(current_scaler_hash, self.initial_scaler_hash)
        self.assertEqual(current_model_hash.lower(), EXPECTED_MODEL_HASH.lower())
        self.assertEqual(current_scaler_hash.lower(), EXPECTED_SCALER_HASH.lower())


if __name__ == '__main__':
    unittest.main()
