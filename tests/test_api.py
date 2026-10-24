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

        # Feature single branch (1 commit ahead of main)
        self._run_git(repo_dir, ["checkout", "-b", "feature/single"])
        with open(file_path, "a", encoding="utf-8") as f:
            f.write("def single_helper():\n    return 'single'\n")
        self._run_git(repo_dir, ["add", "initial.py"])
        self._run_git(repo_dir, ["commit", "-m", "Single PR commit"])

        # Feature multi branch (2 commits ahead of main)
        self._run_git(repo_dir, ["checkout", "main"])
        self._run_git(repo_dir, ["checkout", "-b", "feature/multi"])
        utils_path = os.path.join(repo_dir, "utils.py")
        with open(utils_path, "w", encoding="utf-8") as f:
            f.write("def util():\n    return 100\n")
        self._run_git(repo_dir, ["add", "utils.py"])
        self._run_git(repo_dir, ["commit", "-m", "Multi commit 1"])

        with open(file_path, "a", encoding="utf-8") as f:
            f.write("from utils import util\n")
        self._run_git(repo_dir, ["add", "initial.py"])
        self._run_git(repo_dir, ["commit", "-m", "Multi commit 2"])

        # Return to main
        self._run_git(repo_dir, ["checkout", "main"])


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

    # -------------------------------------------------------------------------
    # 8. Operational Readiness & Deployment Hardening (Phase 10.5)
    # -------------------------------------------------------------------------
    def test_ready_endpoint_missing_repo_root_503(self):
        """GET /ready returns HTTP 503 when configured repository root does not exist."""
        nonexistent_root = os.path.join(self.workspace_dir, "nonexistent_subfolder")
        cfg = APIConfig(
            allowed_repos_root=nonexistent_root,
            models_dir=self.models_dir
        )
        svc = PredictionService(config=cfg)
        test_app = create_app(config=cfg, service=svc)
        test_client = TestClient(test_app)

        response = test_client.get("/ready")
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["error"]["code"], "SERVICE_NOT_READY")

    def test_500_engine_error_logging_redaction(self):
        """Verifies that 500 PatchGuardError handler logs sanitized diagnostics with tracebacks."""
        from unittest.mock import patch
        from src.engine.exceptions import PatchGuardError

        dummy_token = "ghp_" + ("1" * 36)
        with patch.object(
            self.service,
            "predict_commit",
            side_effect=PatchGuardError(f"Internal failure in C:\\private\\data.py with {dummy_token}")
        ):
            with self.assertLogs("patchguard.api", level="ERROR") as log_capture:
                response = self.client.post(
                    "/v1/predictions/commit",
                    json={"repository": self.repo_name, "commit_hash": "HEAD"}
                )

            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertEqual(data["error"]["code"], "INTERNAL_ENGINE_ERROR")
            # Ensure paths and secrets were redacted in response and log
            self.assertNotIn("C:\\private", data["error"]["message"])
            self.assertNotIn("ghp_", data["error"]["message"])
            self.assertIn("[protected-path]", data["error"]["message"])
            self.assertIn("[protected-token]", data["error"]["message"])

            log_output = "\n".join(log_capture.output)
            self.assertIn("Internal engine error [PatchGuardError]:", log_output)
            self.assertNotIn("C:\\private", log_output)
            self.assertNotIn("ghp_", log_output)

    def test_500_unhandled_exception_logging_redaction(self):
        """Verifies that unhandled server exceptions log tracebacks and return sanitized error."""
        from unittest.mock import patch

        unhandled_client = TestClient(self.app, raise_server_exceptions=False)
        with patch.object(
            self.service,
            "predict_commit",
            side_effect=RuntimeError("Unexpected runtime crash in /app/secret_code.py")
        ):
            with self.assertLogs("patchguard.api", level="ERROR") as log_capture:
                response = unhandled_client.post(
                    "/v1/predictions/commit",
                    json={"repository": self.repo_name, "commit_hash": "HEAD"}
                )

            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertEqual(data["error"]["code"], "INTERNAL_SERVER_ERROR")

            log_output = "\n".join(log_capture.output)
            self.assertIn("Unhandled server exception [RuntimeError]:", log_output)
            self.assertNotIn("/app/secret_code.py", log_output)
            self.assertIn("[protected-path]", log_output)

    def test_git_subprocess_timeout_expired_handling(self):
        """Verifies that GitRepositoryExtractor raises GitExtractionError on subprocess timeout."""
        from src.git_extractor import GitRepositoryExtractor
        from src.engine.exceptions import GitExtractionError

        extractor = GitRepositoryExtractor(self.repo_path)
        with self.assertRaises(GitExtractionError) as ctx:
            extractor._run_git(["status"], timeout=0.0000001)

        err_msg = str(ctx.exception)
        self.assertIn("Git command timed out after", err_msg)
        self.assertIn("git status", err_msg)

    def test_startup_security_warning_on_wildcard_host_without_key(self):
        """Verifies actionable security warning when binding 0.0.0.0 without API key."""
        wildcard_cfg = APIConfig(
            host="0.0.0.0",
            port=8000,
            api_key=None,
            allowed_repos_root=self.allowed_repos_root,
            models_dir=self.models_dir
        )
        with self.assertLogs("patchguard.api", level="WARNING") as log_capture:
            app_wildcard = create_app(config=wildcard_cfg)
            with TestClient(app_wildcard):
                pass

        log_output = "\n".join(log_capture.output)
        self.assertIn("SECURITY WARNING: Server is bound to all network interfaces (0.0.0.0)", log_output)

    def test_predict_commit_observability_logging(self):
        """Verifies that predict_commit logs performance duration and non-sensitive metadata."""
        with self.assertLogs("patchguard.api", level="INFO") as log_capture:
            response = self.client.post(
                "/v1/predictions/commit",
                json={"repository": self.repo_name, "commit_hash": "HEAD"}
            )

        self.assertEqual(response.status_code, 200)
        log_output = "\n".join(log_capture.output)
        self.assertIn("Prediction completed: repo=test-repo", log_output)
        self.assertIn("risk_level=", log_output)
        self.assertIn("score=", log_output)
        self.assertIn("duration_ms=", log_output)

    # -------------------------------------------------------------------------
    # 9. Pull Request Prediction Endpoint Tests (Phase 11.1A)
    # -------------------------------------------------------------------------
    def test_pr_prediction_single_commit_success(self):
        """POST /v1/predictions/pr returns 200 OK for valid single-commit PR."""
        response = self.client.post(
            "/v1/predictions/pr",
            json={
                "repository": self.repo_name,
                "base_ref": "main",
                "head_ref": "feature/single"
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # PR metadata
        self.assertEqual(data["pr"]["repository"], self.repo_name)
        self.assertEqual(data["pr"]["base_ref"], "main")
        self.assertEqual(data["pr"]["head_ref"], "feature/single")
        self.assertFalse(data["pr"]["is_empty"])
        self.assertEqual(data["pr"]["commit_count"], 1)
        self.assertEqual(len(data["pr"]["resolved_base_sha"]), 40)
        self.assertEqual(len(data["pr"]["resolved_head_sha"]), 40)
        self.assertEqual(len(data["pr"]["resolved_merge_base_sha"]), 40)

        # Commits list
        self.assertEqual(len(data["commits"]), 1)
        c = data["commits"][0]
        self.assertEqual(c["commit_hash"], c["full_hash"][:7])
        self.assertFalse(c["is_merge_commit"])
        self.assertIsNone(c["error"])
        self.assertIsNotNone(c["prediction"])
        self.assertTrue(0.0 <= c["prediction"]["raw_probability"] <= 1.0)
        self.assertEqual(c["prediction"]["decision_threshold"], 0.35)
        self.assertIn(c["prediction"]["risk_level"], ["LOW", "MEDIUM", "HIGH"])
        self.assertIsNotNone(c["model_signals"])
        self.assertEqual(len(c["features"]), 10)

        # Summary statistics
        summary = data["summary"]
        self.assertEqual(summary["total_commits"], 1)
        self.assertEqual(summary["analyzed_commits"], 1)
        self.assertEqual(summary["merge_commits"], 0)
        self.assertEqual(summary["skipped_commits"], 0)
        self.assertEqual(summary["decision_threshold"], 0.35)
        self.assertEqual(summary["max_commit_risk_score"], c["prediction"]["raw_probability"])

        # Provenance and governance disclaimer
        self.assertEqual(data["provenance"]["model_id"], "logistic_regression_real")
        self.assertIn("does NOT generate a fabricated PR-wide risk probability score", data["analysis_disclaimer"])

    def test_pr_prediction_multi_commit_success(self):
        """POST /v1/predictions/pr returns 200 OK for valid multi-commit PR with ordered commits."""
        response = self.client.post(
            "/v1/predictions/pr",
            json={
                "repository": self.repo_name,
                "base_ref": "main",
                "head_ref": "feature/multi"
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["pr"]["commit_count"], 2)
        self.assertEqual(len(data["commits"]), 2)
        self.assertFalse(data["pr"]["is_empty"])

        summary = data["summary"]
        self.assertEqual(summary["total_commits"], 2)
        self.assertEqual(summary["analyzed_commits"], 2)
        expected_max = max(c["prediction"]["raw_probability"] for c in data["commits"])
        self.assertEqual(summary["max_commit_risk_score"], expected_max)

    def test_pr_prediction_empty_pr_same_ref(self):
        """POST /v1/predictions/pr returns 200 OK with is_empty=True when base_ref == head_ref."""
        response = self.client.post(
            "/v1/predictions/pr",
            json={
                "repository": self.repo_name,
                "base_ref": "main",
                "head_ref": "main"
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["pr"]["is_empty"])
        self.assertEqual(data["pr"]["commit_count"], 0)
        self.assertEqual(len(data["commits"]), 0)
        self.assertEqual(data["summary"]["total_commits"], 0)
        self.assertEqual(data["summary"]["analyzed_commits"], 0)
        self.assertIsNone(data["summary"]["max_commit_risk_score"])

    def test_pr_prediction_merge_commit_handled(self):
        """POST /v1/predictions/pr gracefully handles and tallies merge commits within PR."""
        # Create a branch with a merge commit
        self._run_git(self.repo_path, ["checkout", "main"])
        self._run_git(self.repo_path, ["checkout", "-b", "branch-a"])
        file_a = os.path.join(self.repo_path, "file_a.py")
        with open(file_a, "w", encoding="utf-8") as f:
            f.write("def fa(): pass\n")
        self._run_git(self.repo_path, ["add", "file_a.py"])
        self._run_git(self.repo_path, ["commit", "-m", "Commit on branch-a"])

        self._run_git(self.repo_path, ["checkout", "main"])
        self._run_git(self.repo_path, ["checkout", "-b", "branch-b"])
        file_b = os.path.join(self.repo_path, "file_b.py")
        with open(file_b, "w", encoding="utf-8") as f:
            f.write("def fb(): pass\n")
        self._run_git(self.repo_path, ["add", "file_b.py"])
        self._run_git(self.repo_path, ["commit", "-m", "Commit on branch-b"])

        # Merge branch-a into branch-b
        self._run_git(self.repo_path, ["merge", "branch-a", "--no-ff", "-m", "Merge branch-a into branch-b"])

        response = self.client.post(
            "/v1/predictions/pr",
            json={
                "repository": self.repo_name,
                "base_ref": "main",
                "head_ref": "branch-b"
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertGreaterEqual(data["summary"]["merge_commits"], 1)
        merge_commits = [c for c in data["commits"] if c["is_merge_commit"]]
        self.assertTrue(len(merge_commits) >= 1)
        self.assertIsNone(merge_commits[0]["prediction"])
        self.assertIn("merge commit", merge_commits[0]["error"].lower())

        # Cleanup
        self._run_git(self.repo_path, ["checkout", "main"])

    def test_pr_prediction_invalid_ref_leading_hyphen(self):
        """POST /v1/predictions/pr rejects option-like references starting with hyphen."""
        # Test leading hyphen on base_ref
        response = self.client.post(
            "/v1/predictions/pr",
            json={
                "repository": self.repo_name,
                "base_ref": "--upload-pack",
                "head_ref": "main"
            }
        )
        self.assertIn(response.status_code, [400, 422])
        err = response.json()["error"]
        self.assertIn(err["code"], ["INVALID_PR_REF", "VALIDATION_ERROR"])

        # Test leading hyphen on head_ref
        response2 = self.client.post(
            "/v1/predictions/pr",
            json={
                "repository": self.repo_name,
                "base_ref": "main",
                "head_ref": "-o"
            }
        )
        self.assertIn(response2.status_code, [400, 422])

    def test_pr_prediction_nonexistent_ref_400(self):
        """POST /v1/predictions/pr returns HTTP 400 INVALID_PR_REF for unresolvable branch."""
        response = self.client.post(
            "/v1/predictions/pr",
            json={
                "repository": self.repo_name,
                "base_ref": "main",
                "head_ref": "nonexistent-branch-404"
            }
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["error"]["code"], "INVALID_PR_REF")
        self.assertIn("Invalid head reference", data["error"]["message"])

    def test_pr_prediction_ambiguous_ref_400(self):
        """POST /v1/predictions/pr returns HTTP 400 INVALID_PR_REF when reference matches multiple Git objects."""
        # Create a branch and tag with the same name
        self._run_git(self.repo_path, ["branch", "ambig-ref-test"])
        self._run_git(self.repo_path, ["tag", "ambig-ref-test"])
        try:
            response = self.client.post(
                "/v1/predictions/pr",
                json={
                    "repository": self.repo_name,
                    "base_ref": "main",
                    "head_ref": "ambig-ref-test"
                }
            )
            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertEqual(data["error"]["code"], "INVALID_PR_REF")
            self.assertIn("Ambiguous", data["error"]["message"])
        finally:
            self._run_git(self.repo_path, ["branch", "-D", "ambig-ref-test"])
            self._run_git(self.repo_path, ["tag", "-d", "ambig-ref-test"])

    def test_pr_prediction_unrelated_histories_no_merge_base_400(self):
        """POST /v1/predictions/pr returns HTTP 400 INVALID_PR_REF when branches share no common ancestor."""
        # Create an orphan branch with independent history
        self._run_git(self.repo_path, ["checkout", "--orphan", "orphan-test"])
        orphan_file = os.path.join(self.repo_path, "orphan.txt")
        with open(orphan_file, "w", encoding="utf-8") as f:
            f.write("orphan\n")
        self._run_git(self.repo_path, ["add", "orphan.txt"])
        self._run_git(self.repo_path, ["commit", "-m", "Orphan initial commit"])
        self._run_git(self.repo_path, ["checkout", "main"])

        response = self.client.post(
            "/v1/predictions/pr",
            json={
                "repository": self.repo_name,
                "base_ref": "main",
                "head_ref": "orphan-test"
            }
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["error"]["code"], "INVALID_PR_REF")
        self.assertIn("merge-base", data["error"]["message"].lower())

    def test_pr_prediction_workload_limit_exceeded_400(self):
        """POST /v1/predictions/pr returns HTTP 400 PR_WORKLOAD_LIMIT_EXCEEDED when commit count exceeds limit."""
        limited_cfg = APIConfig(
            host="127.0.0.1",
            port=8000,
            api_key=None,
            allowed_repos_root=self.allowed_repos_root,
            models_dir=self.models_dir,
            max_pr_commits=1  # Only allow 1 commit
        )
        limited_svc = PredictionService(config=limited_cfg)
        limited_app = create_app(config=limited_cfg, service=limited_svc)
        limited_client = TestClient(limited_app)

        # feature/multi has 2 commits
        response = limited_client.post(
            "/v1/predictions/pr",
            json={
                "repository": self.repo_name,
                "base_ref": "main",
                "head_ref": "feature/multi"
            }
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["error"]["code"], "PR_WORKLOAD_LIMIT_EXCEEDED")
        self.assertIn("exceeds the maximum allowable limit of 1 commits", data["error"]["message"])

    def test_pr_prediction_overall_timeout_handling_504(self):
        """POST /v1/predictions/pr returns HTTP 504 PR_ANALYSIS_TIMEOUT when execution exceeds budget."""
        timeout_cfg = APIConfig(
            host="127.0.0.1",
            port=8000,
            api_key=None,
            allowed_repos_root=self.allowed_repos_root,
            models_dir=self.models_dir,
            pr_timeout_seconds=0.000001  # Immediate timeout
        )
        timeout_svc = PredictionService(config=timeout_cfg)
        timeout_app = create_app(config=timeout_cfg, service=timeout_svc)
        timeout_client = TestClient(timeout_app)

        response = timeout_client.post(
            "/v1/predictions/pr",
            json={
                "repository": self.repo_name,
                "base_ref": "main",
                "head_ref": "feature/multi"
            }
        )
        self.assertEqual(response.status_code, 504)
        data = response.json()
        self.assertEqual(data["error"]["code"], "PR_ANALYSIS_TIMEOUT")
        self.assertIn("exceeded the maximum execution time budget", data["error"]["message"])

    def test_pr_analyzer_git_subprocess_timeout_raises_pr_analysis_timeout_error(self):
        """PRAnalyzer._run_git raises PRAnalysisTimeoutError on subprocess timeout."""
        from src.analysis.pr_analyzer import PRAnalyzer
        from src.analysis.exceptions import PRAnalysisTimeoutError
        from unittest.mock import patch
        import subprocess

        analyzer = PRAnalyzer()
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["git", "status"], timeout=30.0)):
            with self.assertRaises(PRAnalysisTimeoutError) as ctx:
                analyzer._run_git(self.repo_path, ["status"], timeout=30.0)
            self.assertIn("Git command timed out after 30.0s: git status", str(ctx.exception))

    def test_pr_prediction_git_subprocess_timeout_maps_to_504(self):
        """POST /v1/predictions/pr maps PRAnalyzer Git subprocess timeout to HTTP 504 PR_ANALYSIS_TIMEOUT."""
        from unittest.mock import patch
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["git", "rev-parse"], timeout=30.0)):
            response = self.client.post(
                "/v1/predictions/pr",
                json={
                    "repository": self.repo_name,
                    "base_ref": "main",
                    "head_ref": "feature/single"
                }
            )
            self.assertEqual(response.status_code, 504)
            data = response.json()
            self.assertEqual(data["error"]["code"], "PR_ANALYSIS_TIMEOUT")
            self.assertIn("Git command timed out after 30.0s", data["error"]["message"])

    def test_pr_prediction_zero_external_llm_calls(self):
        """POST /v1/predictions/pr runs 100% offline without invoking external LLMs or requiring OPENAI_API_KEY."""
        # Ensure OPENAI_API_KEY is not set
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            response = self.client.post(
                "/v1/predictions/pr",
                json={
                    "repository": self.repo_name,
                    "base_ref": "main",
                    "head_ref": "feature/single"
                }
            )
            self.assertEqual(response.status_code, 200)
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_pr_prediction_client_threshold_override_rejected_422(self):
        """POST /v1/predictions/pr strictly rejects client-supplied threshold override with HTTP 422."""
        response = self.client.post(
            "/v1/predictions/pr",
            json={
                "repository": self.repo_name,
                "base_ref": "main",
                "head_ref": "feature/single",
                "threshold": 0.50
            }
        )
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")
        self.assertIn("extra", data["error"]["message"].lower())

    def test_pr_prediction_path_traversal_rejection_403(self):
        """POST /v1/predictions/pr rejects path traversal attempts with HTTP 403 or 422."""
        # Relative traversal
        response = self.client.post(
            "/v1/predictions/pr",
            json={"repository": "../outside", "base_ref": "main", "head_ref": "main"}
        )
        self.assertIn(response.status_code, [403, 422])

        # Dot references
        response2 = self.client.post(
            "/v1/predictions/pr",
            json={"repository": "..", "base_ref": "main", "head_ref": "main"}
        )
        self.assertEqual(response2.status_code, 403)
        self.assertEqual(response2.json()["error"]["code"], "FORBIDDEN_REPOSITORY_ACCESS")

    def test_pr_prediction_symlink_escape_rejection(self):
        """Symlink pointing outside allowed repository workspace is rejected with 403."""
        outside_dir = tempfile.mkdtemp(prefix="test_outside_")
        symlink_path = os.path.join(self.allowed_repos_root, "symlink-repo")
        try:
            try:
                os.symlink(outside_dir, symlink_path, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("Symlinks not supported in current environment without elevated privileges.")

            response = self.client.post(
                "/v1/predictions/pr",
                json={
                    "repository": "symlink-repo",
                    "base_ref": "main",
                    "head_ref": "main"
                }
            )
            self.assertEqual(response.status_code, 403)
            data = response.json()
            self.assertEqual(data["error"]["code"], "FORBIDDEN_REPOSITORY_ACCESS")
        finally:
            if os.path.exists(symlink_path):
                try:
                    os.unlink(symlink_path)
                except OSError:
                    pass
            if os.path.exists(outside_dir):
                shutil.rmtree(outside_dir, ignore_errors=True)

    def test_pr_prediction_authentication_enforced(self):
        """POST /v1/predictions/pr enforces API key authentication when configured."""
        auth_config = APIConfig(
            host="127.0.0.1",
            port=8000,
            api_key="secret-api-key-xyz",
            allowed_repos_root=self.allowed_repos_root,
            models_dir=self.models_dir
        )
        auth_svc = PredictionService(config=auth_config)
        auth_app = create_app(config=auth_config, service=auth_svc)
        auth_client = TestClient(auth_app)

        payload = {"repository": self.repo_name, "base_ref": "main", "head_ref": "feature/single"}

        # Missing API Key -> 401
        res_no_key = auth_client.post("/v1/predictions/pr", json=payload)
        self.assertEqual(res_no_key.status_code, 401)
        self.assertEqual(res_no_key.json()["error"]["code"], "AUTHENTICATION_REQUIRED")

        # Invalid API Key -> 401
        res_bad_key = auth_client.post(
            "/v1/predictions/pr",
            json=payload,
            headers={"X-API-Key": "wrong-key"}
        )
        self.assertEqual(res_bad_key.status_code, 401)

        # Valid API Key -> 200
        res_valid = auth_client.post(
            "/v1/predictions/pr",
            json=payload,
            headers={"X-API-Key": "secret-api-key-xyz"}
        )
        self.assertEqual(res_valid.status_code, 200)

    def test_pr_prediction_nonexistent_repository_404(self):
        """POST /v1/predictions/pr returns HTTP 404 REPOSITORY_NOT_FOUND when repo does not exist."""
        response = self.client.post(
            "/v1/predictions/pr",
            json={
                "repository": "nonexistent-repo",
                "base_ref": "main",
                "head_ref": "main"
            }
        )
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertEqual(data["error"]["code"], "REPOSITORY_NOT_FOUND")

    def test_pr_prediction_service_not_ready_503(self):
        """POST /v1/predictions/pr returns HTTP 503 SERVICE_NOT_READY when prediction engine is unready."""
        bad_config = APIConfig(
            host="127.0.0.1",
            port=8000,
            allowed_repos_root=self.allowed_repos_root,
            models_dir=os.path.join(self.workspace_dir, "nonexistent-models")
        )
        unready_svc = PredictionService(config=bad_config)
        unready_app = create_app(config=bad_config, service=unready_svc)
        unready_client = TestClient(unready_app)

        response = unready_client.post(
            "/v1/predictions/pr",
            json={
                "repository": self.repo_name,
                "base_ref": "main",
                "head_ref": "feature/single"
            }
        )
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["error"]["code"], "SERVICE_NOT_READY")

    def test_pr_prediction_observability_logging(self):
        """POST /v1/predictions/pr logs operational execution duration and non-sensitive metrics."""
        with self.assertLogs("patchguard.api", level="INFO") as log_capture:
            response = self.client.post(
                "/v1/predictions/pr",
                json={
                    "repository": self.repo_name,
                    "base_ref": "main",
                    "head_ref": "feature/single"
                }
            )

        self.assertEqual(response.status_code, 200)
        log_output = "\n".join(log_capture.output)
        self.assertIn("PR prediction completed: repo=test-repo", log_output)
        self.assertIn("base=main", log_output)
        self.assertIn("head=feature/single", log_output)
        self.assertIn("commits=1", log_output)
        self.assertIn("duration_ms=", log_output)

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
