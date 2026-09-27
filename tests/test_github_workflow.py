"""
Offline unit tests for PatchGuard Phase 7.3 GitHub Webhook Workflow & Reporting.
Validates HMAC-SHA256 signature verification, event parsing/filtering, workflow orchestration,
PR comment formatting/idempotency, and credential hygiene without external network calls.
"""

import unittest
import hmac
import hashlib
import json
import tempfile
import os
import shutil
from typing import Dict, Any, List

from src.github.schema import GitHubPullRequest
from src.github.webhook import (
    verify_webhook_signature,
    WebhookEventHandler,
    WebhookEventData
)
from src.github.reporter import GitHubPRReporter, IDEMPOTENCY_MARKER
from src.github.workflow import GitHubPRWorkflow
from src.github.exceptions import (
    GitHubIntegrationError,
    WebhookAuthenticationError,
    WebhookValidationError,
    UnsupportedGitHubEvent,
    PRAnalysisWorkflowError,
    GitHubReportingError
)

from src.analysis.pr_schema import PRAnalysisResult, PRCommitPrediction
from src.analysis.schema import LLMAnalysisResult, RiskFactor, ReviewAction
from src.engine.schema import PredictionResult, ModelSignals


class TestGitHubWebhookSignature(unittest.TestCase):
    """Test suite for HMAC-SHA256 signature verification."""
    def setUp(self):
        self.secret = "test_webhook_secret_key_123"
        self.payload = json.dumps({"action": "opened", "number": 1}).encode("utf-8")

    def _compute_sig(self, payload_bytes: bytes, secret: str) -> str:
        sig = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
        return f"sha256={sig}"

    def test_valid_signature(self):
        sig_header = self._compute_sig(self.payload, self.secret)
        self.assertTrue(verify_webhook_signature(self.payload, sig_header, self.secret))

    def test_invalid_signature(self):
        sig_header = "sha256=" + "0" * 64
        self.assertFalse(verify_webhook_signature(self.payload, sig_header, self.secret))

    def test_missing_signature(self):
        self.assertFalse(verify_webhook_signature(self.payload, None, self.secret))
        self.assertFalse(verify_webhook_signature(self.payload, "", self.secret))

    def test_missing_secret(self):
        sig_header = self._compute_sig(self.payload, self.secret)
        self.assertFalse(verify_webhook_signature(self.payload, sig_header, ""))


class TestWebhookEventHandler(unittest.TestCase):
    """Test suite for event parsing and filtering."""
    def setUp(self):
        self.secret = "my_secret_token"
        self.valid_payload_dict = {
            "action": "opened",
            "number": 42,
            "pull_request": {
                "number": 42,
                "title": "Fix critical bug in core module",
                "html_url": "https://github.com/bottlepy/bottle/pull/42",
                "state": "open",
                "base": {
                    "ref": "main",
                    "sha": "d94d341c6f10d94d341c6f10d94d341c6f10d94d",
                    "repo": {
                        "name": "bottle",
                        "owner": {"login": "bottlepy"}
                    }
                },
                "head": {
                    "ref": "feature/fix",
                    "sha": "340a95467927340a95467927340a95467927340a"
                }
            }
        }
        self.valid_payload_bytes = json.dumps(self.valid_payload_dict).encode("utf-8")
        sig = hmac.new(self.secret.encode("utf-8"), self.valid_payload_bytes, hashlib.sha256).hexdigest()
        self.valid_headers = {
            "X-Hub-Signature-256": f"sha256={sig}",
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "delivery-uuid-12345"
        }

    def test_signature_checked_before_json_parsing(self):
        """Verify that signature check fails BEFORE attempting JSON parsing of invalid payload."""
        invalid_json_bytes = b"{ invalid json content"
        sig = hmac.new(self.secret.encode("utf-8"), invalid_json_bytes, hashlib.sha256).hexdigest()
        bad_sig_headers = {
            "X-Hub-Signature-256": "sha256=" + "f" * 64,
            "X-GitHub-Event": "pull_request"
        }
        # With wrong signature, must raise WebhookAuthenticationError (not WebhookValidationError)
        with self.assertRaises(WebhookAuthenticationError):
            WebhookEventHandler.parse_event(invalid_json_bytes, bad_sig_headers, webhook_secret=self.secret)

    def test_supported_pr_actions(self):
        """Verify opened, synchronize, and reopened actions are supported."""
        for act in ["opened", "synchronize", "reopened"]:
            data = dict(self.valid_payload_dict)
            data["action"] = act
            p_bytes = json.dumps(data).encode("utf-8")
            sig = hmac.new(self.secret.encode("utf-8"), p_bytes, hashlib.sha256).hexdigest()
            hdrs = dict(self.valid_headers)
            hdrs["X-Hub-Signature-256"] = f"sha256={sig}"

            parsed = WebhookEventHandler.parse_event(p_bytes, hdrs, webhook_secret=self.secret)
            self.assertEqual(parsed.action, act)
            self.assertEqual(parsed.pull_request.number, 42)
            self.assertEqual(parsed.delivery_id, "delivery-uuid-12345")

    def test_ignored_pr_action(self):
        """Verify actions like 'closed' or 'labeled' raise UnsupportedGitHubEvent."""
        data = dict(self.valid_payload_dict)
        data["action"] = "closed"
        p_bytes = json.dumps(data).encode("utf-8")
        sig = hmac.new(self.secret.encode("utf-8"), p_bytes, hashlib.sha256).hexdigest()
        hdrs = dict(self.valid_headers)
        hdrs["X-Hub-Signature-256"] = f"sha256={sig}"

        with self.assertRaises(UnsupportedGitHubEvent):
            WebhookEventHandler.parse_event(p_bytes, hdrs, webhook_secret=self.secret)

    def test_ignored_non_pr_event(self):
        """Verify non-pull_request events (e.g. 'push') raise UnsupportedGitHubEvent."""
        hdrs = dict(self.valid_headers)
        hdrs["X-GitHub-Event"] = "push"
        with self.assertRaises(UnsupportedGitHubEvent):
            WebhookEventHandler.parse_event(self.valid_payload_bytes, hdrs, webhook_secret=self.secret)

    def test_closed_merged_pr_ignored(self):
        """Verify closed PR state raises UnsupportedGitHubEvent."""
        data = dict(self.valid_payload_dict)
        data["pull_request"]["state"] = "closed"
        p_bytes = json.dumps(data).encode("utf-8")
        sig = hmac.new(self.secret.encode("utf-8"), p_bytes, hashlib.sha256).hexdigest()
        hdrs = dict(self.valid_headers)
        hdrs["X-Hub-Signature-256"] = f"sha256={sig}"

        with self.assertRaises(UnsupportedGitHubEvent):
            WebhookEventHandler.parse_event(p_bytes, hdrs, webhook_secret=self.secret)

    def test_delivery_id_handling(self):
        """Verify delivery ID is extracted correctly from header."""
        parsed = WebhookEventHandler.parse_event(
            self.valid_payload_bytes,
            self.valid_headers,
            webhook_secret=self.secret
        )
        self.assertEqual(parsed.delivery_id, "delivery-uuid-12345")


class TestGitHubPRReporter(unittest.TestCase):
    """Test suite for markdown formatting and comment API operations."""
    def setUp(self):
        self.mock_analysis_result = PRAnalysisResult(
            repo_path="/tmp/mock_repo",
            base_ref="main",
            head_ref="feature/test",
            resolved_base_sha="d94d341c6f10d94d341c6f10d94d341c6f10d94d",
            resolved_head_sha="340a95467927340a95467927340a95467927340a",
            resolved_merge_base_sha="d94d341c6f10d94d341c6f10d94d341c6f10d94d",
            is_empty=False,
            commit_count=1,
            commits=[
                PRCommitPrediction(
                    commit_hash="340a954",
                    full_hash="340a95467927340a95467927340a95467927340a",
                    author="Dev <dev@example.com>",
                    timestamp="2026-09-27T00:00:00Z",
                    commit_message="Fix bug",
                    is_merge_commit=False,
                    prediction=PredictionResult(
                        full_hash="340a95467927340a95467927340a95467927340a",
                        commit_hash="340a954",
                        repo_path="/tmp/mock_repo",
                        raw_probability=0.45,
                        is_above_threshold=True,
                        decision_threshold=0.35,
                        prediction_label="Defect Risk Change",
                        risk_level="HIGH",
                        presentation_disclaimer="Model probabilities represent estimated risk scores.",
                        features={"lines_added": 10},
                        model_signals=ModelSignals(
                            top_positive_factors={"lines_added": 0.5},
                            top_negative_factors={},
                            log_odds_contributions={}
                        ),
                        model_id="mock_model",
                        model_version="v1.0",
                        feature_schema_version="v1.0",
                        prediction_timestamp="2026-09-27T00:00:00Z",
                        model_artifact_hash="abc",
                        scaler_artifact_hash="def"
                    )
                )
            ],
            cumulative_analysis=LLMAnalysisResult(
                commit_hash="340a954",
                summary="Qualitative diff summary.",
                key_changes=["Updated bug fix logic."],
                potential_risk_factors=[
                    RiskFactor(
                        category="Boundary Check",
                        description="Check boundary edge case.",
                        review_priority="HIGH",
                        file_path="src/core.py",
                        line_range="L10-L12",
                        evidence_snippet="if x is None:",
                        evidence_verified=True
                    )
                ],
                affected_areas=["core"],
                testing_observations=["Unit tests added."],
                recommended_review_actions=[
                    ReviewAction(action="Review boundary condition", target_file="src/core.py")
                ],
                confidence_notes="High confidence.",
                model_provider="mock",
                model_name="mock-model"
            )
        )


    def test_reporter_creates_comment(self):
        """Verify POST request executed when no existing PatchGuard comment is found."""
        calls = []

        def mock_fetcher(url, headers, method="GET", payload_dict=None):
            calls.append((url, method, payload_dict))
            if method == "GET":
                return []  # No existing comments
            elif method == "POST":
                return {"id": 101, "html_url": "https://github.com/owner/repo/issues/comments/101"}

        reporter = GitHubPRReporter(
            token_provider=lambda: "mock_token",
            http_fetcher=mock_fetcher
        )

        formatted = reporter.format_report(self.mock_analysis_result)
        res = reporter.post_or_update_comment("owner", "repo", 42, formatted)

        self.assertEqual(res["action"], "created")
        self.assertEqual(res["comment_id"], 101)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], "GET")
        self.assertEqual(calls[1][1], "POST")

    def test_reporter_updates_existing_comment(self):
        """Verify PATCH request executed when an existing comment with IDEMPOTENCY_MARKER is found."""
        calls = []

        def mock_fetcher(url, headers, method="GET", payload_dict=None):
            calls.append((url, method, payload_dict))
            if method == "GET":
                return [
                    {"id": 99, "body": "Some other comment"},
                    {"id": 202, "body": f"{IDEMPOTENCY_MARKER}\nPrevious report"}
                ]
            elif method == "PATCH":
                return {"id": 202, "html_url": "https://github.com/owner/repo/issues/comments/202"}

        reporter = GitHubPRReporter(
            token_provider=lambda: "mock_token",
            http_fetcher=mock_fetcher
        )

        formatted = reporter.format_report(self.mock_analysis_result)
        res = reporter.post_or_update_comment("owner", "repo", 42, formatted)

        self.assertEqual(res["action"], "updated")
        self.assertEqual(res["comment_id"], 202)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], "GET")
        self.assertEqual(calls[1][1], "PATCH")

    def test_idempotency_marker_present(self):
        """Verify IDEMPOTENCY_MARKER is included at top of report body."""
        reporter = GitHubPRReporter(token_provider=lambda: "token")
        body = reporter.format_report(self.mock_analysis_result)
        self.assertTrue(body.startswith(IDEMPOTENCY_MARKER))

    def test_reporting_error_sanitization(self):
        """Verify GitHubReportingError sanitizes Authorization credentials."""
        def mock_failing_fetcher(url, headers, method="GET", payload_dict=None):
            raise Exception("Failed with Authorization: Bearer secret_token_abc")

        reporter = GitHubPRReporter(
            token_provider=lambda: "secret_token_abc",
            http_fetcher=mock_failing_fetcher
        )
        with self.assertRaises(GitHubReportingError) as ctx:
            reporter.post_or_update_comment("owner", "repo", 42, "body")

        self.assertNotIn("secret_token_abc", str(ctx.exception))


class MockPRAnalyzer:
    """Mock PRAnalyzer for workflow testing."""
    def __init__(self, analysis_result: PRAnalysisResult):
        self.analysis_result = analysis_result

    def analyze(self, pr_input, threshold=None):
        return self.analysis_result


class MockAcquisitionManager:
    """Mock RepositoryAcquisitionManager context manager for workflow testing."""
    def __init__(self, temp_path: str):
        self.temp_path = temp_path

    class TempRepoContext:
        def __init__(self, path):
            self.path = path
        def __enter__(self):
            return self.path
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def acquire(self, owner, repo, base_sha, head_sha, token=None, local_repo_override=None):
        return self.TempRepoContext(local_repo_override or self.temp_path)


class TestGitHubPRWorkflow(unittest.TestCase):
    """Test suite for full GitHubPRWorkflow orchestration."""
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        self.mock_analysis_result = PRAnalysisResult(
            repo_path=self.temp_dir,
            base_ref="main",
            head_ref="feature/fix",
            resolved_base_sha="d94d341c6f10d94d341c6f10d94d341c6f10d94d",
            resolved_head_sha="340a95467927340a95467927340a95467927340a",
            resolved_merge_base_sha="d94d341c6f10d94d341c6f10d94d341c6f10d94d",
            is_empty=False,
            commit_count=1,
            commits=[],
            cumulative_analysis=None
        )

        self.secret = "secret123"
        self.payload_dict = {
            "action": "opened",
            "number": 10,
            "pull_request": {
                "number": 10,
                "title": "Test PR",
                "html_url": "https://github.com/org/repo/pull/10",
                "state": "open",
                "base": {
                    "ref": "main",
                    "sha": "d94d341c6f10d94d341c6f10d94d341c6f10d94d",
                    "repo": {"name": "repo", "owner": {"login": "org"}}
                },
                "head": {
                    "ref": "feature/fix",
                    "sha": "340a95467927340a95467927340a95467927340a"
                }
            }
        }
        self.payload_bytes = json.dumps(self.payload_dict).encode("utf-8")
        sig = hmac.new(self.secret.encode("utf-8"), self.payload_bytes, hashlib.sha256).hexdigest()
        self.headers = {
            "X-Hub-Signature-256": f"sha256={sig}",
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "delivery-777"
        }

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_workflow_orchestration_success(self):
        """Test full workflow orchestration with mocked dependencies."""
        mock_analyzer = MockPRAnalyzer(self.mock_analysis_result)
        mock_acq = MockAcquisitionManager(self.temp_dir)

        posted_comments = []
        def mock_fetcher(url, headers, method="GET", payload_dict=None):
            if method == "GET":
                return []
            elif method == "POST":
                posted_comments.append(payload_dict["body"])
                return {"id": 555, "html_url": "https://github.com/org/repo/issues/comments/555"}

        reporter = GitHubPRReporter(
            token_provider=lambda: "token",
            http_fetcher=mock_fetcher
        )

        workflow = GitHubPRWorkflow(
            acquisition_manager=mock_acq,
            pr_analyzer=mock_analyzer,
            reporter=reporter
        )

        res = workflow.process_webhook(
            payload_bytes=self.payload_bytes,
            headers=self.headers,
            webhook_secret=self.secret,
            local_repo_override=self.temp_dir
        )

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["delivery_id"], "delivery-777")
        self.assertEqual(res["action"], "opened")
        self.assertEqual(len(posted_comments), 1)
        self.assertTrue(posted_comments[0].startswith(IDEMPOTENCY_MARKER))

    def test_secret_hygiene(self):
        """Verify credentials and secret keys are never exposed in workflow exceptions."""
        mock_acq = MockAcquisitionManager(self.temp_dir)
        mock_analyzer = MockPRAnalyzer(self.mock_analysis_result)

        def mock_failing_fetcher(url, headers, method="GET", payload_dict=None):
            raise Exception("Authorization: Bearer super_secret_jwt_token_abc")

        reporter = GitHubPRReporter(
            token_provider=lambda: "super_secret_jwt_token_abc",
            http_fetcher=mock_failing_fetcher
        )

        workflow = GitHubPRWorkflow(
            acquisition_manager=mock_acq,
            pr_analyzer=mock_analyzer,
            reporter=reporter
        )

        with self.assertRaises(GitHubIntegrationError) as ctx:
            workflow.process_webhook(
                payload_bytes=self.payload_bytes,
                headers=self.headers,
                webhook_secret=self.secret,
                local_repo_override=self.temp_dir
            )

        self.assertNotIn("super_secret_jwt_token_abc", str(ctx.exception))



if __name__ == "__main__":
    unittest.main()
