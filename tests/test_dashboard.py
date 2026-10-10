"""
Unit tests for PatchGuard UI Helpers (Phase 9.1).
Tests pure-Python presentation formatting, model signal transformations,
canonical feature ordering, badge derivations, exception mapping, and
evidence display formatting without any Streamlit runtime dependency.
"""

import os
import shutil
import tempfile
import subprocess
import json
import unittest
from typing import Dict, Any

try:
    from streamlit.testing.v1 import AppTest
    HAS_APP_TEST = True
except ImportError:
    HAS_APP_TEST = False

from src.engine.schema import FEATURE_COLUMNS, ModelSignals
from src.analysis.schema import RiskFactor
from src.engine.exceptions import (
    PatchGuardError,
    RepositoryNotFoundError,
    InvalidGitRepositoryError,
    CommitNotFoundError,
    UnsupportedCommitTypeError,
    EmptyCommitError,
    ModelArtifactNotFoundError,
    ScalerArtifactNotFoundError,
    CorruptedArtifactError,
    IncompatibleSchemaError,
    InvalidFeatureValueError,
    GitExtractionError,
)
from src.analysis.exceptions import (
    MalformedLLMResponseError,
    OpenAIProviderError,
    MissingAPIKeyError,
    InvalidPRRefError,
)
from src.github.exceptions import (
    GitHubIntegrationError,
    GitHubPRNotFoundError,
    GitHubConfigError,
)
from src.ui.helpers import (
    prepare_signals_chart_data,
    format_features_dataframe,
    get_risk_level_badge,
    format_error_message,
    format_risk_factor_display,
)
from src.ui.benchmark_helpers import (
    load_benchmark_data,
    format_benchmark_comparison_records,
    prepare_benchmark_chart_records,
    get_split_integrity_summary,
    get_decision_justification,
    DEFAULT_BENCHMARK_PATH,
)
from src.analysis.pr_schema import PRCommitPrediction, PRAnalysisResult, PRAnalysisInput
from src.engine.schema import PredictionResult
from src.github.exceptions import (
    GitHubAuthError,
    GitHubPRValidationError,
    GitHubAPIError,
    RepositoryAcquisitionError,
    CommitNotAvailableError,
)
from src.ui.pr_helpers import (
    validate_local_pr_inputs,
    parse_github_pr_identifier,
    is_github_app_configured,
    get_pr_risk_summary_stats,
    format_pr_commits_table,
    format_pr_error,
)


class TestPrepareSignalsChartData(unittest.TestCase):
    """Tests for prepare_signals_chart_data()."""

    def test_full_model_signals_preserves_names_and_values(self):
        """Preserves exact feature names and numeric contribution values."""
        signals = ModelSignals(
            top_positive_factors={"code_churn": 0.4521, "files_changed": 0.1234},
            top_negative_factors={"is_test_file_modified": -0.3210},
            log_odds_contributions={
                "code_churn": 0.4521,
                "files_changed": 0.1234,
                "is_test_file_modified": -0.3210,
                "lines_added": 0.0,
            },
        )
        records = prepare_signals_chart_data(signals)
        self.assertEqual(len(records), 4)

        # Sorted by absolute contribution descending
        self.assertEqual(records[0]["feature"], "code_churn")
        self.assertAlmostEqual(records[0]["contribution"], 0.4521)
        self.assertEqual(records[0]["direction"], "Positive")
        self.assertEqual(records[0]["impact"], "Increases Risk")

        self.assertEqual(records[1]["feature"], "is_test_file_modified")
        self.assertAlmostEqual(records[1]["contribution"], -0.3210)
        self.assertEqual(records[1]["direction"], "Negative")
        self.assertEqual(records[1]["impact"], "Decreases Risk")

        self.assertEqual(records[2]["feature"], "files_changed")
        self.assertAlmostEqual(records[2]["contribution"], 0.1234)

        self.assertEqual(records[3]["feature"], "lines_added")
        self.assertAlmostEqual(records[3]["contribution"], 0.0)
        self.assertEqual(records[3]["direction"], "Neutral")
        self.assertEqual(records[3]["impact"], "Neutral")

    def test_empty_and_none_signals_return_empty_list(self):
        """Handles None and empty signal collections gracefully."""
        self.assertEqual(prepare_signals_chart_data(None), [])

        empty_signals = ModelSignals(
            top_positive_factors={},
            top_negative_factors={},
            log_odds_contributions={},
        )
        self.assertEqual(prepare_signals_chart_data(empty_signals), [])

    def test_fallback_to_top_factors_when_log_odds_empty(self):
        """Falls back to merging positive and negative factors if log_odds dict is empty."""
        signals = ModelSignals(
            top_positive_factors={"lines_added": 0.5},
            top_negative_factors={"num_directories_touched": -0.2},
            log_odds_contributions={},
        )
        records = prepare_signals_chart_data(signals)
        self.assertEqual(len(records), 2)
        features = {r["feature"] for r in records}
        self.assertEqual(features, {"lines_added", "num_directories_touched"})

    def test_dict_input_support(self):
        """Supports dictionary input conforming to ModelSignals schema."""
        dict_signals = {
            "top_positive_factors": {"code_churn": 0.8},
            "top_negative_factors": {},
            "log_odds_contributions": {"code_churn": 0.8, "files_changed": -0.1},
        }
        records = prepare_signals_chart_data(dict_signals)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["feature"], "code_churn")
        self.assertAlmostEqual(records[0]["contribution"], 0.8)

    def test_invalid_type_raises_type_error(self):
        """Rejects invalid types with TypeError."""
        with self.assertRaises(TypeError):
            prepare_signals_chart_data("not_signals")  # type: ignore
        with self.assertRaises(TypeError):
            prepare_signals_chart_data(12345)  # type: ignore


class TestFormatFeaturesDataframe(unittest.TestCase):
    """Tests for format_features_dataframe()."""

    def test_exact_canonical_schema_and_ordering(self):
        """Guarantees all 10 canonical features in exact production order."""
        features_dict = {
            "lines_added": 50,
            "lines_deleted": 10,
            "code_churn": 60,
            "files_changed": 3,
            "functions_changed": 2,
            "num_directories_touched": 1,
            "is_test_file_modified": 0,
            "avg_lines_changed_per_file": 20.0,
            "max_lines_changed_in_single_file": 45,
            "num_source_files_changed": 2,
        }
        records = format_features_dataframe(features_dict)
        self.assertEqual(len(records), 10)

        # Check exact names, order, and canonical flag
        for idx, col in enumerate(FEATURE_COLUMNS):
            self.assertEqual(records[idx]["feature"], col)
            self.assertEqual(records[idx]["value"], features_dict[col])
            self.assertTrue(records[idx]["is_canonical"])

    def test_missing_features_represented_as_none_not_invented(self):
        """Missing features are represented as None, never silently invented as 0."""
        partial_features = {
            "lines_added": 15,
            "lines_deleted": 5,
            # 'code_churn' missing
        }
        records = format_features_dataframe(partial_features)
        self.assertEqual(len(records), 10)
        self.assertEqual(records[0]["feature"], "lines_added")
        self.assertEqual(records[0]["value"], 15)

        # code_churn is index 2
        self.assertEqual(records[2]["feature"], "code_churn")
        self.assertIsNone(records[2]["value"])

    def test_extra_unexpected_keys_appended_after_canonical_ten(self):
        """Extra unexpected keys are preserved after the canonical 10 features."""
        features_dict = {col: 1 for col in FEATURE_COLUMNS}
        features_dict["unexpected_metric"] = 999

        records = format_features_dataframe(features_dict)
        self.assertEqual(len(records), 11)
        self.assertEqual(records[10]["feature"], "unexpected_metric")
        self.assertEqual(records[10]["value"], 999)
        self.assertFalse(records[10]["is_canonical"])

    def test_non_dict_input_raises_type_error(self):
        """Rejects non-dict and None inputs."""
        with self.assertRaises(TypeError):
            format_features_dataframe(None)
        with self.assertRaises(TypeError):
            format_features_dataframe(["not", "a", "dict"])  # type: ignore


class TestGetRiskLevelBadge(unittest.TestCase):
    """Tests for get_risk_level_badge()."""

    def test_low_risk_below_threshold(self):
        """LOW risk below threshold returns green with checkmark icon."""
        badge = get_risk_level_badge("LOW", is_above_threshold=False)
        self.assertEqual(badge["risk_level"], "LOW")
        self.assertFalse(badge["is_above_threshold"])
        self.assertEqual(badge["color"], "green")
        self.assertEqual(badge["icon"], "✅")
        self.assertEqual(badge["label"], "Standard Risk Change")
        self.assertIn("Below Threshold", badge["threshold_status"])

    def test_medium_risk_above_threshold(self):
        """MEDIUM risk above threshold returns amber with warning icon."""
        badge = get_risk_level_badge("MEDIUM", is_above_threshold=True)
        self.assertEqual(badge["risk_level"], "MEDIUM")
        self.assertTrue(badge["is_above_threshold"])
        self.assertEqual(badge["color"], "amber")
        self.assertEqual(badge["icon"], "⚠️")
        self.assertEqual(badge["label"], "Elevated Defect Risk")
        self.assertIn("Above Threshold", badge["threshold_status"])

    def test_high_risk_above_threshold(self):
        """HIGH risk above threshold returns red with warning icon."""
        badge = get_risk_level_badge("HIGH", is_above_threshold=True)
        self.assertEqual(badge["risk_level"], "HIGH")
        self.assertTrue(badge["is_above_threshold"])
        self.assertEqual(badge["color"], "red")
        self.assertEqual(badge["icon"], "⚠️")
        self.assertEqual(badge["label"], "Elevated Defect Risk")

    def test_case_insensitivity_and_whitespace(self):
        """Normalizes lowercase and trailing whitespace in risk level."""
        badge = get_risk_level_badge("  low  ", is_above_threshold=False)
        self.assertEqual(badge["risk_level"], "LOW")
        self.assertEqual(badge["color"], "green")

    def test_unknown_and_none_risk_level(self):
        """Handles None or unrecognized risk levels safely as UNKNOWN."""
        badge1 = get_risk_level_badge(None, is_above_threshold=False)
        self.assertEqual(badge1["risk_level"], "UNKNOWN")
        self.assertEqual(badge1["color"], "gray")
        self.assertEqual(badge1["icon"], "ℹ️")

        badge2 = get_risk_level_badge("CRITICAL_UNKNOWN", is_above_threshold=True)
        self.assertEqual(badge2["risk_level"], "UNKNOWN")
        self.assertEqual(badge2["color"], "gray")

    def test_disclaimer_present(self):
        """Ensures presentation disclaimer is explicitly included."""
        badge = get_risk_level_badge("LOW", False)
        self.assertIn("presentation labels only", badge["disclaimer"])


class TestFormatErrorMessage(unittest.TestCase):
    """Tests for format_error_message()."""

    def test_repository_not_found(self):
        exc = RepositoryNotFoundError("Directory does not exist: /path/to/missing")
        err = format_error_message(exc)
        self.assertEqual(err["title"], "Repository Not Found")
        self.assertEqual(err["error_type"], "RepositoryNotFoundError")
        self.assertIn("Verify the repository directory path", err["remediation"])

    def test_invalid_git_repository(self):
        exc = InvalidGitRepositoryError("Not a git repository: /not/git")
        err = format_error_message(exc)
        self.assertEqual(err["title"], "Invalid Git Repository")
        self.assertIn("'.git'", err["remediation"])

    def test_commit_not_found(self):
        exc = CommitNotFoundError("Commit abcdef123 not found")
        err = format_error_message(exc)
        self.assertEqual(err["title"], "Commit Not Found")
        self.assertIn("Check the commit SHA", err["remediation"])

    def test_unsupported_commit_type_merge(self):
        exc = UnsupportedCommitTypeError("Commit 1234567 is a merge commit")
        err = format_error_message(exc)
        self.assertEqual(err["title"], "Unsupported Commit Type")
        self.assertIn("Merge commits are excluded", err["message"])
        self.assertIn("non-merge commit", err["remediation"])

    def test_empty_commit(self):
        exc = EmptyCommitError("Commit has 0 files changed")
        err = format_error_message(exc)
        self.assertEqual(err["title"], "Empty Commit")
        self.assertIn("no modified", err["message"])

    def test_missing_and_corrupted_model_artifacts(self):
        exc_model = ModelArtifactNotFoundError("Missing model.pkl")
        err_model = format_error_message(exc_model)
        self.assertEqual(err_model["title"], "Model Artifact Missing")

        exc_scaler = ScalerArtifactNotFoundError("Missing scaler.pkl")
        err_scaler = format_error_message(exc_scaler)
        self.assertEqual(err_scaler["title"], "Scaler Artifact Missing")

        exc_corrupt = CorruptedArtifactError("Cannot unpickle")
        err_corrupt = format_error_message(exc_corrupt)
        self.assertEqual(err_corrupt["title"], "Corrupted Artifact")

    def test_incompatible_schema_and_invalid_values(self):
        exc_schema = IncompatibleSchemaError("Feature mismatch")
        err_schema = format_error_message(exc_schema)
        self.assertEqual(err_schema["title"], "Incompatible Feature Schema")

        exc_vals = InvalidFeatureValueError("NaN detected")
        err_vals = format_error_message(exc_vals)
        self.assertEqual(err_vals["title"], "Invalid Feature Values")

    def test_llm_and_provider_errors(self):
        exc_key = MissingAPIKeyError("OPENAI_API_KEY missing")
        err_key = format_error_message(exc_key)
        self.assertEqual(err_key["title"], "Missing OpenAI API Key")

        exc_provider = OpenAIProviderError("Rate limit exceeded")
        err_provider = format_error_message(exc_provider)
        self.assertEqual(err_provider["title"], "OpenAI Provider Error")

        exc_json = MalformedLLMResponseError("Invalid JSON")
        err_json = format_error_message(exc_json)
        self.assertEqual(err_json["title"], "Malformed LLM Response")

    def test_github_and_pr_errors(self):
        exc_pr = GitHubPRNotFoundError(owner="bottlepy", repo="bottle", pull_number=42)
        err_pr = format_error_message(exc_pr)
        self.assertEqual(err_pr["title"], "GitHub PR Not Found")
        self.assertIn("#42", err_pr["message"])

        exc_ref = InvalidPRRefError("No common merge-base")
        err_ref = format_error_message(exc_ref)
        self.assertEqual(err_ref["title"], "Invalid Pull Request Reference")

    def test_secret_sanitization(self):
        """Ensures Authorization headers or bearer tokens are sanitized from messages."""
        exc = GitHubIntegrationError("Failed with Authorization: Bearer secret_token_12345 in header")
        err = format_error_message(exc)
        self.assertNotIn("secret_token_12345", err["message"])

    def test_unknown_generic_exception(self):
        """Unknown generic exceptions receive safe generic handling."""
        exc = RuntimeError("Some internal system error with /secret/path/info")
        err = format_error_message(exc)
        self.assertEqual(err["title"], "Unexpected Error")
        self.assertEqual(err["error_type"], "RuntimeError")
        self.assertIn("unexpected error occurred", err["message"])

    def test_non_exception_object(self):
        """Handles non-exception objects safely."""
        err = format_error_message("string_error")  # type: ignore
        self.assertEqual(err["title"], "Invalid Error Object")


class TestFormatRiskFactorDisplay(unittest.TestCase):
    """Tests for format_risk_factor_display()."""

    def test_verified_risk_factor(self):
        """Verified risk factor formats correctly with verified badge."""
        rf = RiskFactor(
            category="Error Handling",
            description="Missing None check on param.",
            review_priority="HIGH",
            file_path="src/module.py",
            line_range="L45-L52",
            evidence_snippet="if param is None: return",
            evidence_verified=True,
        )
        display = format_risk_factor_display(rf)
        self.assertEqual(display["category"], "Error Handling")
        self.assertEqual(display["description"], "Missing None check on param.")
        self.assertEqual(display["review_priority"], "HIGH")
        self.assertEqual(display["priority_badge"], "[HIGH]")
        self.assertEqual(display["file_path"], "src/module.py")
        self.assertEqual(display["line_range"], "L45-L52")
        self.assertEqual(display["location_display"], "src/module.py (L45-L52)")
        self.assertEqual(display["evidence_snippet"], "if param is None: return")
        self.assertTrue(display["evidence_verified"])
        self.assertEqual(display["verification_badge"], "[Verified Grounding]")
        self.assertEqual(display["verification_status"], "Verified")

    def test_unverified_risk_factor_never_labeled_verified(self):
        """Unverified risk factor explicitly gets ungrounded badge."""
        rf = RiskFactor(
            category="Concurrency",
            description="Race condition possible.",
            review_priority="MEDIUM",
            file_path="src/worker.py",
            line_range=None,
            evidence_snippet="",
            evidence_verified=False,
        )
        display = format_risk_factor_display(rf)
        self.assertFalse(display["evidence_verified"])
        self.assertEqual(display["verification_badge"], "[Ungrounded Citation]")
        self.assertEqual(display["verification_status"], "Ungrounded")
        self.assertIsNone(display["line_range"])
        self.assertEqual(display["location_display"], "src/worker.py")
        self.assertIsNone(display["evidence_snippet"])

    def test_dict_input_support(self):
        """Supports dictionary representation conforming to RiskFactor schema."""
        rf_dict = {
            "category": "Security",
            "description": "SQL query format string.",
            "review_priority": "HIGH",
            "file_path": "db/query.py",
            "line_range": "L10",
            "evidence_snippet": "f'SELECT * FROM {table}'",
            "evidence_verified": True,
        }
        display = format_risk_factor_display(rf_dict)
        self.assertEqual(display["category"], "Security")
        self.assertEqual(display["review_priority"], "HIGH")
        self.assertTrue(display["evidence_verified"])

    def test_dict_with_invalid_priority_raises_value_error(self):
        """Raises ValueError if dictionary has invalid review_priority."""
        invalid_dict = {
            "category": "Security",
            "review_priority": "CRITICAL",  # Not in LOW, MEDIUM, HIGH
        }
        with self.assertRaises(ValueError):
            format_risk_factor_display(invalid_dict)

    def test_invalid_type_raises_type_error(self):
        """Raises TypeError if input is not RiskFactor or dict."""
        with self.assertRaises(TypeError):
            format_risk_factor_display(12345)  # type: ignore


class TestCommitRiskInspectorApp(unittest.TestCase):
    """
    Headless integration tests for Commit Risk Inspector UI using Streamlit AppTest.
    """

    @classmethod
    def setUpClass(cls):
        if not HAS_APP_TEST:
            raise unittest.SkipTest("Streamlit AppTest is not available.")
        cls.script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "src", "ui", "app.py")
        )

    def setUp(self):
        self.temp_repos = []

    def tearDown(self):
        for repo_dir in self.temp_repos:
            if os.path.exists(repo_dir):
                shutil.rmtree(repo_dir, ignore_errors=True)

    def _create_temp_git_repo(self) -> str:
        temp_dir = tempfile.mkdtemp(prefix="test_ui_repo_")
        self.temp_repos.append(temp_dir)
        subprocess.run(["git", "init"], cwd=temp_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test Author"], cwd=temp_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_dir, check=True, capture_output=True)
        test_file = os.path.join(temp_dir, "test.py")
        with open(test_file, "w") as f:
            f.write("def sample_function():\n    return 42\n")
        subprocess.run(["git", "add", "."], cwd=temp_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial test commit"], cwd=temp_dir, check=True, capture_output=True)
        return temp_dir

    def test_app_initial_render(self):
        """Verifies initial app layout, default inputs, and absence of premature results."""
        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("PatchGuard" in str(t.value) for t in at.title))
        self.assertEqual(at.text_input(key="commit_ref_input").value, "HEAD")
        self.assertTrue(len(at.text_input(key="repo_path_input").value) > 0)
        self.assertEqual(len(at.metric), 0)
        self.assertEqual(len(at.dataframe), 0)
        self.assertTrue(len(at.info) >= 1)

    def test_app_successful_analysis_temporary_repo(self):
        """Verifies end-to-end commit risk prediction, metric cards, feature table, and provenance."""
        temp_dir = self._create_temp_git_repo()
        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.text_input(key="repo_path_input").input(temp_dir)
        at.text_input(key="commit_ref_input").input("HEAD")
        at.button(key="analyze_button").click().run(timeout=15)

        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.error), 0)
        self.assertEqual(len(at.metric), 4)
        self.assertEqual(at.metric[1].value, "0.35")
        self.assertEqual(len(at.dataframe), 1)
        self.assertTrue(len(at.expander) >= 1)

    def test_app_invalid_repository_path(self):
        """Verifies error handling for non-existent repository directories."""
        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.text_input(key="repo_path_input").input("C:\\non_existent_directory_98765")
        at.button(key="analyze_button").click().run(timeout=15)

        self.assertEqual(len(at.exception), 0)
        self.assertTrue(len(at.error) >= 1)
        self.assertIn("Repository Not Found", str(at.error[0].value))
        self.assertEqual(len(at.metric), 0)

    def test_app_missing_commit_hash(self):
        """Verifies error handling for non-existent commit hashes."""
        temp_dir = self._create_temp_git_repo()
        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.text_input(key="repo_path_input").input(temp_dir)
        at.text_input(key="commit_ref_input").input("missing_commit_sha_12345")
        at.button(key="analyze_button").click().run(timeout=15)

        self.assertEqual(len(at.exception), 0)
        self.assertTrue(len(at.error) >= 1)
        self.assertIn("Commit Not Found", str(at.error[0].value))
        self.assertEqual(len(at.metric), 0)

    def test_app_unsupported_merge_commit(self):
        """Verifies that merge commits are rejected with clear explanation."""
        temp_dir = self._create_temp_git_repo()
        subprocess.run(["git", "checkout", "-b", "feature_branch"], cwd=temp_dir, check=True, capture_output=True)
        feat_file = os.path.join(temp_dir, "feat.py")
        with open(feat_file, "w") as f:
            f.write("# feature\n")
        subprocess.run(["git", "add", "."], cwd=temp_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Feature commit"], cwd=temp_dir, check=True, capture_output=True)
        subprocess.run(["git", "checkout", "-"], cwd=temp_dir, check=True, capture_output=True)
        main_file = os.path.join(temp_dir, "main.py")
        with open(main_file, "w") as f:
            f.write("# main line\n")
        subprocess.run(["git", "add", "."], cwd=temp_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Main commit"], cwd=temp_dir, check=True, capture_output=True)
        subprocess.run(["git", "merge", "feature_branch", "-m", "Merge commit"], cwd=temp_dir, check=True, capture_output=True)

        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.text_input(key="repo_path_input").input(temp_dir)
        at.text_input(key="commit_ref_input").input("HEAD")
        at.button(key="analyze_button").click().run(timeout=15)

        self.assertEqual(len(at.exception), 0)
        self.assertTrue(len(at.error) >= 1)
        self.assertIn("Unsupported Commit Type", str(at.error[0].value))
        self.assertEqual(len(at.metric), 0)

    def test_app_empty_commit(self):
        """Verifies that commits with zero file modifications are handled cleanly."""
        temp_dir = self._create_temp_git_repo()
        subprocess.run(["git", "commit", "--allow-empty", "-m", "Empty commit"], cwd=temp_dir, check=True, capture_output=True)

        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.text_input(key="repo_path_input").input(temp_dir)
        at.text_input(key="commit_ref_input").input("HEAD")
        at.button(key="analyze_button").click().run(timeout=15)

        self.assertEqual(len(at.exception), 0)
        self.assertTrue(len(at.error) >= 1)
        self.assertIn("Empty Commit", str(at.error[0].value))
        self.assertEqual(len(at.metric), 0)

    def test_app_failure_state_clears_stale_results(self):
        """Verifies that an error in a subsequent analysis cleanly clears previous results."""
        temp_dir = self._create_temp_git_repo()
        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.text_input(key="repo_path_input").input(temp_dir)
        at.text_input(key="commit_ref_input").input("HEAD")
        at.button(key="analyze_button").click().run(timeout=15)

        # 1. First run succeeds
        self.assertEqual(len(at.error), 0)
        self.assertEqual(len(at.metric), 4)

        # 2. Next run with invalid commit must clear previous metric cards
        at.text_input(key="commit_ref_input").input("bad_commit_hash")
        at.button(key="analyze_button").click().run(timeout=15)

        self.assertTrue(len(at.error) >= 1)
        self.assertEqual(len(at.metric), 0)
        self.assertEqual(len(at.dataframe), 0)


class TestQualitativeLLMAnalysisUI(unittest.TestCase):
    """
    Headless integration tests for Phase 9.3 Qualitative LLM Analysis UI in Streamlit.
    Validates optional opt-in, offline mock provider, OpenAI provider error handling,
    grounding verification badge rendering, and stale state isolation.
    """

    @classmethod
    def setUpClass(cls):
        if not HAS_APP_TEST:
            raise unittest.SkipTest("Streamlit AppTest is not available.")
        cls.script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "src", "ui", "app.py")
        )

    def setUp(self):
        self.temp_repos = []

    def tearDown(self):
        for repo_dir in self.temp_repos:
            if os.path.exists(repo_dir):
                shutil.rmtree(repo_dir, ignore_errors=True)

    def _create_temp_git_repo(self) -> str:
        temp_dir = tempfile.mkdtemp(prefix="test_llm_ui_repo_")
        self.temp_repos.append(temp_dir)
        subprocess.run(["git", "init"], cwd=temp_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test Author"], cwd=temp_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_dir, check=True, capture_output=True)
        test_file = os.path.join(temp_dir, "test.py")
        with open(test_file, "w") as f:
            f.write("def sample_function():\n    return 42\n")
        subprocess.run(["git", "add", "."], cwd=temp_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial test commit"], cwd=temp_dir, check=True, capture_output=True)
        return temp_dir

    def test_initial_render_does_not_invoke_llm(self):
        """1. Validates that initial page render does not render LLM controls or run LLM analysis."""
        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        self.assertEqual(len(at.exception), 0)
        button_keys = [b.key for b in at.button]
        self.assertIn("analyze_button", button_keys)
        self.assertNotIn("run_llm_button", button_keys)
        self.assertFalse(any("Commit Summary" in str(m.value) for m in at.markdown))
        self.assertEqual(len(at.error), 0)

    def test_ml_prediction_available_when_qualitative_analysis_disabled(self):
        """2. Validates that ML prediction works independently and LLM analysis is not run automatically."""
        temp_dir = self._create_temp_git_repo()
        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.text_input(key="repo_path_input").input(temp_dir)
        at.text_input(key="commit_ref_input").input("HEAD")
        at.button(key="analyze_button").click().run(timeout=15)

        self.assertEqual(len(at.error), 0)
        self.assertEqual(len(at.metric), 4)
        button_keys = [b.key for b in at.button]
        self.assertIn("run_llm_button", button_keys)
        self.assertFalse(any("Commit Summary" in str(m.value) for m in at.markdown))
        self.assertFalse(any("Mock commit summary" in str(m.value) for m in at.markdown))

    def test_explicit_mock_analysis_uses_mock_provider(self):
        """3. Validates explicit mock analysis execution, rendering summary, factors, and review actions."""
        temp_dir = self._create_temp_git_repo()
        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.text_input(key="repo_path_input").input(temp_dir)
        at.text_input(key="commit_ref_input").input("HEAD")
        at.button(key="analyze_button").click().run(timeout=15)

        self.assertEqual(at.selectbox(key="llm_provider_select").value, "mock")
        at.button(key="run_llm_button").click().run(timeout=15)

        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.error), 0)
        self.assertEqual(len(at.metric), 4)

        all_markdown = " ".join(str(m.value) for m in at.markdown)
        self.assertIn("Commit Summary", all_markdown)
        self.assertIn("Mock commit summary", all_markdown)
        self.assertIn("Key Code Changes", all_markdown)
        self.assertIn("Potential Risk Factors", all_markdown)
        self.assertIn("[MEDIUM]", all_markdown)
        self.assertIn("Error Handling", all_markdown)
        self.assertIn("Recommended Review Actions", all_markdown)
        self.assertIn("Verify boundary condition in example.py", all_markdown)

        caption_text = " ".join(str(c.value) for c in at.caption)
        self.assertIn("[Ungrounded Citation]", caption_text)
        self.assertIn("mock", caption_text)

    def test_mock_analysis_requires_no_api_key_or_network(self):
        """4. Validates that mock provider operates completely offline without OPENAI_API_KEY."""
        temp_dir = self._create_temp_git_repo()
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            at = AppTest.from_file(self.script_path, default_timeout=15).run()
            at.text_input(key="repo_path_input").input(temp_dir)
            at.text_input(key="commit_ref_input").input("HEAD")
            at.button(key="analyze_button").click().run(timeout=15)
            at.button(key="run_llm_button").click().run(timeout=15)

            self.assertEqual(len(at.exception), 0)
            self.assertEqual(len(at.error), 0)
            all_markdown = " ".join(str(m.value) for m in at.markdown)
            self.assertIn("Mock commit summary", all_markdown)
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_grounded_evidence_display_with_matching_diff(self):
        """5. Validates that citations matching actual commit diff are marked as [Verified Grounding]."""
        temp_dir = self._create_temp_git_repo()
        ex_file = os.path.join(temp_dir, "example.py")
        with open(ex_file, "w") as f:
            lines = ["# line\n"] * 9 + ["if param is None: return\n"] + ["# end\n"] * 5
            f.writelines(lines)
        subprocess.run(["git", "add", "."], cwd=temp_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add example file with snippet"], cwd=temp_dir, check=True, capture_output=True)

        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.text_input(key="repo_path_input").input(temp_dir)
        at.text_input(key="commit_ref_input").input("HEAD")
        at.button(key="analyze_button").click().run(timeout=15)
        at.button(key="run_llm_button").click().run(timeout=15)

        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.error), 0)
        caption_text = " ".join(str(c.value) for c in at.caption)
        self.assertIn("[Verified Grounding]", caption_text)

    def test_provider_failure_displays_sanitized_error_preserves_ml_prediction(self):
        """6. Validates that an LLM failure displays a sanitized error without clearing ML prediction."""
        temp_dir = self._create_temp_git_repo()
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            at = AppTest.from_file(self.script_path, default_timeout=15).run()
            at.text_input(key="repo_path_input").input(temp_dir)
            at.text_input(key="commit_ref_input").input("HEAD")
            at.button(key="analyze_button").click().run(timeout=15)

            self.assertEqual(len(at.metric), 4)

            at.selectbox(key="llm_provider_select").select("openai")
            at.button(key="run_llm_button").click().run(timeout=15)

            self.assertEqual(len(at.exception), 0)
            self.assertTrue(len(at.error) >= 1)
            error_text = " ".join(str(e.value) for e in at.error)
            self.assertIn("Missing OpenAI API Key", error_text)
            self.assertIn("OPENAI_API_KEY environment variable is missing or empty", error_text)

            info_text = " ".join(str(i.value) for i in at.info)
            self.assertIn("Set OPENAI_API_KEY in your environment or use the offline 'mock' provider", info_text)

            self.assertEqual(len(at.metric), 4)
            self.assertEqual(len(at.dataframe), 1)
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_changing_inputs_invalidates_stale_results(self):
        """7. Validates that changing the commit reference invalidates both ML metrics and LLM results."""
        temp_dir = self._create_temp_git_repo()
        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.text_input(key="repo_path_input").input(temp_dir)
        at.text_input(key="commit_ref_input").input("HEAD")
        at.button(key="analyze_button").click().run(timeout=15)
        at.button(key="run_llm_button").click().run(timeout=15)

        self.assertEqual(len(at.metric), 4)
        self.assertTrue(any("Commit Summary" in str(m.value) for m in at.markdown))

        at.text_input(key="commit_ref_input").input("HEAD~1").run(timeout=15)

        self.assertEqual(len(at.metric), 0)
        self.assertFalse(any("Commit Summary" in str(m.value) for m in at.markdown))

    def test_openai_missing_config_does_not_silently_run_mock(self):
        """8. Validates that requesting OpenAI with missing credentials fails explicitly and never runs mock."""
        temp_dir = self._create_temp_git_repo()
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            at = AppTest.from_file(self.script_path, default_timeout=15).run()
            at.text_input(key="repo_path_input").input(temp_dir)
            at.text_input(key="commit_ref_input").input("HEAD")
            at.button(key="analyze_button").click().run(timeout=15)

            at.selectbox(key="llm_provider_select").select("openai")
            at.button(key="run_llm_button").click().run(timeout=15)

            self.assertTrue(len(at.error) >= 1)
            all_markdown = " ".join(str(m.value) for m in at.markdown)
            self.assertNotIn("Mock commit summary", all_markdown)
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_no_stale_analysis_shown_after_failed_analysis_request(self):
        """9. Validates that a failed LLM analysis request clears prior qualitative results."""
        temp_dir = self._create_temp_git_repo()
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            at = AppTest.from_file(self.script_path, default_timeout=15).run()
            at.text_input(key="repo_path_input").input(temp_dir)
            at.text_input(key="commit_ref_input").input("HEAD")
            at.button(key="analyze_button").click().run(timeout=15)

            # 1. Run mock analysis (succeeds)
            at.button(key="run_llm_button").click().run(timeout=15)
            self.assertTrue(any("Mock commit summary" in str(m.value) for m in at.markdown))

            # 2. Run openai without key -> fails
            at.selectbox(key="llm_provider_select").select("openai")
            at.button(key="run_llm_button").click().run(timeout=15)

            all_markdown = " ".join(str(m.value) for m in at.markdown)
            self.assertNotIn("Mock commit summary", all_markdown)
            self.assertTrue(len(at.error) >= 1)
            self.assertEqual(len(at.metric), 4)
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key


class TestBenchmarkHelpers(unittest.TestCase):
    """
    Unit tests for Phase 9.4 pure-Python benchmark helpers in src/ui/benchmark_helpers.py.
    """

    def setUp(self):
        self.temp_files = []

    def tearDown(self):
        for f in self.temp_files:
            if os.path.exists(f):
                os.remove(f)

    def _create_temp_json(self, content: Any) -> str:
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        if isinstance(content, str):
            tf.write(content)
        else:
            json.dump(content, tf)
        tf.close()
        self.temp_files.append(tf.name)
        return tf.name

    def test_load_benchmark_data_real_file(self):
        """1. Verifies that the real Phase 8 benchmark JSON file loads cleanly with all required sections."""
        data = load_benchmark_data()
        self.assertIsInstance(data, dict)
        self.assertIn("metadata", data)
        self.assertIn("production_logistic_regression_anchor", data)
        self.assertIn("benchmark_candidates", data)
        self.assertIn("benchmark_summary", data)

    def test_load_benchmark_data_file_not_found(self):
        """2. Verifies FileNotFoundError when passing a non-existent benchmark path."""
        with self.assertRaises(FileNotFoundError):
            load_benchmark_data("C:\\non_existent_benchmark_file_9999.json")

    def test_load_benchmark_data_malformed_json(self):
        """3. Verifies ValueError when loading corrupted or malformed JSON."""
        bad_json_path = self._create_temp_json("{bad_json: missing_quotes,")
        with self.assertRaises(ValueError) as ctx:
            load_benchmark_data(bad_json_path)
        self.assertIn("Malformed benchmark JSON report", str(ctx.exception))

    def test_load_benchmark_data_missing_sections(self):
        """4. Verifies ValueError when required sections are missing from benchmark report."""
        incomplete_json_path = self._create_temp_json({"metadata": {}})
        with self.assertRaises(ValueError) as ctx:
            load_benchmark_data(incomplete_json_path)
        self.assertIn("Incomplete benchmark report schema", str(ctx.exception))

    def test_format_benchmark_comparison_records(self):
        """5. Verifies extraction of comparison records, distinguishing production anchor from candidates."""
        data = load_benchmark_data()
        records = format_benchmark_comparison_records(data)
        self.assertEqual(len(records), 4)

        # Anchor checks
        anchor = records[0]
        self.assertTrue(anchor["is_production"])
        self.assertFalse(anchor["is_winner"])
        self.assertEqual(anchor["threshold"], 0.35)
        self.assertAlmostEqual(anchor["val_pr_auc"], 0.6220, places=4)
        self.assertAlmostEqual(anchor["test_pr_auc"], 0.3086, places=4)

        # XGBoost winner checks
        xgboost_rec = [r for r in records if "XGBoost" in r["model_name"]][0]
        self.assertFalse(xgboost_rec["is_production"])
        self.assertTrue(xgboost_rec["is_winner"])
        self.assertAlmostEqual(xgboost_rec["val_pr_auc"], 0.6293, places=4)
        self.assertAlmostEqual(xgboost_rec["test_pr_auc"], 0.3535, places=4)

        # Check all records distinguish val vs test
        for r in records:
            self.assertIsNotNone(r["val_pr_auc"])
            self.assertIsNotNone(r["test_pr_auc"])
            # In all models, val PR-AUC > test PR-AUC due to temporal prevalence shift
            self.assertGreater(r["val_pr_auc"], r["test_pr_auc"])

    def test_format_benchmark_comparison_records_missing_metrics_preserved_as_none(self):
        """6. Verifies missing metrics in candidate performance are preserved as None, never 0."""
        mock_data = {
            "metadata": {},
            "production_logistic_regression_anchor": {
                "model_name": "Anchor LR",
                "locked_threshold": 0.35,
                "validation_metrics": {"pr_auc": 0.60},
                "test_metrics": {}
            },
            "benchmark_candidates": {
                "ModelX": {
                    "family_name": "ModelX",
                    "validation_performance": {"pr_auc": 0.65},
                    "final_test_performance_at_frozen_threshold": {}
                }
            },
            "benchmark_summary": {"validation_winning_model": "ModelX"}
        }
        records = format_benchmark_comparison_records(mock_data)
        self.assertEqual(len(records), 2)
        self.assertIsNone(records[0]["test_pr_auc"])
        self.assertIsNone(records[1]["test_pr_auc"])
        self.assertEqual(records[1]["val_pr_auc"], 0.65)

    def test_prepare_benchmark_chart_records(self):
        """7. Verifies chart records separate Validation and Test into distinct tidy rows."""
        data = load_benchmark_data()
        comp_records = format_benchmark_comparison_records(data)
        chart_records = prepare_benchmark_chart_records(comp_records, metric_base="pr_auc")

        # 4 models * 2 splits = 8 records
        self.assertEqual(len(chart_records), 8)
        splits = {r["split"] for r in chart_records}
        self.assertEqual(splits, {"Validation", "Test (Unseen)"})

        for r in chart_records:
            self.assertIn("model", r)
            self.assertIn("value", r)
            self.assertIsInstance(r["value"], float)

    def test_get_decision_justification(self):
        """8. Verifies extraction of decision justification preserving Logistic Regression."""
        data = load_benchmark_data()
        dec = get_decision_justification(data)
        self.assertEqual(dec["decision_code"], "PRESERVE_PRODUCTION_LOGISTIC_REGRESSION")
        self.assertEqual(dec["validation_winner"], "XGBoost")
        self.assertAlmostEqual(dec["improvement_margin"], 0.0073, places=4)
        self.assertFalse(dec["is_meaningful_improvement"])
        self.assertIn("Logistic Regression", dec["preserved_model"])
        self.assertTrue(len(dec["rationale"]) > 50)


class TestMLBenchmarkViewerApp(unittest.TestCase):
    """
    Headless integration tests for Phase 9.4 ML Benchmark Viewer in Streamlit.
    """

    @classmethod
    def setUpClass(cls):
        if not HAS_APP_TEST:
            raise unittest.SkipTest("Streamlit AppTest is not available.")
        cls.script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "src", "ui", "app.py")
        )

    def test_switch_to_benchmark_viewer_renders_ui(self):
        """1. Verifies switching navigation to ML Benchmark Viewer renders all sections."""
        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.radio(key="dashboard_view_select").set_value("ML Benchmark Viewer").run(timeout=15)

        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.error), 0)

        # Title
        self.assertTrue(any("Benchmark Viewer" in str(t.value) for t in at.title))

        # Decision banner
        all_info = " ".join(str(i.value) for i in at.info)
        self.assertIn("PRESERVE_PRODUCTION_LOGISTIC_REGRESSION", all_info)
        self.assertIn("XGBoost", all_info)

        # 4 Split integrity metrics
        self.assertEqual(len(at.metric), 4)
        metric_labels = [m.label for m in at.metric]
        self.assertIn("Total Mined Commits", metric_labels)
        self.assertIn("Train Split (70%)", metric_labels)
        self.assertIn("Validation Split (15%)", metric_labels)
        self.assertIn("Unseen Test Split (15%)", metric_labels)

        # Model comparison dataframe
        self.assertEqual(len(at.dataframe), 1)

    def test_benchmark_viewer_distinguishes_val_and_test_metrics(self):
        """2. Verifies that the comparison table clearly separates validation from test metrics."""
        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.radio(key="dashboard_view_select").set_value("ML Benchmark Viewer").run(timeout=15)

        df = at.dataframe[0].value
        columns = list(df.columns)
        self.assertIn("Val PR-AUC", columns)
        self.assertIn("Test PR-AUC", columns)
        self.assertIn("Val ROC-AUC", columns)
        self.assertIn("Test ROC-AUC", columns)
        self.assertIn("Val F1", columns)
        self.assertIn("Test F1", columns)

    def test_benchmark_viewer_displays_all_three_candidate_models_and_anchor(self):
        """3. Verifies that all 3 candidate algorithms and the production anchor are displayed."""
        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.radio(key="dashboard_view_select").set_value("ML Benchmark Viewer").run(timeout=15)

        df = at.dataframe[0].value
        models_in_table = list(df["Model"].values)
        self.assertTrue(any("Production LogisticRegression" in m for m in models_in_table))
        self.assertTrue(any("Phase 8 LogisticRegression" in m for m in models_in_table))
        self.assertTrue(any("Phase 8 RandomForest" in m for m in models_in_table))
        self.assertTrue(any("Phase 8 XGBoost" in m for m in models_in_table))

    def test_benchmark_viewer_metric_filter_change(self):
        """4. Verifies changing the metric focus selectbox updates cleanly without error."""
        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.radio(key="dashboard_view_select").set_value("ML Benchmark Viewer").run(timeout=15)

        at.selectbox(key="benchmark_metric_select").select("ROC-AUC").run(timeout=15)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.error), 0)

        at.selectbox(key="benchmark_metric_select").select("F1 Score").run(timeout=15)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.error), 0)

    def test_benchmark_viewer_does_not_mutate_models(self):
        """5. Verifies opening benchmark viewer triggers zero ML training and leaves models untouched."""
        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.radio(key="dashboard_view_select").set_value("ML Benchmark Viewer").run(timeout=15)
        self.assertEqual(len(at.exception), 0)


class TestPRHelpers(unittest.TestCase):
    """Unit tests for Phase 9.5 pure-Python PR helper functions."""

    def test_validate_local_pr_inputs_valid(self):
        """1. Valid local PR inputs pass validation without error."""
        try:
            validate_local_pr_inputs("/path/to/repo", "main", "feature/test")
        except ValueError:
            self.fail("validate_local_pr_inputs raised ValueError unexpectedly on valid inputs.")

    def test_validate_local_pr_inputs_empty_fields(self):
        """2. Empty repo_path, base_ref, or head_ref raises descriptive ValueError."""
        with self.assertRaises(ValueError) as ctx:
            validate_local_pr_inputs("", "main", "feature")
        self.assertIn("Repository path", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            validate_local_pr_inputs("/path", "", "feature")
        self.assertIn("Base reference", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            validate_local_pr_inputs("/path", "main", "   ")
        self.assertIn("Head reference", str(ctx.exception))

    def test_parse_github_pr_identifier_owner_repo(self):
        """3. Parses 'owner/repo' and integer PR number."""
        owner, repo, num = parse_github_pr_identifier("octocat/Hello-World", 42)
        self.assertEqual(owner, "octocat")
        self.assertEqual(repo, "Hello-World")
        self.assertEqual(num, 42)

    def test_parse_github_pr_identifier_url(self):
        """4. Parses full GitHub PR URL."""
        owner, repo, num = parse_github_pr_identifier("https://github.com/bottlepy/bottle/pull/123")
        self.assertEqual(owner, "bottlepy")
        self.assertEqual(repo, "bottle")
        self.assertEqual(num, 123)

    def test_parse_github_pr_identifier_invalid(self):
        """5. Rejects invalid repository formats or non-positive PR numbers."""
        with self.assertRaises(ValueError):
            parse_github_pr_identifier("invalid-no-slash", 1)
        with self.assertRaises(ValueError):
            parse_github_pr_identifier("owner/repo", -5)
        with self.assertRaises(ValueError):
            parse_github_pr_identifier("owner/repo", "not-a-number")
        with self.assertRaises(ValueError):
            parse_github_pr_identifier("")

    def test_is_github_app_configured(self):
        """6. Verifies detection of GitHub App configuration without secret leakage."""
        with unittest.mock.patch.dict(os.environ, {
            "GITHUB_APP_ID": "12345",
            "GITHUB_INSTALLATION_ID": "67890",
            "GITHUB_APP_PRIVATE_KEY": "fake_key_content"
        }, clear=True):
            self.assertTrue(is_github_app_configured())

        with unittest.mock.patch.dict(os.environ, {
            "GITHUB_APP_ID": "",
            "GITHUB_INSTALLATION_ID": "",
            "GITHUB_APP_PRIVATE_KEY": ""
        }, clear=True):
            self.assertFalse(is_github_app_configured())

    def test_get_pr_risk_summary_stats(self):
        """7. Computes accurate summary counts without fabricating a PR-wide risk score."""
        mock_pred_1 = unittest.mock.MagicMock(spec=PredictionResult)
        mock_pred_1.raw_probability = 0.75
        mock_pred_1.is_above_threshold = True

        mock_pred_2 = unittest.mock.MagicMock(spec=PredictionResult)
        mock_pred_2.raw_probability = 0.20
        mock_pred_2.is_above_threshold = False

        commits = [
            PRCommitPrediction(
                commit_hash="abc1234",
                full_hash="abc1234" + "0" * 33,
                author="Alice",
                timestamp="2026-10-10T12:00:00Z",
                commit_message="Fix issue 1",
                is_merge_commit=False,
                prediction=mock_pred_1,
            ),
            PRCommitPrediction(
                commit_hash="def5678",
                full_hash="def5678" + "0" * 33,
                author="Bob",
                timestamp="2026-10-10T13:00:00Z",
                commit_message="Refactor logic",
                is_merge_commit=False,
                prediction=mock_pred_2,
            ),
            PRCommitPrediction(
                commit_hash="mrg9999",
                full_hash="mrg9999" + "0" * 33,
                author="Charlie",
                timestamp="2026-10-10T14:00:00Z",
                commit_message="Merge branch 'main'",
                is_merge_commit=True,
                prediction=None,
            ),
            PRCommitPrediction(
                commit_hash="skp0000",
                full_hash="skp0000" + "0" * 33,
                author="Dave",
                timestamp="2026-10-10T15:00:00Z",
                commit_message="Empty commit",
                is_merge_commit=False,
                prediction=None,
                error="Empty commit skipped",
            ),
        ]

        stats = get_pr_risk_summary_stats(commits, threshold=0.35)
        self.assertEqual(stats["total_commits"], 4)
        self.assertEqual(stats["analyzed_commits"], 2)
        self.assertEqual(stats["defect_prone_commits"], 1)
        self.assertEqual(stats["standard_risk_commits"], 1)
        self.assertEqual(stats["merge_commits"], 1)
        self.assertEqual(stats["skipped_commits"], 1)
        self.assertEqual(stats["max_commit_risk_score"], 0.75)
        self.assertEqual(stats["decision_threshold"], 0.35)

        # Confirm that no PR-wide aggregated probability score exists
        self.assertNotIn("pr_risk_score", stats)
        self.assertNotIn("average_probability", stats)

    def test_format_pr_commits_table(self):
        """8. Formats PR commits list into presentation records."""
        mock_pred = unittest.mock.MagicMock(spec=PredictionResult)
        mock_pred.raw_probability = 0.4215
        mock_pred.risk_level = "HIGH"
        mock_pred.is_above_threshold = True

        commits = [
            PRCommitPrediction(
                commit_hash="a1b2c3d",
                full_hash="a1b2c3d" + "0" * 33,
                author="Author One <author@example.com>",
                timestamp="2026-10-10",
                commit_message="Commit headline\nDetails body",
                is_merge_commit=False,
                prediction=mock_pred,
            ),
            PRCommitPrediction(
                commit_hash="m1m2m3m",
                full_hash="m1m2m3m" + "0" * 33,
                author="Author Two",
                timestamp="2026-10-10",
                commit_message="Merge PR #5",
                is_merge_commit=True,
                prediction=None,
            )
        ]

        table = format_pr_commits_table(commits)
        self.assertEqual(len(table), 2)
        self.assertEqual(table[0]["Commit"], "a1b2c3d")
        self.assertEqual(table[0]["Author"], "Author One")
        self.assertEqual(table[0]["Message"], "Commit headline")
        self.assertEqual(table[0]["Risk Score"], "0.4215")
        self.assertIn("HIGH", table[0]["Risk Level"])
        self.assertEqual(table[0]["Status"], "Analyzed")

        self.assertEqual(table[1]["Commit"], "m1m2m3m")
        self.assertEqual(table[1]["Risk Score"], "—")
        self.assertIn("MERGE", table[1]["Risk Level"])

    def test_format_pr_error_domain_mappings(self):
        """9. Maps domain exceptions to sanitized guidance tuples without leaking secrets."""
        title, msg = format_pr_error(InvalidPRRefError("Base branch 'feat' not found"))
        self.assertEqual(title, "Invalid Git Reference")
        self.assertIn("Base branch", msg)

        title, msg = format_pr_error(GitHubConfigError("Missing GITHUB_APP_ID"))
        self.assertEqual(title, "GitHub App Configuration Missing")

        title, msg = format_pr_error(GitHubAuthError("HTTP 401"))
        self.assertEqual(title, "GitHub Authentication Failed")

        title, msg = format_pr_error(GitHubPRNotFoundError("owner", "repo", 42))
        self.assertEqual(title, "GitHub Pull Request Found" if False else "GitHub Pull Request Not Found")

        title, msg = format_pr_error(RepositoryNotFoundError("No such dir"))
        self.assertEqual(title, "Repository Directory Not Found")

        title, msg = format_pr_error(RuntimeError("Unknown crash"))
        self.assertEqual(title, "Analysis Error")


@unittest.skipUnless(HAS_APP_TEST, "streamlit.testing.v1.AppTest is not available")
class TestPullRequestExplorerApp(unittest.TestCase):
    """Headless Streamlit AppTest integration suite for Phase 9.5 Pull Request Explorer."""

    @classmethod
    def setUpClass(cls):
        cls.script_dir = os.path.dirname(os.path.abspath(__file__))
        cls.project_root = os.path.abspath(os.path.join(cls.script_dir, ".."))
        cls.script_path = os.path.join(cls.project_root, "src", "ui", "app.py")

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_pr_app_repo_")
        self.repo_path = os.path.abspath(self.temp_dir)
        self._init_test_git_repo()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _run_git(self, args):
        cmd = ["git"] + args
        env = os.environ.copy()
        env["GIT_AUTHOR_NAME"] = "Test Author"
        env["GIT_AUTHOR_EMAIL"] = "test@example.com"
        env["GIT_COMMITTER_NAME"] = "Test Author"
        env["GIT_COMMITTER_EMAIL"] = "test@example.com"
        return subprocess.run(
            cmd,
            cwd=self.repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

    def _init_test_git_repo(self):
        self._run_git(["init"])
        self._run_git(["config", "user.name", "Test Author"])
        self._run_git(["config", "user.email", "test@example.com"])
        app_file = os.path.join(self.repo_path, "calc.py")
        with open(app_file, "w", encoding="utf-8") as f:
            f.write("def add(a, b):\n    return a + b\n")
        self._run_git(["add", "."])
        self._run_git(["commit", "-m", "Initial commit on main"])
        self._run_git(["branch", "-M", "main"])

        # Create branch 'feature' with 1 commit ahead
        self._run_git(["checkout", "-b", "feature"])
        with open(app_file, "a", encoding="utf-8") as f:
            f.write("def sub(a, b):\n    return a - b\n")
        self._run_git(["add", "."])
        self._run_git(["commit", "-m", "Add subtraction function"])
        self._run_git(["checkout", "main"])

    def test_pr_explorer_initial_render(self):
        """1. Verifies that switching to Pull Request Explorer view displays prompt and zero premature metrics."""
        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.radio(key="dashboard_view_select").set_value("Pull Request Explorer").run(timeout=15)

        self.assertEqual(len(at.exception), 0)
        # Should not display premature metric cards before clicking analyze
        self.assertEqual(len(at.metric), 0)
        # Informational prompt is displayed
        self.assertTrue(any("Select a local Git repository" in info.value for info in at.info))

    def test_pr_explorer_local_mode_invalid_input_shows_error(self):
        """2. Verifies that entering invalid base reference displays error alert."""
        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.radio(key="dashboard_view_select").set_value("Pull Request Explorer").run(timeout=15)

        at.text_input(key="pr_base_ref").set_value("nonexistent_branch_12345").run(timeout=15)
        at.button(key="analyze_pr_button").click().run(timeout=15)

        self.assertEqual(len(at.exception), 0)
        self.assertTrue(len(at.error) > 0)
        self.assertTrue(any("Invalid Git Reference" in err.value for err in at.error))

    def test_pr_explorer_local_mode_successful_analysis(self):
        """3. End-to-end analysis on local Git repo fixture renders summary cards and commits table."""
        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.radio(key="dashboard_view_select").set_value("Pull Request Explorer").run(timeout=15)

        at.text_input(key="pr_local_repo_path").set_value(self.repo_path).run(timeout=15)
        at.text_input(key="pr_base_ref").set_value("main").run(timeout=15)
        at.text_input(key="pr_head_ref").set_value("feature").run(timeout=15)
        at.button(key="analyze_pr_button").click().run(timeout=15)

        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.error), 0)

        # 4 summary metrics rendered
        self.assertEqual(len(at.metric), 4)
        labels = [m.label for m in at.metric]
        self.assertIn("Total Commits", labels)
        self.assertIn("Analyzed Commits", labels)
        self.assertIn("Flagged Defect-Prone", labels)
        self.assertIn("Skipped / Merges", labels)

        # Total Commits is 1
        total_metric = next(m for m in at.metric if m.label == "Total Commits")
        self.assertEqual(total_metric.value, "1")

        # Commits table rendered
        self.assertTrue(len(at.dataframe) >= 1)

    def test_pr_explorer_github_mode_missing_config_shows_error(self):
        """4. GitHub mode without configured credentials safely shows error without crashing."""
        at = AppTest.from_file(self.script_path, default_timeout=15).run()
        at.radio(key="dashboard_view_select").set_value("Pull Request Explorer").run(timeout=15)

        at.radio(key="pr_input_mode").set_value("GitHub Pull Request").run(timeout=15)
        at.text_input(key="gh_repo_input").set_value("octocat/Hello-World").run(timeout=15)
        at.button(key="analyze_pr_button").click().run(timeout=15)

        self.assertEqual(len(at.exception), 0)
        self.assertTrue(len(at.error) > 0)
        self.assertTrue(any("GitHub App Configuration Missing" in err.value for err in at.error))

    def test_pr_explorer_optional_mock_llm_analysis(self):
        """5. Explicit optional cumulative qualitative LLM analysis with mock provider succeeds."""
        at = AppTest.from_file(self.script_path, default_timeout=30).run()
        at.radio(key="dashboard_view_select").set_value("Pull Request Explorer").run(timeout=30)

        at.text_input(key="pr_local_repo_path").set_value(self.repo_path).run(timeout=30)
        at.text_input(key="pr_base_ref").set_value("main").run(timeout=30)
        at.text_input(key="pr_head_ref").set_value("feature").run(timeout=30)
        at.button(key="analyze_pr_button").click().run(timeout=30)

        self.assertEqual(len(at.metric), 4)

        # Trigger qualitative LLM analysis
        at.button(key="run_pr_llm_button").click().run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.error), 0)

        # Metrics remain intact
        self.assertEqual(len(at.metric), 4)

    def test_pr_explorer_llm_failure_preserves_ml_predictions(self):
        """6. Failure during qualitative analysis displays error without corrupting ML predictions."""
        at = AppTest.from_file(self.script_path, default_timeout=30).run()
        at.radio(key="dashboard_view_select").set_value("Pull Request Explorer").run(timeout=30)

        at.text_input(key="pr_local_repo_path").set_value(self.repo_path).run(timeout=30)
        at.text_input(key="pr_base_ref").set_value("main").run(timeout=30)
        at.text_input(key="pr_head_ref").set_value("feature").run(timeout=30)
        at.button(key="analyze_pr_button").click().run(timeout=30)

        self.assertEqual(len(at.metric), 4)

        # Switch provider to openai without key
        with unittest.mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True):
            at.selectbox(key="pr_llm_provider_select").select("openai").run(timeout=30)
            at.button(key="run_pr_llm_button").click().run(timeout=30)

        self.assertEqual(len(at.exception), 0)
        # Error alert rendered for LLM
        self.assertTrue(len(at.error) > 0)
        # ML metrics and dataframe remain intact
        self.assertEqual(len(at.metric), 4)
        self.assertTrue(len(at.dataframe) >= 1)

    def test_pr_explorer_changing_input_invalidates_stale_results(self):
        """7. Modifying PR inputs immediately purges stale PR results."""
        at = AppTest.from_file(self.script_path, default_timeout=30).run()
        at.radio(key="dashboard_view_select").set_value("Pull Request Explorer").run(timeout=30)

        at.text_input(key="pr_local_repo_path").set_value(self.repo_path).run(timeout=30)
        at.text_input(key="pr_base_ref").set_value("main").run(timeout=30)
        at.text_input(key="pr_head_ref").set_value("feature").run(timeout=30)
        at.button(key="analyze_pr_button").click().run(timeout=30)

        self.assertEqual(len(at.metric), 4)

        # Change head reference in sidebar
        at.text_input(key="pr_head_ref").set_value("other_branch").run(timeout=15)

        # Stale results cleared
        self.assertEqual(len(at.metric), 0)
        self.assertTrue(any("Select a local Git repository" in info.value for info in at.info))


if __name__ == "__main__":
    unittest.main()
