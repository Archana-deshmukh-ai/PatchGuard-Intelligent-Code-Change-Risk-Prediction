"""
Unit test suite for Phase 6.4 LLM Analysis Pipeline (PromptBuilder, ResponseParser, EvidenceValidator, LLMCodeAnalyzer).
Updated with Hardening Pass validations: line-range checks, source file prioritization, size metadata, and strict nested parser validation.
"""

import json
import os
import shutil
import tempfile
import unittest
import subprocess

from src.analysis.diff_extractor import (
    CommitDiff, FileDiff, DiffHunk, DiffLine
)
from src.analysis.schema import (
    LLMAnalysisResult, RiskFactor, ReviewAction
)
from src.analysis.providers import MockLLMProvider, LLMProvider
from src.analysis.exceptions import (
    AnalysisError, MalformedLLMResponseError
)
from src.analysis.prompt import PromptBuilder, PromptResult
from src.analysis.parser import ResponseParser
from src.analysis.validator import EvidenceValidator
from src.analysis.analyzer import LLMCodeAnalyzer

from src.engine.schema import PredictionResult, ModelSignals

class TestPromptBuilder(unittest.TestCase):
    def setUp(self):
        self.sample_diff = CommitDiff(
            full_hash="abc1234def567890abc1234def567890abc12345",
            short_hash="abc1234",
            author="Alice Developer",
            author_email="alice@example.com",
            timestamp="2026-09-12T12:00:00Z",
            commit_message="Fix null pointer exception in user auth flow",
            files_changed=[
                FileDiff(
                    old_path="package-lock.json",
                    new_path="package-lock.json",
                    status="MODIFIED",
                    is_binary=False,
                    is_generated=True,
                    additions=100,
                    deletions=50,
                    hunks=[]
                ),
                FileDiff(
                    old_path="src/auth.py",
                    new_path="src/auth.py",
                    status="MODIFIED",
                    is_binary=False,
                    is_generated=False,
                    additions=10,
                    deletions=2,
                    hunks=[
                        DiffHunk(
                            old_start=40,
                            old_lines=5,
                            new_start=40,
                            new_lines=8,
                            header="@@ -40,5 +40,8 @@ def authenticate(user):",
                            lines=[
                                DiffLine(line_type=" ", old_lineno=40, new_lineno=40, content="def authenticate(user):"),
                                DiffLine(line_type="-", old_lineno=41, new_lineno=None, content="    return user.token"),
                                DiffLine(line_type="+", old_lineno=None, new_lineno=41, content="    if user is None:"),
                                DiffLine(line_type="+", old_lineno=None, new_lineno=42, content="        raise ValueError('User cannot be None')"),
                                DiffLine(line_type="+", old_lineno=None, new_lineno=43, content="    return user.token")
                            ]
                        )
                    ]
                )
            ],
            total_additions=110,
            total_deletions=52,
            is_merge_commit=False,
            is_empty_commit=False
        )

        self.sample_prediction = PredictionResult(
            full_hash="abc1234def567890fullhash",
            commit_hash="abc1234",
            repo_path="/path/to/repo",
            raw_probability=0.7521,
            is_above_threshold=True,
            decision_threshold=0.35,
            prediction_label="Elevated Defect Risk",
            risk_level="HIGH",
            presentation_disclaimer="Presentation label only",
            features={
                "lines_added": 110,
                "lines_deleted": 52,
                "code_churn": 162,
                "files_changed": 2,
                "functions_changed": 2,
                "num_directories_touched": 2,
                "is_test_file_modified": 0,
                "avg_lines_changed_per_file": 81.0,
                "max_lines_changed_in_single_file": 150,
                "num_source_files_changed": 1
            },
            model_signals=ModelSignals(
                top_positive_factors={"code_churn": 0.45},
                top_negative_factors={"num_source_files_changed": -0.10},
                log_odds_contributions={"code_churn": 0.45}
            ),
            model_id="logreg-v1",
            model_version="1.0.0",
            feature_schema_version="1.0",
            prediction_timestamp="2026-09-12T12:00:00Z",
            model_artifact_hash="dummy_hash_1",
            scaler_artifact_hash="dummy_hash_2"
        )

    def test_build_prompt_without_prediction_result(self):
        builder = PromptBuilder()
        prompt = builder.build_prompt(self.sample_diff)
        self.assertIn("abc1234", prompt)
        self.assertIn("Fix null pointer exception in user auth flow", prompt)
        self.assertIn("src/auth.py", prompt)
        self.assertIn("Machine learning risk prediction: Not provided", prompt)
        self.assertFalse(builder.last_is_truncated)

    def test_build_prompt_with_prediction_result(self):
        builder = PromptBuilder()
        prompt = builder.build_prompt(self.sample_diff, self.sample_prediction)
        self.assertIn("Predicted Risk Score: 0.7521", prompt)
        self.assertIn("ML Risk Level: HIGH", prompt)
        self.assertIn("code_churn", prompt)
        self.assertIn("NOT a causal explanation of software bugs", prompt)
        self.assertIn("permitted and encouraged to disagree", prompt)

    def test_context_budget_truncation(self):
        builder = PromptBuilder(max_diff_bytes=100)
        res = builder.build(self.sample_diff)
        self.assertTrue(res.is_truncated)
        self.assertIn("[TRUNCATED:", res.prompt_text)

    def test_source_files_prioritized_over_generated_files(self):
        # Budget enough to fit src/auth.py but truncate package-lock.json
        builder = PromptBuilder(max_diff_bytes=350)
        res = builder.build(self.sample_diff)
        # src/auth.py should appear in diff before package-lock.json
        auth_pos = res.prompt_text.find("+++ src/auth.py")
        lock_pos = res.prompt_text.find("+++ package-lock.json")
        self.assertNotEqual(auth_pos, -1, "Source file diff must be present")
        if lock_pos != -1:
            self.assertLess(auth_pos, lock_pos, "Source file diff must precede generated file diff")

    def test_all_changed_files_remain_in_summary_despite_truncation(self):
        builder = PromptBuilder(max_diff_bytes=100)
        res = builder.build(self.sample_diff)
        self.assertTrue(res.is_truncated)
        # Check that file summary section contains ALL changed files
        self.assertIn("- src/auth.py (MODIFIED): +10 -2", res.prompt_text)
        self.assertIn("- package-lock.json (MODIFIED): +100 -50 [GENERATED]", res.prompt_text)

    def test_diff_size_metadata(self):
        builder = PromptBuilder(max_diff_bytes=150)
        res = builder.build(self.sample_diff)
        self.assertGreater(res.original_diff_bytes, 0)
        self.assertGreater(res.included_diff_bytes, 0)
        self.assertLessEqual(res.included_diff_bytes, res.original_diff_bytes)
        self.assertEqual(builder.last_original_diff_bytes, res.original_diff_bytes)
        self.assertEqual(builder.last_included_diff_bytes, res.included_diff_bytes)


class TestResponseParser(unittest.TestCase):
    def setUp(self):
        self.parser = ResponseParser()
        self.valid_payload = {
            "commit_hash": "abc1234",
            "summary": "Fixes null reference error in login.",
            "key_changes": ["Added null check", "Added test case"],
            "potential_risk_factors": [
                {
                    "category": "Error Handling",
                    "description": "Possible unhandled exception if user token is empty",
                    "review_priority": "HIGH",
                    "file_path": "src/auth.py",
                    "line_range": "L40-L45",
                    "evidence_snippet": "if user is None:"
                }
            ],
            "affected_areas": ["auth"],
            "testing_observations": ["Unit tests added"],
            "recommended_review_actions": [
                {
                    "action": "Review auth edge cases",
                    "target_file": "src/auth.py"
                }
            ],
            "confidence_notes": "Clear diff."
        }

    def test_parse_valid_json(self):
        raw_json = json.dumps(self.valid_payload)
        result = self.parser.parse(raw_json)
        self.assertIsInstance(result, LLMAnalysisResult)
        self.assertEqual(result.summary, "Fixes null reference error in login.")
        self.assertEqual(len(result.potential_risk_factors), 1)
        self.assertEqual(result.potential_risk_factors[0].review_priority, "HIGH")

    def test_parse_markdown_code_fences(self):
        raw_text = f"```json\n{json.dumps(self.valid_payload)}\n```"
        result = self.parser.parse(raw_text)
        self.assertEqual(result.summary, "Fixes null reference error in login.")

    def test_parse_empty_response(self):
        with self.assertRaises(MalformedLLMResponseError):
            self.parser.parse("")

    def test_parse_invalid_json_syntax(self):
        with self.assertRaises(MalformedLLMResponseError):
            self.parser.parse("{ invalid json content ...")

    def test_parse_missing_required_field(self):
        invalid_payload = dict(self.valid_payload)
        del invalid_payload["summary"]
        with self.assertRaises(MalformedLLMResponseError):
            self.parser.parse(json.dumps(invalid_payload))

    def test_parse_invalid_review_priority(self):
        invalid_payload = dict(self.valid_payload)
        invalid_payload["potential_risk_factors"] = [
            {
                "category": "Security",
                "description": "Critical security flaw",
                "review_priority": "EXTREME",
                "file_path": "src/auth.py",
                "line_range": None,
                "evidence_snippet": "foo"
            }
        ]
        with self.assertRaises(MalformedLLMResponseError):
            self.parser.parse(json.dumps(invalid_payload))

    def test_malformed_nested_risk_factor_rejected(self):
        invalid_payload = dict(self.valid_payload)
        # Element is a string instead of dict
        invalid_payload["potential_risk_factors"] = ["not a dict risk factor"]
        with self.assertRaises(MalformedLLMResponseError):
            self.parser.parse(json.dumps(invalid_payload))

    def test_malformed_nested_review_action_rejected(self):
        invalid_payload = dict(self.valid_payload)
        # Missing required 'action' string field
        invalid_payload["recommended_review_actions"] = [{"target_file": "src/auth.py"}]
        with self.assertRaises(MalformedLLMResponseError):
            self.parser.parse(json.dumps(invalid_payload))

    def test_parse_override_commit_hash(self):
        raw_json = json.dumps(self.valid_payload)
        result = self.parser.parse(raw_json, expected_commit_hash="expected_hash_999")
        self.assertEqual(result.commit_hash, "expected_hash_999")


class TestEvidenceValidator(unittest.TestCase):
    def setUp(self):
        self.validator = EvidenceValidator()
        self.diff = CommitDiff(
            full_hash="def4567890def4567890def4567890def4567890",
            short_hash="def4567",
            author="Dev",
            author_email="dev@example.com",
            timestamp="2026-09-12T12:00:00Z",
            commit_message="Update logic",
            files_changed=[
                FileDiff(
                    old_path="src/service.py",
                    new_path="src/service.py",
                    status="MODIFIED",
                    is_binary=False,
                    is_generated=False,
                    additions=5,
                    deletions=1,
                    hunks=[
                        DiffHunk(
                            old_start=10, old_lines=3, new_start=10, new_lines=5,
                            header="@@ -10,3 +10,5 @@",
                            lines=[
                                DiffLine(line_type=" ", old_lineno=10, new_lineno=10, content="def process_data(data):"),
                                DiffLine(line_type="-", old_lineno=11, new_lineno=None, content="    old_legacy_check(data)"),
                                DiffLine(line_type="+", old_lineno=None, new_lineno=11, content="    if not data:"),
                                DiffLine(line_type="+", old_lineno=None, new_lineno=12, content="        return False")
                            ]
                        )
                    ]
                )
            ],
            total_additions=5,
            total_deletions=1,
            is_merge_commit=False,
            is_empty_commit=False
        )

    def test_valid_evidence_citation(self):
        rf = RiskFactor(
            category="Logic",
            description="Empty data return check",
            review_priority="MEDIUM",
            file_path="src/service.py",
            line_range="L10-L12",
            evidence_snippet="if not data:",
            evidence_verified=True
        )
        analysis = LLMAnalysisResult(
            commit_hash="def4567",
            summary="Sum",
            key_changes=[],
            potential_risk_factors=[rf],
            affected_areas=[],
            testing_observations=[],
            recommended_review_actions=[],
            confidence_notes="",
            model_provider="mock",
            model_name="test"
        )
        validated = self.validator.validate_analysis(analysis, self.diff)
        self.assertTrue(validated.potential_risk_factors[0].evidence_verified)

    def test_valid_line_range(self):
        rf = RiskFactor(
            category="Logic",
            description="Valid line range check",
            review_priority="LOW",
            file_path="src/service.py",
            line_range="L10-L12",
            evidence_snippet="",
            evidence_verified=True
        )
        self.assertTrue(self.validator.verify_risk_factor(rf, self.diff))

    def test_invalid_line_range_format(self):
        rf = RiskFactor(
            category="Logic",
            description="Unparseable garbage line range",
            review_priority="LOW",
            file_path="src/service.py",
            line_range="invalid_range_format",
            evidence_snippet="",
            evidence_verified=True
        )
        self.assertFalse(self.validator.verify_risk_factor(rf, self.diff))

    def test_line_range_inconsistent_with_cited_diff(self):
        rf = RiskFactor(
            category="Logic",
            description="Line range far outside diff hunks",
            review_priority="HIGH",
            file_path="src/service.py",
            line_range="L900-L950",
            evidence_snippet="",
            evidence_verified=True
        )
        self.assertFalse(self.validator.verify_risk_factor(rf, self.diff))

    def test_evidence_from_deleted_lines_is_handled_intentionally(self):
        # Line 11 was deleted in diff ('-')
        rf = RiskFactor(
            category="Refactoring",
            description="Removed legacy check",
            review_priority="MEDIUM",
            file_path="src/service.py",
            line_range="L11",
            evidence_snippet="old_legacy_check(data)",
            evidence_verified=True
        )
        # Grounded in diff -> verify returns True
        self.assertTrue(self.validator.verify_risk_factor(rf, self.diff))

    def test_invalid_evidence_remains_present_with_evidence_verified_false(self):
        rf_valid = RiskFactor(
            category="Logic",
            description="Valid check",
            review_priority="LOW",
            file_path="src/service.py",
            line_range="L10",
            evidence_snippet="def process_data(data):",
            evidence_verified=True
        )
        rf_invalid = RiskFactor(
            category="Security",
            description="Ungrounded claim in missing file",
            review_priority="HIGH",
            file_path="missing_file.py",
            line_range="L99",
            evidence_snippet="secret_key = 123",
            evidence_verified=True
        )
        analysis = LLMAnalysisResult(
            commit_hash="def4567",
            summary="Sum",
            key_changes=[],
            potential_risk_factors=[rf_valid, rf_invalid],
            affected_areas=[],
            testing_observations=[],
            recommended_review_actions=[],
            confidence_notes="",
            model_provider="mock",
            model_name="test"
        )
        validated = self.validator.validate_analysis(analysis, self.diff)
        self.assertEqual(len(validated.potential_risk_factors), 2, "Invalid risk factor must NOT be deleted")
        self.assertTrue(validated.potential_risk_factors[0].evidence_verified)
        self.assertFalse(validated.potential_risk_factors[1].evidence_verified, "Invalid risk factor must receive evidence_verified=False")


class CustomTestProvider(LLMProvider):
    def __init__(self, response_text: str):
        self.response_text = response_text

    def generate_analysis(self, prompt: str, schema=None) -> str:
        return self.response_text


class TestLLMCodeAnalyzer(unittest.TestCase):
    def setUp(self):
        self.diff = CommitDiff(
            full_hash="789xyz789xyz789xyz789xyz789xyz789xyz",
            short_hash="789xyz",
            author="Dev",
            author_email="dev@example.com",
            timestamp="2026-09-12T12:00:00Z",
            commit_message="Mock commit",
            files_changed=[
                FileDiff(
                    old_path="example.py",
                    new_path="example.py",
                    status="MODIFIED",
                    is_binary=False,
                    is_generated=False,
                    additions=1,
                    deletions=0,
                    hunks=[
                        DiffHunk(
                            old_start=1, old_lines=1, new_start=1, new_lines=1,
                            header="@@ -1 +1 @@",
                            lines=[
                                DiffLine(line_type="+", old_lineno=None, new_lineno=1, content="if param is None: return")
                            ]
                        )
                    ]
                )
            ],
            total_additions=1,
            total_deletions=0,
            is_merge_commit=False,
            is_empty_commit=False
        )

    def test_analyze_commit_with_mock_provider(self):
        analyzer = LLMCodeAnalyzer()
        result = analyzer.analyze_commit(self.diff)
        self.assertIsInstance(result, LLMAnalysisResult)
        self.assertEqual(result.commit_hash, "789xyz")
        self.assertTrue(len(result.potential_risk_factors) > 0)
        self.assertEqual(result.model_provider, "mock")

    def test_analyze_commit_custom_malformed_provider_raises(self):
        malformed_provider = CustomTestProvider("This is not JSON!")
        analyzer = LLMCodeAnalyzer(provider=malformed_provider)
        with self.assertRaises(MalformedLLMResponseError):
            analyzer.analyze_commit(self.diff)

    def test_analyze_repo_commit_real_git_repo(self):
        temp_dir = tempfile.mkdtemp()
        try:
            subprocess.run(["git", "init"], cwd=temp_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["git", "config", "user.name", "TestUser"], cwd=temp_dir, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_dir, check=True)
            
            test_file = os.path.join(temp_dir, "test.py")
            with open(test_file, "w") as f:
                f.write("print('hello world')\n")
            
            subprocess.run(["git", "add", "."], cwd=temp_dir, check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=temp_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            analyzer = LLMCodeAnalyzer()
            result = analyzer.analyze_repo_commit(temp_dir, "HEAD")
            self.assertIsInstance(result, LLMAnalysisResult)
            self.assertIsNotNone(result.commit_hash)
        finally:
            def remove_readonly(func, path, exc_info):
                try:
                    os.chmod(path, 0o777)
                    func(path)
                except Exception:
                    pass
            shutil.rmtree(temp_dir, onerror=remove_readonly)


if __name__ == "__main__":
    unittest.main()
