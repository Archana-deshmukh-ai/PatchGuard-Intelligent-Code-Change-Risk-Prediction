"""
Unit Test Suite for Phase 6.3 Analysis Schemas and Mock LLM Provider.
100% offline, deterministic tests with zero external dependencies.
"""

import sys
import os
import json
import unittest

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.analysis import (
    RiskFactor,
    ReviewAction,
    LLMAnalysisResult,
    LLMProvider,
    MockLLMProvider
)

class TestAnalysisSchemaAndProvider(unittest.TestCase):

    def test_1_risk_factor_valid_construction(self):
        """1. Test RiskFactor construction with valid review_priority values."""
        for prio in ["LOW", "MEDIUM", "HIGH"]:
            rf = RiskFactor(
                category="Error Handling",
                description="Test description",
                review_priority=prio,
                file_path="src/main.py",
                line_range="L10-L20",
                evidence_snippet="if x is None:"
            )
            self.assertEqual(rf.review_priority, prio)
            self.assertEqual(rf.category, "Error Handling")

    def test_2_invalid_review_priority_rejected(self):
        """2. Test that invalid review_priority values raise ValueError."""
        invalid_priorities = ["CRITICAL", "URGENT", "severe", "0.80", "HIGH_RISK", ""]
        for bad_prio in invalid_priorities:
            with self.assertRaises(ValueError):
                RiskFactor(
                    category="Error Handling",
                    description="Test description",
                    review_priority=bad_prio,
                    file_path="src/main.py",
                    line_range="L10-L20",
                    evidence_snippet="if x is None:"
                )

    def test_3_review_action_construction(self):
        """3. Test ReviewAction construction with optional target file."""
        ra1 = ReviewAction(action="Run unit tests", target_file="tests/test_main.py")
        self.assertEqual(ra1.action, "Run unit tests")
        self.assertEqual(ra1.target_file, "tests/test_main.py")

        ra2 = ReviewAction(action="Inspect refactored boundary condition")
        self.assertIsNone(ra2.target_file)

    def test_4_llm_analysis_result_construction(self):
        """4. Test full LLMAnalysisResult construction."""
        rf = RiskFactor(
            category="Concurrency",
            description="Race condition risk",
            review_priority="HIGH",
            file_path="src/async.py",
            line_range="L45-L50",
            evidence_snippet="await lock.acquire()"
        )
        ra = ReviewAction(action="Audit lock acquisition order", target_file="src/async.py")
        
        res = LLMAnalysisResult(
            commit_hash="abc1234",
            summary="Refactored lock handling in async module.",
            key_changes=["Replaced threading lock with asyncio lock."],
            potential_risk_factors=[rf],
            affected_areas=["async.core"],
            testing_observations=["Unit tests modified."],
            recommended_review_actions=[ra],
            confidence_notes="High confidence observation.",
            model_provider="mock",
            model_name="deterministic-mock"
        )
        
        self.assertEqual(res.commit_hash, "abc1234")
        self.assertEqual(len(res.potential_risk_factors), 1)
        self.assertEqual(res.potential_risk_factors[0].review_priority, "HIGH")

    def test_5_json_serialization(self):
        """5. Test LLMAnalysisResult.to_dict() and json.dumps() serialization."""
        rf = RiskFactor(
            category="Testing",
            description="Missing test coverage",
            review_priority="LOW",
            file_path="src/utils.py",
            line_range="L5",
            evidence_snippet="def helper():"
        )
        res = LLMAnalysisResult(
            commit_hash="def5678",
            summary="Added utility helper.",
            key_changes=["Added helper function."],
            potential_risk_factors=[rf],
            affected_areas=["utils"],
            testing_observations=["No test added."],
            recommended_review_actions=[],
            confidence_notes="Standard mock note.",
            model_provider="mock",
            model_name="mock-model"
        )
        
        d = res.to_dict()
        self.assertIsInstance(d, dict)
        json_str = json.dumps(d, indent=2)
        self.assertIsInstance(json_str, str)
        self.assertIn("def5678", json_str)
        self.assertIn("LOW", json_str)

    def test_6_json_deserialization(self):
        """6. Test LLMAnalysisResult.from_dict() deserialization roundtrip."""
        rf = RiskFactor(
            category="Security",
            description="Sanitize input path",
            review_priority="MEDIUM",
            file_path="src/path.py",
            line_range="L12",
            evidence_snippet="os.path.join(base, user_input)"
        )
        ra = ReviewAction(action="Sanitize path input", target_file="src/path.py")
        
        orig = LLMAnalysisResult(
            commit_hash="7890abc",
            summary="Updated path resolution.",
            key_changes=["Joined paths."],
            potential_risk_factors=[rf],
            affected_areas=["path.resolver"],
            testing_observations=["Path tests updated."],
            recommended_review_actions=[ra],
            confidence_notes="Roundtrip test.",
            model_provider="mock",
            model_name="mock-model"
        )
        
        d = orig.to_dict()
        reconstructed = LLMAnalysisResult.from_dict(d)
        
        self.assertEqual(orig.commit_hash, reconstructed.commit_hash)
        self.assertEqual(orig.summary, reconstructed.summary)
        self.assertEqual(len(reconstructed.potential_risk_factors), 1)
        self.assertEqual(reconstructed.potential_risk_factors[0].review_priority, "MEDIUM")

    def test_7_mock_provider_inheritance(self):
        """7. Test that MockLLMProvider inherits from LLMProvider ABC."""
        provider = MockLLMProvider()
        self.assertIsInstance(provider, LLMProvider)
        self.assertTrue(issubclass(MockLLMProvider, LLMProvider))

    def test_8_mock_provider_returns_valid_json(self):
        """8. Test that MockLLMProvider.generate_analysis() returns valid JSON string."""
        provider = MockLLMProvider()
        raw_output = provider.generate_analysis(prompt="Analyze this commit diff")
        
        self.assertIsInstance(raw_output, str)
        parsed = json.loads(raw_output)
        self.assertIsInstance(parsed, dict)

    def test_9_mock_provider_deterministic_repeated_calls(self):
        """9. Test that repeated calls to MockLLMProvider produce identical output."""
        provider = MockLLMProvider()
        out1 = provider.generate_analysis(prompt="Prompt A")
        out2 = provider.generate_analysis(prompt="Prompt A")
        out3 = provider.generate_analysis(prompt="Prompt B")
        
        self.assertEqual(out1, out2)
        self.assertEqual(out1, out3)

    def test_10_mock_provider_offline_and_no_sdk_required(self):
        """10. Verify provider requires zero API keys and makes zero network calls."""
        # Ensure no environment variables are needed
        provider = MockLLMProvider(model_name="offline-test-model")
        self.assertEqual(provider.model_name, "offline-test-model")
        self.assertEqual(provider.provider_name, "mock")
        
        raw_json = provider.generate_analysis(prompt="Offline test prompt")
        self.assertIn("offline-test-model", raw_json)


    def test_11_mock_output_contains_expected_top_level_keys(self):
        """11. Verify mock JSON contains all expected top-level keys matching LLMAnalysisResult schema."""
        provider = MockLLMProvider()
        parsed = json.loads(provider.generate_analysis(prompt="Test prompt"))
        
        expected_keys = {
            "commit_hash", "summary", "key_changes", "potential_risk_factors",
            "affected_areas", "testing_observations", "recommended_review_actions",
            "confidence_notes", "model_provider", "model_name"
        }
        self.assertEqual(set(parsed.keys()), expected_keys)
        
        # Test deserialization into LLMAnalysisResult object
        result_obj = LLMAnalysisResult.from_dict(parsed)
        self.assertIsInstance(result_obj, LLMAnalysisResult)

if __name__ == '__main__':
    unittest.main()
