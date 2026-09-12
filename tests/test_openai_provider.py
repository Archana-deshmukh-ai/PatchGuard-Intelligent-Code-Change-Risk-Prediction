"""
Unit tests for OpenAIProvider (Phase 6.5).
100% offline tests using unittest.mock — zero network calls, zero real API keys required.
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import openai

from src.analysis.providers import (
    LLMProvider,
    MockLLMProvider,
    OpenAIProvider
)
from src.analysis.exceptions import (
    OpenAIProviderError,
    MissingAPIKeyError
)
from src.analysis.analyzer import LLMCodeAnalyzer
from src.analysis.schema import LLMAnalysisResult
from src.analysis.diff_extractor import (
    CommitDiff, FileDiff, DiffHunk, DiffLine
)

class TestOpenAIProvider(unittest.TestCase):

    def setUp(self):
        self.mock_client = MagicMock()

    def test_1_provider_construction_with_configured_model(self):
        provider = OpenAIProvider(model="gpt-4o", max_retries=3, client=self.mock_client)
        self.assertEqual(provider.model, "gpt-4o")
        self.assertEqual(provider.max_retries, 3)
        self.assertEqual(provider.provider_name, "openai")

    def test_2_missing_api_key_raises_exception(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(MissingAPIKeyError):
                OpenAIProvider()

    def test_3_sends_expected_prompt_and_structured_output_format(self):
        provider = OpenAIProvider(model="gpt-4o-mini", client=self.mock_client)
        
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({"summary": "Test response"})
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        self.mock_client.chat.completions.create.return_value = mock_response

        res = provider.generate_analysis("Test prompt text")

        self.assertEqual(res, json.dumps({"summary": "Test response"}))
        self.mock_client.chat.completions.create.assert_called_once()
        call_kwargs = self.mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "gpt-4o-mini")
        self.assertEqual(call_kwargs["messages"], [{"role": "user", "content": "Test prompt text"}])
        self.assertEqual(call_kwargs["response_format"], {"type": "json_object"})

    def test_4_successful_mocked_response_parsed_by_analyzer(self):
        mock_json_payload = {
            "commit_hash": "mock123",
            "summary": "Fixes authentications edge case",
            "key_changes": ["Added null check"],
            "potential_risk_factors": [
                {
                    "category": "Error Handling",
                    "description": "Null parameter check",
                    "review_priority": "MEDIUM",
                    "file_path": "auth.py",
                    "line_range": "L10",
                    "evidence_snippet": "if not param:"
                }
            ],
            "affected_areas": ["auth"],
            "testing_observations": ["Unit tests added"],
            "recommended_review_actions": [
                {
                    "action": "Review auth edge cases",
                    "target_file": "auth.py"
                }
            ],
            "confidence_notes": "Good diff"
        }

        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(mock_json_payload)
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        self.mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(client=self.mock_client)
        analyzer = LLMCodeAnalyzer(provider=provider)

        diff = CommitDiff(
            full_hash="mock1234567890full",
            short_hash="mock123",
            author="Dev",
            author_email="dev@example.com",
            timestamp="2026-09-12",
            commit_message="Fix auth",
            files_changed=[
                FileDiff(
                    old_path="auth.py",
                    new_path="auth.py",
                    status="MODIFIED",
                    is_binary=False,
                    is_generated=False,
                    additions=1,
                    deletions=0,
                    hunks=[
                        DiffHunk(
                            old_start=10, old_lines=1, new_start=10, new_lines=1,
                            header="@@ -10 +10 @@",
                            lines=[DiffLine(line_type="+", old_lineno=None, new_lineno=10, content="if not param:")]
                        )
                    ]
                )
            ],
            total_additions=1,
            total_deletions=0,
            is_merge_commit=False,
            is_empty_commit=False
        )

        result = analyzer.analyze_commit(diff)
        self.assertIsInstance(result, LLMAnalysisResult)
        self.assertEqual(result.summary, "Fixes authentications edge case")
        self.assertTrue(result.potential_risk_factors[0].evidence_verified)

    def test_5_authentication_failure_handled(self):
        provider = OpenAIProvider(client=self.mock_client)
        mock_response = MagicMock()
        mock_response.status_code = 401
        err = openai.AuthenticationError(
            message="Invalid API Key",
            response=mock_response,
            body={"error": {"message": "Invalid API Key"}}
        )
        self.mock_client.chat.completions.create.side_effect = err

        with self.assertRaises(OpenAIProviderError) as ctx:
            provider.generate_analysis("prompt")
        self.assertIn("authentication failed", str(ctx.exception).lower())

    def test_6_rate_limit_failure_handled(self):
        provider = OpenAIProvider(client=self.mock_client)
        mock_response = MagicMock()
        mock_response.status_code = 429
        err = openai.RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}}
        )
        self.mock_client.chat.completions.create.side_effect = err

        with self.assertRaises(OpenAIProviderError) as ctx:
            provider.generate_analysis("prompt")
        self.assertIn("rate limit exceeded", str(ctx.exception).lower())

    def test_7_timeout_connection_failure_handled(self):
        provider = OpenAIProvider(client=self.mock_client)
        err = openai.APITimeoutError(request=MagicMock())
        self.mock_client.chat.completions.create.side_effect = err

        with self.assertRaises(OpenAIProviderError) as ctx:
            provider.generate_analysis("prompt")
        self.assertIn("network/timeout error", str(ctx.exception).lower())

    def test_8_mock_llm_provider_remains_functional(self):
        mock_prov = MockLLMProvider()
        res = mock_prov.generate_analysis("test")
        parsed = json.loads(res)
        self.assertEqual(parsed["model_provider"], "mock")


if __name__ == "__main__":
    unittest.main()
