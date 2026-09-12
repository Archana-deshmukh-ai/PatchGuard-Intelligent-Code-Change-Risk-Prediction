"""
LLM provider abstractions and implementations for PatchGuard Phase 6.
Supports deterministic offline MockLLMProvider and real OpenAIProvider.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from src.analysis.exceptions import (
    OpenAIProviderError,
    MissingAPIKeyError
)

class LLMProvider(ABC):
    """
    Minimal abstract interface for LLM providers.
    Boundary interface: takes a prompt and returns raw structured JSON response text.
    """
    @abstractmethod
    def generate_analysis(self, prompt: str, schema: Optional[Dict[str, Any]] = None) -> str:
        """
        Executes prompt completion and returns raw JSON text conforming to analysis schema.
        """
        pass

class MockLLMProvider(LLMProvider):
    """
    Deterministic, offline, dependency-free mock LLM provider for testing the Phase 6 pipeline.
    Produces predictable JSON string responses matching the LLMAnalysisResult schema.
    """
    def __init__(self, model_name: str = "deterministic-mock"):
        self.model_name = model_name
        self.provider_name = "mock"

    def generate_analysis(self, prompt: str, schema: Optional[Dict[str, Any]] = None) -> str:
        """
        Returns a deterministic JSON string response conforming to LLMAnalysisResult structure.
        """
        mock_payload = {
            "commit_hash": "mock_hash",
            "summary": "Mock commit summary: Qualitative code change analysis for test verification.",
            "key_changes": [
                "Mock change 1: Updated logic in target function.",
                "Mock change 2: Added error boundary check."
            ],
            "potential_risk_factors": [
                {
                    "category": "Error Handling",
                    "description": "Mock observation: verify edge case handling when parameter is None.",
                    "review_priority": "MEDIUM",
                    "file_path": "example.py",
                    "line_range": "L10-L15",
                    "evidence_snippet": "if param is None: return"
                }
            ],
            "affected_areas": [
                "core.module"
            ],
            "testing_observations": [
                "Mock test observation: unit tests present for modified path."
            ],
            "recommended_review_actions": [
                {
                    "action": "Verify boundary condition in example.py",
                    "target_file": "example.py"
                }
            ],
            "confidence_notes": "Deterministic offline mock output for pipeline validation.",
            "model_provider": self.provider_name,
            "model_name": self.model_name
        }
        return json.dumps(mock_payload, indent=2)


class OpenAIProvider(LLMProvider):
    """
    Real OpenAI LLM Provider implementation using the current official OpenAI Python SDK.
    Sends code diff prompts to OpenAI Chat Completions API requesting structured JSON responses.
    PRIVACY & SECURITY BOUNDARY:
    This provider sends code diff prompt contents to an external third-party API (api.openai.com).
    API key MUST be supplied via the OPENAI_API_KEY environment variable.
    """
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        client: Optional[Any] = None
    ):
        self.model = model
        self.provider_name = "openai"
        self.timeout = timeout
        self.max_retries = max_retries

        if client is not None:
            self._client = client
        else:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key or not key.strip():
                raise MissingAPIKeyError(
                    "OPENAI_API_KEY environment variable is missing or empty. "
                    "Set OPENAI_API_KEY to use the live OpenAIProvider."
                )

            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=key,
                    timeout=timeout,
                    max_retries=max_retries
                )
            except Exception as e:
                raise OpenAIProviderError(f"Failed to initialize OpenAI client: {e}") from e

    def generate_analysis(self, prompt: str, schema: Optional[Dict[str, Any]] = None) -> str:
        """
        Executes Chat Completion request to OpenAI using structured JSON output mode.
        Returns raw JSON response text conforming to LLMAnalysisResult schema.
        """
        import openai

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "timeout": self.timeout
        }

        try:
            response = self._client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise OpenAIProviderError("OpenAI API returned an empty completion response.")
            return content.strip()
        except openai.AuthenticationError as e:
            raise OpenAIProviderError(f"OpenAI authentication failed: {e.message if hasattr(e, 'message') else e}") from e
        except openai.RateLimitError as e:
            raise OpenAIProviderError(f"OpenAI rate limit exceeded: {e.message if hasattr(e, 'message') else e}") from e
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            raise OpenAIProviderError(f"OpenAI network/timeout error: {e.message if hasattr(e, 'message') else e}") from e
        except openai.OpenAIError as e:
            raise OpenAIProviderError(f"OpenAI API request failed: {e.message if hasattr(e, 'message') else e}") from e
        except OpenAIProviderError:
            raise
        except Exception as e:
            raise OpenAIProviderError(f"Unexpected error during OpenAI generation: {e}") from e
