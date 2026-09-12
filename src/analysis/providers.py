"""
LLM provider abstractions and offline mock implementation for PatchGuard Phase 6.
Zero third-party LLM SDK dependencies, zero network requests, zero API key requirements.
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

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
        # Construct deterministic mock response payload
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
