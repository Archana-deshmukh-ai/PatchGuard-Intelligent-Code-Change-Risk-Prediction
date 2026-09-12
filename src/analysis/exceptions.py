"""
Custom exception hierarchy for Phase 6 LLM Code-Change Analysis.
"""

from src.engine.exceptions import PatchGuardError

class AnalysisError(PatchGuardError):
    """Base exception for code analysis errors."""
    pass

class MalformedLLMResponseError(AnalysisError):
    """Raised when the LLM produces output that is not valid JSON or violates schema requirements."""
    pass

class LLMProviderError(AnalysisError):
    """Base exception for LLM provider errors."""
    pass

class OpenAIProviderError(LLMProviderError):
    """Raised when an OpenAI API or client operation fails."""
    pass

class MissingAPIKeyError(OpenAIProviderError):
    """Raised when the OPENAI_API_KEY environment variable is not set."""
    pass
