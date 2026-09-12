"""
PatchGuard Code Change Analysis package (Phase 6).
"""

from .diff_extractor import (
    GitDiffExtractor,
    CommitDiff,
    FileDiff,
    DiffHunk,
    DiffLine,
    is_generated_file
)
from .schema import (
    RiskFactor,
    ReviewAction,
    LLMAnalysisResult,
    ALLOWED_REVIEW_PRIORITIES
)
from .providers import (
    LLMProvider,
    MockLLMProvider,
    OpenAIProvider
)
from .exceptions import (
    AnalysisError,
    MalformedLLMResponseError,
    LLMProviderError,
    OpenAIProviderError,
    MissingAPIKeyError
)
from .prompt import (
    PromptBuilder,
    PromptResult
)
from .parser import ResponseParser
from .validator import EvidenceValidator
from .analyzer import LLMCodeAnalyzer

__all__ = [
    "GitDiffExtractor",
    "CommitDiff",
    "FileDiff",
    "DiffHunk",
    "DiffLine",
    "is_generated_file",
    "RiskFactor",
    "ReviewAction",
    "LLMAnalysisResult",
    "ALLOWED_REVIEW_PRIORITIES",
    "LLMProvider",
    "MockLLMProvider",
    "OpenAIProvider",
    "AnalysisError",
    "MalformedLLMResponseError",
    "LLMProviderError",
    "OpenAIProviderError",
    "MissingAPIKeyError",
    "PromptBuilder",
    "PromptResult",
    "ResponseParser",
    "EvidenceValidator",
    "LLMCodeAnalyzer"
]
