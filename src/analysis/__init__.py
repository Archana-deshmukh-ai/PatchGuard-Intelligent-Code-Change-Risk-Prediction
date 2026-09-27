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
    MissingAPIKeyError,
    PRAnalysisError,
    InvalidPRRefError
)
from .prompt import (
    PromptBuilder,
    PromptResult
)
from .parser import ResponseParser
from .validator import EvidenceValidator
from .analyzer import LLMCodeAnalyzer
from .pr_schema import (
    PRAnalysisInput,
    PRCommitPrediction,
    PRAnalysisResult
)
from .pr_analyzer import PRAnalyzer

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
    "PRAnalysisError",
    "InvalidPRRefError",
    "PromptBuilder",
    "PromptResult",
    "ResponseParser",
    "EvidenceValidator",
    "LLMCodeAnalyzer",
    "PRAnalysisInput",
    "PRCommitPrediction",
    "PRAnalysisResult",
    "PRAnalyzer"
]

