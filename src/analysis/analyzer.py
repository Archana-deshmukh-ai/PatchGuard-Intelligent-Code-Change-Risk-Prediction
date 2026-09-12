"""
LLM Code Analysis orchestrator for Phase 6.
Combines GitDiffExtractor, PromptBuilder, LLMProvider, ResponseParser, and EvidenceValidator.
"""

from typing import Optional
from src.analysis.diff_extractor import GitDiffExtractor, CommitDiff
from src.analysis.schema import LLMAnalysisResult
from src.analysis.providers import LLMProvider, MockLLMProvider
from src.analysis.prompt import PromptBuilder, PromptResult
from src.analysis.parser import ResponseParser
from src.analysis.validator import EvidenceValidator
from src.engine.schema import PredictionResult

class LLMCodeAnalyzer:
    """
    Orchestrates the full offline qualitative analysis pipeline for git code changes.
    Connects structured diff extraction, prompt context budgeting, LLM generation,
    response parsing, and evidence citation validation.
    """
    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        response_parser: Optional[ResponseParser] = None,
        evidence_validator: Optional[EvidenceValidator] = None
    ):
        self.provider = provider if provider is not None else MockLLMProvider()
        self.prompt_builder = prompt_builder if prompt_builder is not None else PromptBuilder()
        self.response_parser = response_parser if response_parser is not None else ResponseParser()
        self.evidence_validator = evidence_validator if evidence_validator is not None else EvidenceValidator()
        self.last_prompt_result: Optional[PromptResult] = None

    def analyze_commit(
        self,
        commit_diff: CommitDiff,
        prediction_result: Optional[PredictionResult] = None
    ) -> LLMAnalysisResult:
        """
        Runs analysis pipeline on a pre-extracted CommitDiff.
        """
        prompt_res = self.prompt_builder.build(commit_diff, prediction_result)
        self.last_prompt_result = prompt_res

        raw_response = self.provider.generate_analysis(prompt_res.prompt_text)

        parsed_analysis = self.response_parser.parse(
            raw_text=raw_response,
            expected_commit_hash=commit_diff.short_hash or commit_diff.full_hash
        )

        validated_analysis = self.evidence_validator.validate_analysis(
            analysis=parsed_analysis,
            commit_diff=commit_diff
        )

        return validated_analysis

    def analyze_repo_commit(
        self,
        repo_path: str,
        commit_ref: str = "HEAD",
        prediction_result: Optional[PredictionResult] = None
    ) -> LLMAnalysisResult:
        """
        Extracts diff from git repository and runs full qualitative analysis pipeline.
        """
        extractor = GitDiffExtractor(repo_path)
        commit_diff = extractor.extract_commit_diff(commit_ref)
        return self.analyze_commit(commit_diff, prediction_result)
