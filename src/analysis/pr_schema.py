"""
Domain models and schema definitions for Phase 7.1 Pull Request Analysis Core.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from src.engine.schema import PredictionResult
from src.analysis.schema import LLMAnalysisResult

@dataclass(frozen=True)
class PRAnalysisInput:
    """
    Input parameters defining a local Pull Request evaluation target.
    """
    repo_path: str
    base_ref: str
    head_ref: str

@dataclass
class PRCommitPrediction:
    """
    Per-commit historical ML prediction record for a commit introduced by a PR.
    """
    commit_hash: str
    full_hash: str
    author: str
    timestamp: str
    commit_message: str
    is_merge_commit: bool
    prediction: Optional[PredictionResult] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commit_hash": self.commit_hash,
            "full_hash": self.full_hash,
            "author": self.author,
            "timestamp": self.timestamp,
            "commit_message": self.commit_message,
            "is_merge_commit": self.is_merge_commit,
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "error": self.error
        }

@dataclass
class PRAnalysisResult:
    """
    Structured PR-level analysis result combining metadata, per-commit ML predictions,
    and cumulative PR diff LLM qualitative analysis.
    """
    repo_path: str
    base_ref: str
    head_ref: str
    resolved_base_sha: str
    resolved_head_sha: str
    resolved_merge_base_sha: str
    is_empty: bool
    commit_count: int
    commits: List[PRCommitPrediction]
    cumulative_analysis: Optional[LLMAnalysisResult]
    analysis_disclaimer: str = (
        "PatchGuard PR analysis preserves per-commit ML defect predictions and evaluates "
        "the cumulative diff using LLM qualitative analysis. It does NOT generate a fabricated "
        "PR-wide risk probability score."
    )

    def get_summary_stats(self, threshold: float = 0.35) -> Dict[str, Any]:
        """
        Computes summary counts from constituent PRCommitPrediction records.
        Strictly preserves individual commit metrics and does NOT compute
        a fabricated PR-wide probability or risk score.
        """
        total = len(self.commits)
        analyzed = 0
        defect_prone = 0
        standard_risk = 0
        merge_count = 0
        skipped_count = 0
        max_risk_score: Optional[float] = None

        for c in self.commits:
            if c.is_merge_commit:
                merge_count += 1
            elif c.prediction is not None:
                analyzed += 1
                prob = c.prediction.raw_probability
                if max_risk_score is None or prob > max_risk_score:
                    max_risk_score = prob

                if c.prediction.is_above_threshold:
                    defect_prone += 1
                else:
                    standard_risk += 1
            else:
                skipped_count += 1

        return {
            "total_commits": total,
            "analyzed_commits": analyzed,
            "defect_prone_commits": defect_prone,
            "standard_risk_commits": standard_risk,
            "merge_commits": merge_count,
            "skipped_commits": skipped_count,
            "max_commit_risk_score": max_risk_score,
            "decision_threshold": threshold
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_path": self.repo_path,
            "base_ref": self.base_ref,
            "head_ref": self.head_ref,
            "resolved_base_sha": self.resolved_base_sha,
            "resolved_head_sha": self.resolved_head_sha,
            "resolved_merge_base_sha": self.resolved_merge_base_sha,
            "is_empty": self.is_empty,
            "commit_count": self.commit_count,
            "commits": [c.to_dict() for c in self.commits],
            "cumulative_analysis": self.cumulative_analysis.to_dict() if self.cumulative_analysis else None,
            "analysis_disclaimer": self.analysis_disclaimer
        }
