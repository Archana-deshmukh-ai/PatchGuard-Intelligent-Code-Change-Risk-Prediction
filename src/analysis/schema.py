"""
Data model schemas for Phase 6 LLM Code-Change Analysis.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

ALLOWED_REVIEW_PRIORITIES = {"LOW", "MEDIUM", "HIGH"}

@dataclass(frozen=True)
class RiskFactor:
    """
    Represents an observation or hypothesis from LLM analysis.
    NOTE: 'review_priority' indicates developer review priority (LOW/MEDIUM/HIGH) 
    and does NOT represent statistical bug probability or model risk severity.
    """
    category: str
    description: str
    review_priority: str  # "LOW", "MEDIUM", "HIGH"
    file_path: str
    line_range: Optional[str]
    evidence_snippet: str
    evidence_verified: bool = True

    def __post_init__(self):
        if self.review_priority not in ALLOWED_REVIEW_PRIORITIES:
            raise ValueError(
                f"Invalid review_priority: '{self.review_priority}'. Must be one of {sorted(ALLOWED_REVIEW_PRIORITIES)}"
            )


@dataclass(frozen=True)
class ReviewAction:
    """
    Represents a recommended developer review or testing action.
    """
    action: str
    target_file: Optional[str] = None

@dataclass(frozen=True)
class LLMAnalysisResult:
    """
    Structured, LLM-agnostic qualitative code change analysis result.
    """
    commit_hash: str
    summary: str
    key_changes: List[str]
    potential_risk_factors: List[RiskFactor]
    affected_areas: List[str]
    testing_observations: List[str]
    recommended_review_actions: List[ReviewAction]
    confidence_notes: str
    model_provider: str
    model_name: str

    def to_dict(self) -> Dict[str, Any]:
        """Serializes LLMAnalysisResult to a standard dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMAnalysisResult":
        """Deserializes a dictionary into a structured LLMAnalysisResult object."""
        risk_factors = [
            RiskFactor(**rf) if isinstance(rf, dict) else rf
            for rf in data.get("potential_risk_factors", [])
        ]
        review_actions = [
            ReviewAction(**ra) if isinstance(ra, dict) else ra
            for ra in data.get("recommended_review_actions", [])
        ]
        return cls(
            commit_hash=data["commit_hash"],
            summary=data["summary"],
            key_changes=list(data.get("key_changes", [])),
            potential_risk_factors=risk_factors,
            affected_areas=list(data.get("affected_areas", [])),
            testing_observations=list(data.get("testing_observations", [])),
            recommended_review_actions=review_actions,
            confidence_notes=data.get("confidence_notes", ""),
            model_provider=data.get("model_provider", "unknown"),
            model_name=data.get("model_name", "unknown")
        )
