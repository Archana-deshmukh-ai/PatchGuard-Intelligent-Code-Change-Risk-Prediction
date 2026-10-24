"""
Pydantic v2 schemas and data transfer objects for PatchGuard REST API.
"""

from typing import Dict, Union, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class HealthResponse(BaseModel):
    """
    Process health / liveness check response.
    Indicates that the HTTP worker process is responsive.
    Does NOT imply that model artifacts are loaded or ready.
    """
    status: str = Field(default="pass", description="Process liveness status")
    service: str = Field(default="patchguard-api", description="Service identifier")
    version: str = Field(default="1.0.0", description="API version")


class ReadyResponse(BaseModel):
    """
    Service readiness probe response.
    Indicates that the production ML model and scaler are loaded in memory,
    provenance hashes match verified constants, and the engine is ready for inference.
    """
    status: str = Field(default="ready", description="Inference readiness status")
    model_id: str = Field(..., description="Production model identifier")
    model_version: str = Field(..., description="Production model version")
    feature_schema_version: str = Field(..., description="Canonical feature schema version")
    decision_threshold: float = Field(..., description="Locked production decision threshold")
    model_artifact_hash: str = Field(..., description="SHA-256 hash of loaded model artifact")
    scaler_artifact_hash: str = Field(..., description="SHA-256 hash of loaded scaler artifact")


class CommitPredictionRequest(BaseModel):
    """
    Request schema for single commit risk prediction.
    Strictly forbids client-supplied threshold overrides in Phase 10.2.
    """
    model_config = ConfigDict(extra="forbid")

    repository: str = Field(
        ...,
        description="Target repository identifier relative to permitted root",
        min_length=1,
        max_length=255,
        pattern=r"^[a-zA-Z0-9_\-\.]+$"
    )
    commit_hash: str = Field(
        default="HEAD",
        description="Git revision or commit hash (e.g. HEAD, branch, tag, full/short SHA)",
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_\-\.\/]+$"
    )


class CommitMetadata(BaseModel):
    """Metadata regarding the evaluated commit."""
    commit_hash: str = Field(..., description="Abbreviated commit hash")
    full_hash: str = Field(..., description="Full 40-character commit SHA")
    repository: str = Field(..., description="Repository identifier")


class PredictionData(BaseModel):
    """Primary ML defect risk prediction and classification."""
    raw_probability: float = Field(..., description="Estimated risk score in [0.0, 1.0]")
    is_above_threshold: bool = Field(..., description="True if score >= decision_threshold")
    decision_threshold: float = Field(..., description="Effective production threshold")
    prediction_label: str = Field(..., description="Classification label")
    risk_level: str = Field(..., description="Presentation risk badge (LOW/MEDIUM/HIGH)")
    presentation_disclaimer: str = Field(..., description="Risk score interpretation disclaimer")


class ModelSignalsData(BaseModel):
    """Quantitative log-odds contributions to the model risk score."""
    top_positive_factors: Dict[str, float] = Field(..., description="Factors pushing score up")
    top_negative_factors: Dict[str, float] = Field(..., description="Factors pushing score down")
    log_odds_contributions: Dict[str, float] = Field(..., description="All feature contributions")


class ProvenanceData(BaseModel):
    """Provenance and audit metadata for the prediction."""
    model_id: str = Field(..., description="Model ID")
    model_version: str = Field(..., description="Model release version")
    feature_schema_version: str = Field(..., description="Canonical feature schema version")
    prediction_timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    model_artifact_hash: str = Field(..., description="SHA-256 hash of production model")
    scaler_artifact_hash: str = Field(..., description="SHA-256 hash of production scaler")


class CommitPredictionResponse(BaseModel):
    """
    Standard response payload for commit risk prediction.
    """
    commit: CommitMetadata
    prediction: PredictionData
    model_signals: ModelSignalsData
    features: Dict[str, Union[int, float]]
    provenance: ProvenanceData


class ErrorDetails(BaseModel):
    """Structured error details."""
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable description of error")
    remediation: str = Field(..., description="Actionable remediation steps")


class ErrorResponse(BaseModel):
    """Standard structured error response format."""
    error: ErrorDetails


class PRPredictionRequest(BaseModel):
    """
    Request schema for Pull Request defect risk prediction.
    Strictly forbids client-supplied threshold overrides.
    """
    model_config = ConfigDict(extra="forbid")

    repository: str = Field(
        ...,
        description="Target repository identifier relative to permitted root",
        min_length=1,
        max_length=255,
        pattern=r"^[a-zA-Z0-9_\-\.]+$"
    )
    base_ref: str = Field(
        ...,
        description="Base Git reference or branch (e.g. 'main', commit SHA)",
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_][a-zA-Z0-9_\-\.\/]*$"
    )
    head_ref: str = Field(
        ...,
        description="Head Git reference or branch (e.g. 'feature-branch', commit SHA)",
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_][a-zA-Z0-9_\-\.\/]*$"
    )


class PRMetadata(BaseModel):
    """Metadata regarding the evaluated Pull Request."""
    repository: str = Field(..., description="Target repository identifier")
    base_ref: str = Field(..., description="User-supplied base Git reference")
    head_ref: str = Field(..., description="User-supplied head Git reference")
    resolved_base_sha: str = Field(..., description="Full 40-character base commit SHA")
    resolved_head_sha: str = Field(..., description="Full 40-character head commit SHA")
    resolved_merge_base_sha: str = Field(..., description="Full 40-character merge-base commit SHA")
    is_empty: bool = Field(..., description="True if no new commits are introduced by PR")
    commit_count: int = Field(..., description="Number of constituent commits introduced by PR")


class PRCommitRecord(BaseModel):
    """Per-commit prediction and provenance within a Pull Request."""
    commit_hash: str = Field(..., description="Abbreviated commit SHA")
    full_hash: str = Field(..., description="Full 40-character commit SHA")
    author: str = Field(..., description="Commit author name")
    timestamp: str = Field(..., description="Commit author timestamp")
    commit_message: str = Field(..., description="Commit message")
    is_merge_commit: bool = Field(..., description="True if commit has multiple parents")
    prediction: Optional[PredictionData] = Field(default=None, description="ML risk prediction if analyzed")
    model_signals: Optional[ModelSignalsData] = Field(default=None, description="Feature signals if analyzed")
    features: Optional[Dict[str, Union[int, float]]] = Field(default=None, description="Canonical features if analyzed")
    error: Optional[str] = Field(default=None, description="Error message if prediction skipped or failed")


class PRSummaryStats(BaseModel):
    """Quantitative summary statistics across PR constituent commits."""
    total_commits: int = Field(..., description="Total constituent commits in PR")
    analyzed_commits: int = Field(..., description="Commits successfully evaluated by ML engine")
    defect_prone_commits: int = Field(..., description="Commits with risk score >= decision threshold")
    standard_risk_commits: int = Field(..., description="Commits with risk score < decision threshold")
    merge_commits: int = Field(..., description="Merge commits excluded from risk prediction")
    skipped_commits: int = Field(..., description="Commits skipped due to errors")
    max_commit_risk_score: Optional[float] = Field(default=None, description="Maximum risk score across analyzed commits")
    decision_threshold: float = Field(..., description="Locked production decision threshold")


class PRPredictionResponse(BaseModel):
    """Standard response payload for Pull Request defect risk prediction."""
    pr: PRMetadata
    summary: PRSummaryStats
    commits: List[PRCommitRecord]
    provenance: ProvenanceData
    analysis_disclaimer: str = Field(
        default=(
            "PatchGuard PR analysis preserves per-commit ML defect predictions and evaluates "
            "constituent commits against the production threshold. It does NOT generate a fabricated "
            "PR-wide risk probability score."
        ),
        description="Governance disclaimer confirming no fabricated aggregate PR probability is computed"
    )
