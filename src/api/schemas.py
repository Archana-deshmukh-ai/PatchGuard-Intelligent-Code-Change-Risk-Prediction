"""
Pydantic v2 schemas and data transfer objects for PatchGuard REST API.
"""

from typing import Dict, Union, Any
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
