"""
PatchGuard REST API package.
"""

from .config import APIConfig
from .schemas import (
    HealthResponse,
    ReadyResponse,
    CommitPredictionRequest,
    CommitPredictionResponse,
    PRPredictionRequest,
    PRPredictionResponse,
    PRMetadata,
    PRCommitRecord,
    PRSummaryStats,
    ErrorResponse
)
from .service import (
    PredictionService,
    ServiceNotReadyError,
    ForbiddenRepositoryAccessError,
    AuthenticationError
)
from .app import create_app, app

__all__ = [
    "create_app",
    "app",
    "APIConfig",
    "PredictionService",
    "ServiceNotReadyError",
    "ForbiddenRepositoryAccessError",
    "AuthenticationError",
    "HealthResponse",
    "ReadyResponse",
    "CommitPredictionRequest",
    "CommitPredictionResponse",
    "PRPredictionRequest",
    "PRPredictionResponse",
    "PRMetadata",
    "PRCommitRecord",
    "PRSummaryStats",
    "ErrorResponse"
]
