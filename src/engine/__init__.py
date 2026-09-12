"""
PatchGuard Risk Prediction Engine package.
"""

from .predictor import RiskPredictionEngine
from .schema import PredictionResult, ModelSignals, FEATURE_COLUMNS
from .exceptions import (
    PatchGuardError,
    RepositoryError,
    RepositoryNotFoundError,
    InvalidGitRepositoryError,
    CommitError,
    CommitNotFoundError,
    UnsupportedCommitTypeError,
    EmptyCommitError,
    FeatureExtractionError,
    GitExtractionError,
    ModelError,
    ModelArtifactNotFoundError,
    ScalerArtifactNotFoundError,
    CorruptedArtifactError,
    IncompatibleSchemaError,
    InvalidFeatureValueError
)

__all__ = [
    "RiskPredictionEngine",
    "PredictionResult",
    "ModelSignals",
    "FEATURE_COLUMNS",
    "PatchGuardError",
    "RepositoryError",
    "RepositoryNotFoundError",
    "InvalidGitRepositoryError",
    "CommitError",
    "CommitNotFoundError",
    "UnsupportedCommitTypeError",
    "EmptyCommitError",
    "FeatureExtractionError",
    "GitExtractionError",
    "ModelError",
    "ModelArtifactNotFoundError",
    "ScalerArtifactNotFoundError",
    "CorruptedArtifactError",
    "IncompatibleSchemaError",
    "InvalidFeatureValueError"
]
