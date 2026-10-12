"""
Prediction service business logic and repository access validation for PatchGuard REST API.
"""

import os
from typing import Optional, Dict, Any

from src.engine import RiskPredictionEngine, PredictionResult
from src.engine.exceptions import (
    PatchGuardError,
    RepositoryNotFoundError,
    ModelError
)
from .config import APIConfig
from .schemas import (
    ReadyResponse,
    CommitPredictionResponse,
    CommitMetadata,
    PredictionData,
    ModelSignalsData,
    ProvenanceData
)

EXPECTED_MODEL_HASH = "a32b30dff63b3a85c3fd9d9983921d5f5c5c5629acc9c206f77759c46f60f387"
EXPECTED_SCALER_HASH = "c8e65f768dab97acf8df573da60f6b1c0e9232ebf2173677f88e3efa5c479b49"
PRODUCTION_THRESHOLD = 0.35


class ServiceNotReadyError(PatchGuardError):
    """Raised when the prediction engine is not initialized or fails readiness checks."""
    pass


class ForbiddenRepositoryAccessError(PatchGuardError):
    """Raised when a client attempts to access a path outside the permitted repository root."""
    pass


class AuthenticationError(PatchGuardError):
    """Raised when client authentication is missing or invalid."""
    pass


class PredictionService:
    """
    Wraps RiskPredictionEngine with repository allowlist security,
    readiness probing, and response serialization.
    """

    def __init__(
        self,
        config: Optional[APIConfig] = None,
        engine: Optional[RiskPredictionEngine] = None
    ):
        self.config = config if config is not None else APIConfig.from_env()
        self._engine = engine
        self._engine_init_error: Optional[str] = None

        if self._engine is None:
            try:
                self._engine = RiskPredictionEngine(
                    models_dir=self.config.models_dir,
                    default_threshold=PRODUCTION_THRESHOLD
                )
            except Exception as e:
                self._engine_init_error = str(e)
                self._engine = None

    @property
    def engine(self) -> RiskPredictionEngine:
        if self._engine is None:
            detail = self._engine_init_error or "Risk prediction engine is not initialized."
            raise ServiceNotReadyError(f"Prediction engine not ready: {detail}")
        return self._engine

    def check_readiness(self) -> ReadyResponse:
        """
        Validates engine readiness and artifact integrity.
        Raises ServiceNotReadyError if artifacts are missing, corrupted, or have invalid hashes.
        """
        if self._engine is None:
            detail = self._engine_init_error or "Prediction engine could not be loaded."
            raise ServiceNotReadyError(f"Engine unready: {detail}")

        eng = self._engine
        # Verify hashes match expected production constants
        if eng.model_artifact_hash.lower() != EXPECTED_MODEL_HASH.lower():
            raise ServiceNotReadyError(
                f"Model artifact hash mismatch: loaded {eng.model_artifact_hash[:16]}..., expected {EXPECTED_MODEL_HASH[:16]}..."
            )

        if eng.scaler_artifact_hash.lower() != EXPECTED_SCALER_HASH.lower():
            raise ServiceNotReadyError(
                f"Scaler artifact hash mismatch: loaded {eng.scaler_artifact_hash[:16]}..., expected {EXPECTED_SCALER_HASH[:16]}..."
            )

        # Validate configured repository root exists and is an accessible directory
        repos_root = self.config.allowed_repos_root
        if not repos_root or not os.path.exists(repos_root) or not os.path.isdir(repos_root):
            raise ServiceNotReadyError(
                "Configured repository root does not exist or is not an accessible directory."
            )

        return ReadyResponse(
            status="ready",
            model_id=eng.MODEL_ID,
            model_version=eng.MODEL_VERSION,
            feature_schema_version=eng.FEATURE_SCHEMA_VERSION,
            decision_threshold=PRODUCTION_THRESHOLD,
            model_artifact_hash=eng.model_artifact_hash,
            scaler_artifact_hash=eng.scaler_artifact_hash
        )

    def resolve_repository_path(self, repository_id: str) -> str:
        """
        Resolves repository identifier within permitted root and guards against path traversal.
        """
        repo_clean = repository_id.strip()
        if not repo_clean:
            raise ForbiddenRepositoryAccessError("Repository identifier must not be empty.")

        # Reject path separators and traversal tokens
        if "/" in repo_clean or "\\" in repo_clean or ".." in repo_clean or "\0" in repo_clean:
            raise ForbiddenRepositoryAccessError(
                "Invalid repository identifier: path separators and directory traversal sequences are prohibited."
            )

        if repo_clean == "." or repo_clean == "..":
            raise ForbiddenRepositoryAccessError(
                "Access denied: relative dot references are prohibited."
            )

        canonical_root = os.path.realpath(self.config.allowed_repos_root)
        target_path = os.path.realpath(os.path.join(canonical_root, repo_clean))

        # Ensure target_path is strictly a child directory within canonical_root (including symlink resolution)
        if not target_path.startswith(canonical_root + os.sep):
            raise ForbiddenRepositoryAccessError(
                "Access denied: target path escapes permitted repository workspace."
            )

        if not os.path.exists(target_path):
            raise RepositoryNotFoundError(
                f"Repository '{repo_clean}' does not exist in permitted directory."
            )

        return target_path

    def predict_commit(self, repository: str, commit_hash: str = "HEAD") -> CommitPredictionResponse:
        """
        Executes commit risk prediction using production engine at locked threshold 0.35.
        """
        resolved_repo_path = self.resolve_repository_path(repository)

        # Execute prediction through core engine
        result: PredictionResult = self.engine.predict(
            repo_path=resolved_repo_path,
            commit_hash=commit_hash,
            threshold=PRODUCTION_THRESHOLD
        )

        return CommitPredictionResponse(
            commit=CommitMetadata(
                commit_hash=result.commit_hash,
                full_hash=result.full_hash,
                repository=repository
            ),
            prediction=PredictionData(
                raw_probability=result.raw_probability,
                is_above_threshold=result.is_above_threshold,
                decision_threshold=result.decision_threshold,
                prediction_label=result.prediction_label,
                risk_level=result.risk_level,
                presentation_disclaimer=result.presentation_disclaimer
            ),
            model_signals=ModelSignalsData(
                top_positive_factors=result.model_signals.top_positive_factors,
                top_negative_factors=result.model_signals.top_negative_factors,
                log_odds_contributions=result.model_signals.log_odds_contributions
            ),
            features=result.features,
            provenance=ProvenanceData(
                model_id=result.model_id,
                model_version=result.model_version,
                feature_schema_version=result.feature_schema_version,
                prediction_timestamp=result.prediction_timestamp,
                model_artifact_hash=result.model_artifact_hash,
                scaler_artifact_hash=result.scaler_artifact_hash
            )
        )
