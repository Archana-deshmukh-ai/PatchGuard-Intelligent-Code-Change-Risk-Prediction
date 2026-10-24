"""
Prediction service business logic and repository access validation for PatchGuard REST API.
"""

import os
import time
import datetime
from typing import Optional, Dict, Any, List

from src.engine import RiskPredictionEngine, PredictionResult
from src.engine.exceptions import (
    PatchGuardError,
    RepositoryNotFoundError,
    InvalidGitRepositoryError,
    ModelError
)
from src.analysis.pr_analyzer import PRAnalyzer
from src.analysis.analyzer import NullLLMAnalyzer
from src.analysis.exceptions import InvalidPRRefError
from .config import APIConfig
from .schemas import (
    ReadyResponse,
    CommitPredictionResponse,
    CommitMetadata,
    PredictionData,
    ModelSignalsData,
    ProvenanceData,
    PRPredictionResponse,
    PRMetadata,
    PRCommitRecord,
    PRSummaryStats
)

EXPECTED_MODEL_HASH = "a32b30dff63b3a85c3fd9d9983921d5f5c5c5629acc9c206f77759c46f60f387"
EXPECTED_SCALER_HASH = "c8e65f768dab97acf8df573da60f6b1c0e9232ebf2173677f88e3efa5c479b49"
PRODUCTION_THRESHOLD = 0.35


def validate_git_ref(ref: str, field_name: str = "reference") -> None:
    """
    Validates that a Git reference does not attempt option injection,
    ambiguity, or contain characters/sequences forbidden by Git conventions.
    """
    cleaned = ref.strip() if ref else ""
    if not cleaned:
        raise InvalidPRRefError(f"Git {field_name} cannot be empty.")
    if cleaned.startswith("-"):
        raise InvalidPRRefError(f"Invalid {field_name} '{cleaned}': references cannot start with a hyphen.")
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in cleaned):
        raise InvalidPRRefError(f"Invalid {field_name} '{cleaned}': reference contains illegal whitespace or control characters.")
    for bad in ["..", "@{", ":", "^", "~", "?", "*", "[", "\\", "//"]:
        if bad in cleaned:
            raise InvalidPRRefError(f"Invalid {field_name} '{cleaned}': reference contains prohibited sequence '{bad}'.")
    if cleaned.endswith("/") or cleaned.endswith("."):
        raise InvalidPRRefError(f"Invalid {field_name} '{cleaned}': reference cannot end with a slash or dot.")



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

        if not os.path.exists(target_path) or not os.path.isdir(target_path):
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

    def predict_pr(
        self,
        repository: str,
        base_ref: str,
        head_ref: str
    ) -> PRPredictionResponse:
        """
        Executes Pull Request defect risk prediction across constituent commits
        using production engine at locked threshold 0.35.
        Bypasses qualitative LLM analysis for pure ML prediction performance.
        """
        # 1. Readiness check (validates engine initialization and artifact hashes)
        self.check_readiness()

        # 2. Repository validation and path containment
        resolved_repo_path = self.resolve_repository_path(repository)

        # 3. Git reference validation (prohibiting leading hyphens, control chars, illegal syntax)
        validate_git_ref(base_ref, "base_ref")
        validate_git_ref(head_ref, "head_ref")

        # 4. Enforce overall execution deadline and workload limits
        start_time = time.perf_counter()
        deadline = start_time + self.config.pr_timeout_seconds

        pr_analyzer = PRAnalyzer(
            predictor=self.engine,
            llm_analyzer=NullLLMAnalyzer(),
            max_commits=self.config.max_pr_commits,
            timeout=self.config.git_timeout_seconds
        )

        pr_result = pr_analyzer.analyze_pr(
            repo_path=resolved_repo_path,
            base_ref=base_ref,
            head_ref=head_ref,
            threshold=PRODUCTION_THRESHOLD,
            deadline=deadline
        )

        # 5. Extract summary stats directly from domain result
        summary_data = pr_result.get_summary_stats(threshold=PRODUCTION_THRESHOLD)

        # 6. Build commit prediction records
        commit_records: List[PRCommitRecord] = []
        for c in pr_result.commits:
            if c.prediction is not None:
                p = c.prediction
                pred_data = PredictionData(
                    raw_probability=p.raw_probability,
                    is_above_threshold=p.is_above_threshold,
                    decision_threshold=p.decision_threshold,
                    prediction_label=p.prediction_label,
                    risk_level=p.risk_level,
                    presentation_disclaimer=p.presentation_disclaimer
                )
                signals_data = ModelSignalsData(
                    top_positive_factors=p.model_signals.top_positive_factors,
                    top_negative_factors=p.model_signals.top_negative_factors,
                    log_odds_contributions=p.model_signals.log_odds_contributions
                )
                features_data = p.features
            else:
                pred_data = None
                signals_data = None
                features_data = None

            commit_records.append(PRCommitRecord(
                commit_hash=c.commit_hash,
                full_hash=c.full_hash,
                author=c.author,
                timestamp=c.timestamp,
                commit_message=c.commit_message,
                is_merge_commit=c.is_merge_commit,
                prediction=pred_data,
                model_signals=signals_data,
                features=features_data,
                error=c.error
            ))

        eng = self.engine
        timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        return PRPredictionResponse(
            pr=PRMetadata(
                repository=repository,
                base_ref=pr_result.base_ref,
                head_ref=pr_result.head_ref,
                resolved_base_sha=pr_result.resolved_base_sha,
                resolved_head_sha=pr_result.resolved_head_sha,
                resolved_merge_base_sha=pr_result.resolved_merge_base_sha,
                is_empty=pr_result.is_empty,
                commit_count=pr_result.commit_count
            ),
            summary=PRSummaryStats(
                total_commits=summary_data["total_commits"],
                analyzed_commits=summary_data["analyzed_commits"],
                defect_prone_commits=summary_data["defect_prone_commits"],
                standard_risk_commits=summary_data["standard_risk_commits"],
                merge_commits=summary_data["merge_commits"],
                skipped_commits=summary_data["skipped_commits"],
                max_commit_risk_score=summary_data["max_commit_risk_score"],
                decision_threshold=PRODUCTION_THRESHOLD
            ),
            commits=commit_records,
            provenance=ProvenanceData(
                model_id=eng.MODEL_ID,
                model_version=eng.MODEL_VERSION,
                feature_schema_version=eng.FEATURE_SCHEMA_VERSION,
                prediction_timestamp=timestamp_utc,
                model_artifact_hash=eng.model_artifact_hash,
                scaler_artifact_hash=eng.scaler_artifact_hash
            ),
            analysis_disclaimer=pr_result.analysis_disclaimer
        )
