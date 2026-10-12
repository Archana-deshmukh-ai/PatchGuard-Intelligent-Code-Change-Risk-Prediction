"""
FastAPI application entrypoint and route definitions for PatchGuard REST API.
"""

import time
import hmac
import logging
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Header, Depends, status

from .config import APIConfig
from .schemas import (
    HealthResponse,
    ReadyResponse,
    CommitPredictionRequest,
    CommitPredictionResponse,
    ErrorResponse
)
from .service import (
    PredictionService,
    AuthenticationError
)
from .error_handlers import register_error_handlers

logger = logging.getLogger("patchguard.api")


def get_service(request: Request) -> PredictionService:
    """Dependency provider for PredictionService."""
    return request.app.state.service


def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> None:
    """
    Enforces API key authentication when configured via PATCHGUARD_API_KEY.
    Uses constant-time comparison via hmac.compare_digest.
    """
    config: APIConfig = request.app.state.config
    if config.api_key is not None:
        if not x_api_key or not hmac.compare_digest(x_api_key, config.api_key):
            raise AuthenticationError("Valid API key required.")


def create_app(
    config: Optional[APIConfig] = None,
    service: Optional[PredictionService] = None
) -> FastAPI:
    """
    Factory creating and configuring the FastAPI application.
    Supports dependency injection for testing.
    """
    active_config = config if config is not None else APIConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # App startup
        if service is not None:
            app.state.service = service
        else:
            app.state.service = PredictionService(config=active_config)

        app.state.config = active_config

        if active_config.api_key is None:
            if active_config.host in ("0.0.0.0", "::"):
                logger.warning(
                    "SECURITY WARNING: Server is bound to all network interfaces (%s) without PATCHGUARD_API_KEY. "
                    "All incoming requests will be permitted without authentication. "
                    "For production or network deployments, set PATCHGUARD_API_KEY.",
                    active_config.host
                )
            else:
                logger.warning(
                    "PATCHGUARD_API_KEY is not set. API is operating in unauthenticated local-development mode."
                )
        yield
        # App shutdown
        pass

    app = FastAPI(
        title="PatchGuard Code Change Risk Prediction API",
        version="1.0.0",
        description="REST API for predicting defect risk in Git commits using production Machine Learning.",
        lifespan=lifespan
    )

    # Store state immediately for synchronous TestClient compatibility
    app.state.config = active_config
    if service is not None:
        app.state.service = service
    else:
        app.state.service = PredictionService(config=active_config)

    # Register error handlers
    register_error_handlers(app)

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Process Liveness Probe",
        tags=["System"],
        status_code=status.HTTP_200_OK
    )
    async def health_check() -> HealthResponse:
        """
        Reports process liveness only.
        Does not imply that model artifacts are loaded or ready for inference.
        """
        return HealthResponse(
            status="pass",
            service="patchguard-api",
            version="1.0.0"
        )

    @app.get(
        "/ready",
        response_model=ReadyResponse,
        summary="Model & Inference Readiness Probe",
        tags=["System"],
        status_code=status.HTTP_200_OK,
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse}
        }
    )
    async def readiness_check(
        svc: PredictionService = Depends(get_service)
    ) -> ReadyResponse:
        """
        Reports readiness of the production ML prediction engine.
        Returns HTTP 503 if artifacts are missing, uninitialized, or fail hash integrity.
        """
        return svc.check_readiness()

    @app.post(
        "/v1/predictions/commit",
        response_model=CommitPredictionResponse,
        summary="Predict Commit Defect Risk",
        tags=["Predictions"],
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],
        responses={
            status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
            status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse},
            status.HTTP_403_FORBIDDEN: {"model": ErrorResponse},
            status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
            status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse}
        }
    )
    async def predict_commit(
        request_body: CommitPredictionRequest,
        svc: PredictionService = Depends(get_service)
    ) -> CommitPredictionResponse:
        """
        Predicts defect risk for a specific commit in an allowlisted repository
        using the production Logistic Regression model at locked threshold 0.35.
        """
        start_time = time.perf_counter()
        result = svc.predict_commit(
            repository=request_body.repository,
            commit_hash=request_body.commit_hash
        )
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Prediction completed: repo=%s commit=%s risk_level=%s score=%.4f duration_ms=%.2f",
            request_body.repository,
            result.commit.commit_hash,
            result.prediction.risk_level,
            result.prediction.raw_probability,
            duration_ms
        )
        return result

    return app


# Default app instance for ASGI servers (uvicorn src.api.app:app)
app = create_app()
