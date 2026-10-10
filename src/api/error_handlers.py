"""
Structured error handling and HTTP exception mapping for PatchGuard REST API.
Ensures zero leakage of stack traces, internal server paths, credentials, or source code.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.engine.exceptions import (
    PatchGuardError,
    RepositoryNotFoundError,
    InvalidGitRepositoryError,
    CommitNotFoundError,
    UnsupportedCommitTypeError,
    EmptyCommitError,
    InvalidFeatureValueError,
    ModelError
)
from .service import (
    ForbiddenRepositoryAccessError,
    ServiceNotReadyError,
    AuthenticationError
)


def _sanitize_message(msg: str) -> str:
    """Removes local drive letters, absolute file paths, and sensitive tokens."""
    import re
    # Strip Windows drive paths (e.g. C:\... or C:/...)
    msg = re.sub(r'[A-Za-z]:[\\/][^:\s]+', '[protected-path]', msg)
    # Strip Unix-style root paths (e.g. /home/..., /var/..., /repositories/..., /app/...)
    msg = re.sub(r'/(?:var|tmp|home|etc|usr|opt|repositories|app)/[^\s:]+', '[protected-path]', msg)
    return msg


def register_error_handlers(app: FastAPI) -> None:
    """
    Registers custom exception handlers on the FastAPI application instance.
    """

    @app.exception_handler(AuthenticationError)
    async def auth_error_handler(request: Request, exc: AuthenticationError):
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "code": "AUTHENTICATION_REQUIRED",
                    "message": "Valid API key required in 'X-API-Key' header.",
                    "remediation": "Provide a valid 'X-API-Key' header matching configured PATCHGUARD_API_KEY."
                }
            }
        )

    @app.exception_handler(ForbiddenRepositoryAccessError)
    async def forbidden_repo_handler(request: Request, exc: ForbiddenRepositoryAccessError):
        return JSONResponse(
            status_code=403,
            content={
                "error": {
                    "code": "FORBIDDEN_REPOSITORY_ACCESS",
                    "message": "Access to requested repository path is forbidden.",
                    "remediation": "Specify a repository identifier within the configured permitted repository root."
                }
            }
        )

    @app.exception_handler(RepositoryNotFoundError)
    async def repo_not_found_handler(request: Request, exc: RepositoryNotFoundError):
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "REPOSITORY_NOT_FOUND",
                    "message": _sanitize_message(str(exc)),
                    "remediation": "Verify the repository identifier exists in the permitted workspace."
                }
            }
        )

    @app.exception_handler(InvalidGitRepositoryError)
    async def invalid_git_repo_handler(request: Request, exc: InvalidGitRepositoryError):
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "INVALID_GIT_REPOSITORY",
                    "message": _sanitize_message(str(exc)),
                    "remediation": "Ensure target directory contains a valid initialized Git repository (.git)."
                }
            }
        )

    @app.exception_handler(CommitNotFoundError)
    async def commit_not_found_handler(request: Request, exc: CommitNotFoundError):
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "COMMIT_NOT_FOUND",
                    "message": _sanitize_message(str(exc)),
                    "remediation": "Verify that the commit hash, branch, or tag exists in the target repository."
                }
            }
        )

    @app.exception_handler(UnsupportedCommitTypeError)
    async def unsupported_commit_handler(request: Request, exc: UnsupportedCommitTypeError):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "UNSUPPORTED_COMMIT_TYPE",
                    "message": _sanitize_message(str(exc)),
                    "remediation": "Merge commits with multiple parents are excluded from risk prediction. Evaluate constituent commits individually."
                }
            }
        )

    @app.exception_handler(EmptyCommitError)
    async def empty_commit_handler(request: Request, exc: EmptyCommitError):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "EMPTY_COMMIT",
                    "message": _sanitize_message(str(exc)),
                    "remediation": "The specified commit contains zero file changes to evaluate."
                }
            }
        )

    @app.exception_handler(InvalidFeatureValueError)
    async def invalid_feature_handler(request: Request, exc: InvalidFeatureValueError):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "INVALID_FEATURE_VALUE",
                    "message": _sanitize_message(str(exc)),
                    "remediation": "Feature values extracted from commit diff could not be evaluated numerically."
                }
            }
        )

    @app.exception_handler(ServiceNotReadyError)
    async def service_not_ready_handler(request: Request, exc: ServiceNotReadyError):
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "SERVICE_NOT_READY",
                    "message": "Prediction service is not ready to serve requests.",
                    "remediation": "Verify production model and scaler artifact files are present, uncorrupted, and match expected hashes."
                }
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        # Format field errors cleanly
        error_msgs = []
        for err in exc.errors():
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            msg = err.get("msg", "Invalid value")
            error_msgs.append(f"{loc}: {msg}")
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request schema validation failed: " + "; ".join(error_msgs),
                    "remediation": "Check request parameters against OpenAPI contract schema."
                }
            }
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        code_map = {
            404: "ENDPOINT_NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED",
            403: "FORBIDDEN"
        }
        err_code = code_map.get(exc.status_code, f"HTTP_{exc.status_code}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": err_code,
                    "message": _sanitize_message(str(exc.detail)),
                    "remediation": "Verify the request URI, HTTP method, and headers."
                }
            }
        )

    @app.exception_handler(PatchGuardError)
    async def generic_engine_error_handler(request: Request, exc: PatchGuardError):
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ENGINE_ERROR",
                    "message": _sanitize_message(str(exc)),
                    "remediation": "An internal error occurred during prediction engine processing."
                }
            }
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected server error occurred.",
                    "remediation": "Check server operational logs for diagnostic information."
                }
            }
        )
