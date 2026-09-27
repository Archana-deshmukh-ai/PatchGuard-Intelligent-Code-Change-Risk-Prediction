"""
PatchGuard GitHub Integration package (Phase 7.3).
Provides GitHub Pull Request retrieval, App authentication, metadata modeling, adaptation,
authenticated repository acquisition, webhook parsing & signature verification, PR comment reporting,
and automated PR workflow orchestration.
"""

from .exceptions import (
    GitHubIntegrationError,
    GitHubConfigError,
    GitHubAuthError,
    GitHubAPIError,
    GitHubPRNotFoundError,
    GitHubPRValidationError,
    RepositoryAcquisitionError,
    RepositoryCloneError,
    RepositoryFetchError,
    RepositoryValidationError,
    CommitNotAvailableError,
    RepositoryIdentityMismatchError,
    RepositoryCleanupError,
    WebhookAuthenticationError,
    WebhookValidationError,
    UnsupportedGitHubEvent,
    PRAnalysisWorkflowError,
    GitHubReportingError
)
from .schema import GitHubPullRequest
from .auth import GitHubAppConfig, GitHubAppAuthenticator
from .client import GitHubClient
from .adapter import GitHubPRAdapter
from .acquisition import RepositoryAcquisitionManager
from .webhook import verify_webhook_signature, WebhookEventHandler, WebhookEventData
from .reporter import GitHubPRReporter, IDEMPOTENCY_MARKER
from .workflow import GitHubPRWorkflow

__all__ = [
    "GitHubIntegrationError",
    "GitHubConfigError",
    "GitHubAuthError",
    "GitHubAPIError",
    "GitHubPRNotFoundError",
    "GitHubPRValidationError",
    "RepositoryAcquisitionError",
    "RepositoryCloneError",
    "RepositoryFetchError",
    "RepositoryValidationError",
    "CommitNotAvailableError",
    "RepositoryIdentityMismatchError",
    "RepositoryCleanupError",
    "WebhookAuthenticationError",
    "WebhookValidationError",
    "UnsupportedGitHubEvent",
    "PRAnalysisWorkflowError",
    "GitHubReportingError",
    "GitHubPullRequest",
    "GitHubAppConfig",
    "GitHubAppAuthenticator",
    "GitHubClient",
    "GitHubPRAdapter",
    "RepositoryAcquisitionManager",
    "verify_webhook_signature",
    "WebhookEventHandler",
    "WebhookEventData",
    "GitHubPRReporter",
    "IDEMPOTENCY_MARKER",
    "GitHubPRWorkflow"
]
