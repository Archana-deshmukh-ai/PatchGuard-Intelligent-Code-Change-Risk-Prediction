"""
PatchGuard GitHub Integration package (Phase 7.2.1).
Provides read-only GitHub Pull Request retrieval, App authentication, metadata modeling, and adaptation.
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
    RepositoryCleanupError
)
from .schema import GitHubPullRequest
from .auth import GitHubAppConfig, GitHubAppAuthenticator
from .client import GitHubClient
from .adapter import GitHubPRAdapter
from .acquisition import RepositoryAcquisitionManager

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
    "GitHubPullRequest",
    "GitHubAppConfig",
    "GitHubAppAuthenticator",
    "GitHubClient",
    "GitHubPRAdapter",
    "RepositoryAcquisitionManager"
]
