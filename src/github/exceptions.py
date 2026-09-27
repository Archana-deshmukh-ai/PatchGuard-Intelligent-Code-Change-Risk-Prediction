"""
Custom exception hierarchy for PatchGuard Phase 7.2 & 7.3 GitHub integration.
Strictly ensures credentials, tokens, and authorization headers are never exposed in exception messages.
"""

from src.engine.exceptions import PatchGuardError

class GitHubIntegrationError(PatchGuardError):
    """Base exception for all GitHub integration errors."""
    pass

class GitHubConfigError(GitHubIntegrationError):
    """Raised when GitHub App configuration is missing, incomplete, or invalid."""
    pass

class GitHubAuthError(GitHubIntegrationError):
    """Raised when GitHub App JWT signing or installation access token exchange fails."""
    pass

class GitHubAPIError(GitHubIntegrationError):
    """Raised when a GitHub API request fails or returns an unexpected status code."""
    def __init__(self, message: str, status_code: int = 0):
        clean_msg = message.split("Authorization")[0].strip() if "Authorization" in message else message
        super().__init__(clean_msg)
        self.status_code = status_code

class GitHubPRNotFoundError(GitHubAPIError):
    """Raised when the requested GitHub Pull Request or repository is not found (HTTP 404)."""
    def __init__(self, owner: str, repo: str, pull_number: int):
        super().__init__(
            message=f"Pull Request #{pull_number} in repository '{owner}/{repo}' was not found.",
            status_code=404
        )
        self.owner = owner
        self.repo = repo
        self.pull_number = pull_number

class GitHubPRValidationError(GitHubIntegrationError):
    """Raised when GitHub PR response metadata or repository identifiers fail structural validation."""
    pass

class RepositoryAcquisitionError(GitHubIntegrationError):
    """Base exception for repository acquisition and workspace management errors."""
    def __init__(self, message: str):
        clean_msg = message.split("Authorization")[0].strip() if "Authorization" in message else message
        super().__init__(clean_msg)

class RepositoryCloneError(RepositoryAcquisitionError):
    """Raised when cloning a remote Git repository fails."""
    pass

class RepositoryFetchError(RepositoryAcquisitionError):
    """Raised when fetching commits from a remote Git repository fails."""
    pass

class RepositoryValidationError(RepositoryAcquisitionError):
    """Raised when a local or acquired Git repository fails validation."""
    pass

class CommitNotAvailableError(RepositoryAcquisitionError):
    """Raised when requested base or head commit SHAs are missing from the local repository."""
    pass

class RepositoryIdentityMismatchError(RepositoryAcquisitionError):
    """Raised when a local repository does not match the requested owner/repository identity."""
    pass

class RepositoryCleanupError(RepositoryAcquisitionError):
    """Raised when temporary repository workspace cleanup fails."""
    pass

class WebhookAuthenticationError(GitHubIntegrationError):
    """Raised when GitHub webhook HMAC signature verification fails or signature header is missing."""
    pass

class WebhookValidationError(GitHubIntegrationError):
    """Raised when GitHub webhook request payload is missing required attributes or malformed."""
    pass

class UnsupportedGitHubEvent(GitHubIntegrationError):
    """Raised when GitHub webhook event type or pull_request action is not supported for analysis."""
    pass

class PRAnalysisWorkflowError(GitHubIntegrationError):
    """Raised when automated PR analysis workflow orchestration encounters an execution error."""
    def __init__(self, message: str):
        clean_msg = message.split("Authorization")[0].strip() if "Authorization" in message else message
        super().__init__(clean_msg)

class GitHubReportingError(GitHubIntegrationError):
    """Raised when posting or updating a GitHub PR analysis comment fails."""
    def __init__(self, message: str):
        clean_msg = message.split("Authorization")[0].strip() if "Authorization" in message else message
        super().__init__(clean_msg)
