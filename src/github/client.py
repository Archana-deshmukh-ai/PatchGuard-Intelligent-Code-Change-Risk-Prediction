"""
Read-only GitHub API client for PatchGuard Phase 7.2.
Retrieves Pull Request metadata from GitHub API endpoint GET /repos/{owner}/{repo}/pulls/{pull_number}.
Operates strictly in read-only mode and protects credentials/tokens from leakage.
"""

import re
import json
import urllib.request
import urllib.error
from typing import Optional, Callable, Dict, Any

from src.github.schema import GitHubPullRequest
from src.github.auth import GitHubAppConfig, GitHubAppAuthenticator
from src.github.exceptions import (
    GitHubIntegrationError,
    GitHubConfigError,
    GitHubAuthError,
    GitHubAPIError,
    GitHubPRNotFoundError,
    GitHubPRValidationError
)

SHA1_REGEX = re.compile(r'^[0-9a-fA-F]{40}$')
REPO_NAME_REGEX = re.compile(r'^[a-zA-Z0-9_.-]+$')


class GitHubClient:
    """
    Read-only GitHub API client focused on retrieving PR metadata.
    Does not call PRAnalyzer or execute repository operations.
    """
    def __init__(
        self,
        authenticator: Optional[GitHubAppAuthenticator] = None,
        config: Optional[GitHubAppConfig] = None,
        token_provider: Optional[Callable[[], str]] = None,
        base_url: str = "https://api.github.com",
        http_fetcher: Optional[Callable[[str, Dict[str, str]], Dict[str, Any]]] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.http_fetcher = http_fetcher
        self.token_provider = token_provider

        if token_provider is None:
            self.authenticator = authenticator if authenticator is not None else GitHubAppAuthenticator()
            self.config = config  # Defer config resolution if None to get_pull_request

    def _get_token(self) -> str:
        """Resolves access token securely without exposing secrets."""
        if self.token_provider is not None:
            token = self.token_provider()
            if not token:
                raise GitHubAuthError("Token provider returned empty token.")
            return token

        cfg = self.config if self.config is not None else GitHubAppConfig.from_env()
        return self.authenticator.get_installation_access_token(cfg, base_url=self.base_url)

    def _validate_input_params(self, owner: str, repo: str, pull_number: int):
        """Validates owner, repo, and pull_number input parameters."""
        if not owner or not REPO_NAME_REGEX.match(owner):
            raise GitHubPRValidationError(f"Invalid repository owner identifier: '{owner}'")
        if not repo or not REPO_NAME_REGEX.match(repo):
            raise GitHubPRValidationError(f"Invalid repository name identifier: '{repo}'")
        if not isinstance(pull_number, int) or pull_number <= 0:
            raise GitHubPRValidationError(f"Invalid PR number: '{pull_number}'. PR number must be a positive integer.")

    def get_pull_request(self, owner: str, repo: str, pull_number: int) -> GitHubPullRequest:
        """
        Retrieves metadata for a specific GitHub Pull Request.
        """
        self._validate_input_params(owner, repo, pull_number)

        token = self._get_token()
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pull_number}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "PatchGuard-Risk-Prediction"
        }

        # Offline test fetcher path
        if self.http_fetcher is not None:
            try:
                data = self.http_fetcher(url, headers)
            except GitHubIntegrationError:
                raise
            except Exception as e:
                raise GitHubAPIError(f"GitHub API request failed: {e.__class__.__name__}")
        else:
            # Live HTTP request using urllib.request
            req = urllib.request.Request(url, headers=headers, method="GET")
            try:
                with urllib.request.urlopen(req, timeout=30.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as he:
                if he.code == 404:
                    raise GitHubPRNotFoundError(owner=owner, repo=repo, pull_number=pull_number)
                elif he.code in (401, 403):
                    raise GitHubAuthError(f"GitHub API authorization failed for '{owner}/{repo}#{pull_number}' (HTTP {he.code}).")
                else:
                    raise GitHubAPIError(f"GitHub API request failed for '{owner}/{repo}#{pull_number}' with HTTP {he.code}.", status_code=he.code)
            except Exception as e:
                raise GitHubAPIError(f"GitHub API network error: {e.__class__.__name__}")

        # Parse and validate response fields
        if not isinstance(data, dict):
            raise GitHubPRValidationError("Malformed GitHub API response: expected JSON object.")

        base_data = data.get("base", {})
        head_data = data.get("head", {})

        base_ref = base_data.get("ref", "")
        base_sha = base_data.get("sha", "")
        head_ref = head_data.get("ref", "")
        head_sha = head_data.get("sha", "")
        title = data.get("title", "")
        html_url = data.get("html_url", "")
        number = data.get("number", pull_number)
        state = data.get("state", "open")

        # Validate commit SHAs
        if not base_sha or not SHA1_REGEX.match(base_sha):
            raise GitHubPRValidationError(f"GitHub PR response missing or invalid base SHA: '{base_sha}'")
        if not head_sha or not SHA1_REGEX.match(head_sha):
            raise GitHubPRValidationError(f"GitHub PR response missing or invalid head SHA: '{head_sha}'")

        res_owner = base_data.get("repo", {}).get("owner", {}).get("login", owner)
        res_repo = base_data.get("repo", {}).get("name", repo)

        return GitHubPullRequest(
            owner=res_owner,
            repository=res_repo,
            number=number,
            title=title,
            html_url=html_url,
            base_ref=base_ref,
            base_sha=base_sha,
            head_ref=head_ref,
            head_sha=head_sha,
            state=state
        )
