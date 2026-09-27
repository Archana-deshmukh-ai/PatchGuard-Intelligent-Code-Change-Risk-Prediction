"""
GitHub App authentication infrastructure for PatchGuard Phase 7.2.
Handles configuration loading, RS256 JWT generation, and Installation Access Token exchange.
Strictly ensures secrets (private keys, JWTs, bearer tokens) are never leaked.
"""

import os
import time
import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

import jwt
from src.github.exceptions import GitHubConfigError, GitHubAuthError


@dataclass
class GitHubAppConfig:
    """
    Configuration dataclass for GitHub App authentication.
    """
    app_id: str
    private_key: str
    installation_id: str

    def __repr__(self) -> str:
        # Prevent private key exposure in repr or logs
        key_snippet = self.private_key[:25] + "..." if len(self.private_key) > 25 else "***"
        return f"GitHubAppConfig(app_id='{self.app_id}', installation_id='{self.installation_id}', private_key='{key_snippet}')"

    @classmethod
    def from_env(cls) -> "GitHubAppConfig":
        """
        Loads GitHub App configuration strictly from environment variables.
        Supported variables:
        - GITHUB_APP_ID
        - GITHUB_APP_PRIVATE_KEY or GITHUB_APP_PRIVATE_KEY_PATH
        - GITHUB_INSTALLATION_ID
        """
        app_id = os.environ.get("GITHUB_APP_ID", "").strip()
        installation_id = os.environ.get("GITHUB_INSTALLATION_ID", "").strip()
        private_key = os.environ.get("GITHUB_APP_PRIVATE_KEY", "").strip()
        key_path = os.environ.get("GITHUB_APP_PRIVATE_KEY_PATH", "").strip()

        if not private_key and key_path:
            if os.path.exists(key_path):
                with open(key_path, "r", encoding="utf-8") as f:
                    private_key = f.read().strip()
            else:
                raise GitHubConfigError(f"GitHub App private key file not found: '{key_path}'")

        if not app_id:
            raise GitHubConfigError("Missing environment variable: 'GITHUB_APP_ID'")
        if not private_key:
            raise GitHubConfigError("Missing environment variable: 'GITHUB_APP_PRIVATE_KEY' or 'GITHUB_APP_PRIVATE_KEY_PATH'")
        if not installation_id:
            raise GitHubConfigError("Missing environment variable: 'GITHUB_INSTALLATION_ID'")

        return cls(
            app_id=app_id,
            private_key=private_key,
            installation_id=installation_id
        )


class GitHubAppAuthenticator:
    """
    Handles RS256 JWT generation and GitHub App installation access token exchange.
    """
    def generate_jwt(
        self,
        app_id: str,
        private_key: str,
        now: Optional[int] = None
    ) -> str:
        """
        Generates a RS256 signed JWT for GitHub App authentication (valid for 10 minutes).
        """
        if now is None:
            now = int(time.time())

        payload = {
            "iat": now - 60,
            "exp": now + (10 * 60),
            "iss": app_id
        }

        try:
            token = jwt.encode(payload, private_key, algorithm="RS256")
            if isinstance(token, bytes):
                token = token.decode("utf-8")
            return token
        except Exception as e:
            raise GitHubAuthError(f"Failed to generate RS256 JWT: {e.__class__.__name__}")

    def get_installation_access_token(
        self,
        config: GitHubAppConfig,
        http_requester: Optional[Callable[[str, Dict[str, str], str], Dict[str, Any]]] = None,
        base_url: str = "https://api.github.com"
    ) -> str:
        """
        Exchanges a signed JWT for a GitHub App installation access token.
        """
        jwt_token = self.generate_jwt(config.app_id, config.private_key)
        url = f"{base_url.rstrip('/')}/app/installations/{config.installation_id}/access_tokens"
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "PatchGuard-Risk-Prediction"
        }

        if http_requester is not None:
            # Custom requester (used for offline unit testing)
            try:
                res = http_requester(url, headers, "POST")
                token = res.get("token")
                if not token:
                    raise GitHubAuthError("Installation token exchange response missing 'token' field.")
                return token
            except GitHubAuthError:
                raise
            except Exception as e:
                raise GitHubAuthError(f"Installation token exchange failed: {e.__class__.__name__}")

        # Live HTTP request using urllib.request
        req = urllib.request.Request(url, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                token = data.get("token")
                if not token:
                    raise GitHubAuthError("Installation token response missing 'token' field.")
                return token
        except urllib.error.HTTPError as he:
            if he.code in (401, 403):
                raise GitHubAuthError("GitHub App authentication failed (401/403 Unauthorized). Check App ID, Installation ID, and Private Key.")
            raise GitHubAuthError(f"GitHub App authentication request failed with status code {he.code}.")
        except Exception as e:
            raise GitHubAuthError(f"GitHub App authentication network error: {e.__class__.__name__}")
