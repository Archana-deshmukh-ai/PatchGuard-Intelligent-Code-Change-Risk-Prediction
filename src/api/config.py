"""
Configuration settings for PatchGuard REST API.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class APIConfig:
    """
    Configuration parameters for the PatchGuard API service.
    """
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: Optional[str] = None
    allowed_repos_root: str = ""
    models_dir: Optional[str] = None
    default_threshold: float = 0.35
    git_timeout_seconds: float = 30.0
    max_pr_commits: int = 100
    pr_timeout_seconds: float = 60.0

    def __post_init__(self):
        if not self.allowed_repos_root:
            self.allowed_repos_root = os.path.abspath(
                os.getenv("PATCHGUARD_REPOS_ROOT", os.getcwd())
            )
        else:
            self.allowed_repos_root = os.path.abspath(self.allowed_repos_root)

    @classmethod
    def from_env(cls) -> "APIConfig":
        """
        Loads configuration from environment variables.
        """
        host = os.getenv("PATCHGUARD_HOST", "127.0.0.1")
        port = int(os.getenv("PATCHGUARD_PORT", "8000"))
        api_key = os.getenv("PATCHGUARD_API_KEY", None)
        if api_key is not None and not api_key.strip():
            api_key = None

        allowed_repos_root = os.getenv("PATCHGUARD_REPOS_ROOT", os.getcwd())
        models_dir = os.getenv("PATCHGUARD_MODELS_DIR", None)

        try:
            max_pr_commits = int(os.getenv("PATCHGUARD_MAX_PR_COMMITS", "100"))
        except (ValueError, TypeError):
            max_pr_commits = 100

        try:
            pr_timeout_seconds = float(os.getenv("PATCHGUARD_PR_TIMEOUT_SECONDS", "60.0"))
        except (ValueError, TypeError):
            pr_timeout_seconds = 60.0

        return cls(
            host=host,
            port=port,
            api_key=api_key,
            allowed_repos_root=allowed_repos_root,
            models_dir=models_dir,
            default_threshold=0.35,
            git_timeout_seconds=30.0,
            max_pr_commits=max_pr_commits,
            pr_timeout_seconds=pr_timeout_seconds
        )
