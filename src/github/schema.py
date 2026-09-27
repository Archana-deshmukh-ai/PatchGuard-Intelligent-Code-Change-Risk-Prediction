"""
Internal data models for GitHub Pull Request metadata in PatchGuard.
Represents only the minimum structured data required by PatchGuard, ignoring extra API fields.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass(frozen=True)
class GitHubPullRequest:
    """
    Structured representation of a GitHub Pull Request's read-only metadata.
    """
    owner: str
    repository: str
    number: int
    title: str
    html_url: str
    base_ref: str
    base_sha: str
    head_ref: str
    head_sha: str
    state: str = "open"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes GitHubPullRequest to a safe dictionary representation."""
        return {
            "owner": self.owner,
            "repository": self.repository,
            "number": self.number,
            "title": self.title,
            "html_url": self.html_url,
            "base_ref": self.base_ref,
            "base_sha": self.base_sha,
            "head_ref": self.head_ref,
            "head_sha": self.head_sha,
            "state": self.state
        }
