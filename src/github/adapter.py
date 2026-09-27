"""
GitHub Pull Request adapter for PatchGuard.
Converts structured GitHubPullRequest metadata into existing PRAnalysisInput objects.
Strictly isolated from PRAnalyzer, Phase 5 ML prediction, and Phase 6 LLM analysis.
"""

import os
import re
from src.github.schema import GitHubPullRequest
from src.analysis.pr_schema import PRAnalysisInput
from src.github.exceptions import GitHubPRValidationError

HEX_SHA1_REGEX = re.compile(r'^[0-9a-fA-F]{40}$')


class GitHubPRAdapter:
    """
    Converts GitHubPullRequest metadata into existing local PRAnalysisInput objects.
    Preserves exact Phase 7.1 semantics without duplicating merge-base logic or commit enumeration.
    """
    @staticmethod
    def to_analysis_input(gh_pr: GitHubPullRequest, repo_path: str = ".") -> PRAnalysisInput:
        """
        Converts GitHubPullRequest to PRAnalysisInput using resolved GitHub commit SHAs.
        """
        if not gh_pr.base_sha or not HEX_SHA1_REGEX.match(gh_pr.base_sha):
            raise GitHubPRValidationError(f"Invalid base commit SHA in GitHub PR: '{gh_pr.base_sha}'")
        if not gh_pr.head_sha or not HEX_SHA1_REGEX.match(gh_pr.head_sha):
            raise GitHubPRValidationError(f"Invalid head commit SHA in GitHub PR: '{gh_pr.head_sha}'")

        return PRAnalysisInput(
            repo_path=os.path.abspath(repo_path),
            base_ref=gh_pr.base_sha,
            head_ref=gh_pr.head_sha
        )
