"""
Pure-Python helper functions and schema formatters for PatchGuard Phase 9.5
Pull Request Explorer Developer Dashboard.

Contains input validation, summary statistics computation, tabular formatting,
environment configuration inspection, and domain-safe error mapping.
Strictly decoupled from Streamlit rendering and model retraining.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple

from src.analysis.pr_schema import PRCommitPrediction
from src.analysis.exceptions import InvalidPRRefError
from src.engine.exceptions import (
    RepositoryNotFoundError,
    InvalidGitRepositoryError,
    CommitNotFoundError
)
from src.github.exceptions import (
    GitHubConfigError,
    GitHubAuthError,
    GitHubPRNotFoundError,
    GitHubPRValidationError,
    GitHubAPIError,
    RepositoryAcquisitionError,
    CommitNotAvailableError
)

GITHUB_REPO_REGEX = re.compile(r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$')
GITHUB_PR_URL_REGEX = re.compile(
    r'^(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)/pull/(\d+)(?:[/?#].*)?$',
    re.IGNORECASE
)


def validate_local_pr_inputs(repo_path: str, base_ref: str, head_ref: str) -> None:
    """
    Validates input arguments for local Git PR analysis.
    Raises ValueError with descriptive guidance if inputs are invalid.
    """
    if not repo_path or not repo_path.strip():
        raise ValueError("Repository path cannot be empty.")
    if not base_ref or not base_ref.strip():
        raise ValueError("Base reference (e.g., 'main') cannot be empty.")
    if not head_ref or not head_ref.strip():
        raise ValueError("Head reference (e.g., 'HEAD' or feature branch) cannot be empty.")


def parse_github_pr_identifier(
    repo_or_url: str,
    pr_number_val: Optional[Any] = None
) -> Tuple[str, str, int]:
    """
    Parses repository identifier or GitHub PR URL into (owner, repo, pr_number).

    Supported formats:
    - Full URL: 'https://github.com/owner/repo/pull/42'
    - Repo string: 'owner/repo' with explicit pr_number (e.g., 42 or '42')
    """
    cleaned = (repo_or_url or "").strip()
    if not cleaned:
        raise ValueError("GitHub repository identifier or PR URL cannot be empty.")

    # Check if input is a GitHub PR URL
    url_match = GITHUB_PR_URL_REGEX.match(cleaned)
    if url_match:
        owner, repo, num_str = url_match.groups()
        return owner, repo, int(num_str)

    # Otherwise treat as owner/repository
    if not GITHUB_REPO_REGEX.match(cleaned):
        raise ValueError(
            f"Invalid GitHub repository identifier: '{cleaned}'. "
            "Expected 'owner/repository' (e.g. 'octocat/Hello-World') or a full GitHub PR URL."
        )

    parts = cleaned.split('/')
    owner, repo = parts[0], parts[1]

    if pr_number_val is None:
        raise ValueError("PR number is required when specifying repository as 'owner/repository'.")

    try:
        pr_num = int(pr_number_val)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid PR number: '{pr_number_val}'. Must be a positive integer.")

    if pr_num <= 0:
        raise ValueError(f"PR number must be a positive integer, got: {pr_num}.")

    return owner, repo, pr_num


def is_github_app_configured() -> bool:
    """
    Checks if the current environment contains required GitHub App configuration variables.
    Never returns or exposes private keys or secret values.
    """
    app_id = os.environ.get("GITHUB_APP_ID", "").strip()
    inst_id = os.environ.get("GITHUB_INSTALLATION_ID", "").strip()
    priv_key = os.environ.get("GITHUB_APP_PRIVATE_KEY", "").strip()
    priv_key_path = os.environ.get("GITHUB_APP_PRIVATE_KEY_PATH", "").strip()

    has_key = bool(priv_key) or (bool(priv_key_path) and os.path.exists(priv_key_path))
    return bool(app_id and inst_id and has_key)


def get_pr_risk_summary_stats(
    commits: List[PRCommitPrediction],
    threshold: float = 0.35
) -> Dict[str, Any]:
    """
    Computes summary counts from a list of PRCommitPrediction objects.
    Strictly preserves individual commit metrics and does NOT compute
    a fabricated PR-wide probability or risk score.
    """
    total = len(commits)
    analyzed = 0
    defect_prone = 0
    standard_risk = 0
    merge_count = 0
    skipped_count = 0
    max_risk_score: Optional[float] = None

    for c in commits:
        if c.is_merge_commit:
            merge_count += 1
        elif c.prediction is not None:
            analyzed += 1
            prob = c.prediction.raw_probability
            if max_risk_score is None or prob > max_risk_score:
                max_risk_score = prob

            if c.prediction.is_above_threshold:
                defect_prone += 1
            else:
                standard_risk += 1
        else:
            skipped_count += 1

    return {
        "total_commits": total,
        "analyzed_commits": analyzed,
        "defect_prone_commits": defect_prone,
        "standard_risk_commits": standard_risk,
        "merge_commits": merge_count,
        "skipped_commits": skipped_count,
        "max_commit_risk_score": max_risk_score,
        "decision_threshold": threshold
    }


def format_pr_commits_table(
    commits: List[PRCommitPrediction]
) -> List[Dict[str, Any]]:
    """
    Formats a list of PRCommitPrediction objects into clean tabular display records.
    """
    records: List[Dict[str, Any]] = []

    for c in commits:
        author_short = c.author.split('<')[0].strip() if '<' in c.author else c.author
        first_line_msg = c.commit_message.splitlines()[0].strip() if c.commit_message else ""
        if len(first_line_msg) > 60:
            first_line_msg = first_line_msg[:57] + "..."

        if c.prediction is not None:
            p = c.prediction
            score_str = f"{p.raw_probability:.4f}"
            level_str = p.risk_level
            if p.is_above_threshold:
                badge = f"⚠️ {level_str}"
                classification = "Defect-Prone Change"
            else:
                badge = f"✅ {level_str}"
                classification = "Standard Risk Change"
            status = "Analyzed"
        elif c.is_merge_commit:
            score_str = "—"
            badge = "ℹ️ MERGE"
            classification = "Merge Commit"
            status = "Merge commit skipped"
        else:
            score_str = "—"
            badge = "⚠️ SKIPPED"
            classification = "Skipped"
            status = c.error or "Prediction skipped"

        records.append({
            "Commit": c.commit_hash,
            "Author": author_short,
            "Message": first_line_msg,
            "Risk Score": score_str,
            "Classification": classification,
            "Risk Level": badge,
            "Status": status,
            "Full Hash": c.full_hash
        })

    return records


def format_pr_error(exc: Exception) -> Tuple[str, str]:
    """
    Maps domain exceptions into sanitized (title, guidance) tuples.
    Never exposes internal tokens, passwords, or raw stack traces.
    """
    if isinstance(exc, InvalidPRRefError):
        return (
            "Invalid Git Reference",
            f"{exc}. Please verify that the base and head references exist in the target repository and share a common ancestor."
        )
    if isinstance(exc, GitHubConfigError):
        return (
            "GitHub App Configuration Missing",
            f"{exc}. Please configure GITHUB_APP_ID, GITHUB_INSTALLATION_ID, and GITHUB_APP_PRIVATE_KEY in your environment, or use local Git PR mode."
        )
    if isinstance(exc, GitHubAuthError):
        return (
            "GitHub Authentication Failed",
            f"{exc}. Verify that the GitHub App is installed on the target repository and the private key is valid."
        )
    if isinstance(exc, GitHubPRNotFoundError):
        return (
            "GitHub Pull Request Not Found",
            f"{exc}. Verify that the repository name, owner, and PR number are correct and accessible."
        )
    if isinstance(exc, GitHubPRValidationError):
        return (
            "Invalid GitHub PR Request",
            f"{exc}. Please provide a valid repository format ('owner/repo') and a positive integer PR number."
        )
    if isinstance(exc, GitHubAPIError):
        return (
            "GitHub API Network Error",
            f"{exc}. The GitHub API could not be reached or returned an unexpected response. Check network connectivity."
        )
    if isinstance(exc, (RepositoryAcquisitionError, CommitNotAvailableError)):
        return (
            "Repository Acquisition Failed",
            f"{exc}. Unable to clone or fetch the required commit SHAs from GitHub."
        )
    if isinstance(exc, RepositoryNotFoundError):
        return (
            "Repository Directory Not Found",
            f"{exc}. Check that the local repository path exists."
        )
    if isinstance(exc, InvalidGitRepositoryError):
        return (
            "Invalid Git Repository",
            f"{exc}. The specified directory is not a valid Git repository."
        )
    if isinstance(exc, CommitNotFoundError):
        return (
            "Commit Reference Not Found",
            f"{exc}. Verify the specified Git commit reference."
        )

    # General fallback
    return ("Analysis Error", f"{exc.__class__.__name__}: {str(exc)}")
