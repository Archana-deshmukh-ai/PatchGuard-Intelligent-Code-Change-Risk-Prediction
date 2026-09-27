"""
GitHub PR Analysis reporter for PatchGuard Phase 7.3.
Formats markdown PR reports with idempotency marker tags and posts or updates PR comments
via GitHub Issues API endpoints:
- GET /repos/{owner}/{repo}/issues/{issue_number}/comments (search existing report)
- PATCH /repos/{owner}/{repo}/issues/comments/{comment_id} (update existing report)
- POST /repos/{owner}/{repo}/issues/{issue_number}/comments (create new report)
"""

import json
import urllib.request
import urllib.error
from typing import Optional, Callable, Dict, Any, List

from src.analysis.pr_schema import PRAnalysisResult
from src.github.schema import GitHubPullRequest
from src.github.auth import GitHubAppConfig, GitHubAppAuthenticator
from src.github.exceptions import (
    GitHubReportingError,
    GitHubAuthError,
    GitHubIntegrationError
)

IDEMPOTENCY_MARKER = "<!-- patchguard-report -->"


class GitHubPRReporter:
    """
    Formats and publishes PatchGuard PR risk analysis comments to GitHub Pull Requests.
    Ensures idempotency by updating existing PatchGuard report comments instead of spamming duplicates.
    """
    def __init__(
        self,
        authenticator: Optional[GitHubAppAuthenticator] = None,
        config: Optional[GitHubAppConfig] = None,
        token_provider: Optional[Callable[[], str]] = None,
        base_url: str = "https://api.github.com",
        http_fetcher: Optional[Callable[[str, Dict[str, str], str, Optional[Dict[str, Any]]], Any]] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.http_fetcher = http_fetcher
        self.token_provider = token_provider

        if token_provider is None:
            self.authenticator = authenticator if authenticator is not None else GitHubAppAuthenticator()
            self.config = config

    def _get_token(self) -> str:
        """Resolves GitHub installation access token securely."""
        if self.token_provider is not None:
            token = self.token_provider()
            if not token:
                raise GitHubAuthError("Token provider returned empty token.")
            return token

        cfg = self.config if self.config is not None else GitHubAppConfig.from_env()
        return self.authenticator.get_installation_access_token(cfg, base_url=self.base_url)

    def format_report(
        self,
        pr_analysis: PRAnalysisResult,
        gh_pr: Optional[GitHubPullRequest] = None
    ) -> str:
        """
        Formats a PRAnalysisResult into GitHub Flavored Markdown for a PR comment.
        Includes idempotency marker tag at the very top.
        """
        lines = [IDEMPOTENCY_MARKER, ""]
        lines.append("## 🛡️ PatchGuard — Pull Request Risk & Code Change Analysis")
        lines.append("")

        if gh_pr:
            lines.append(f"**Target Repository**: `{gh_pr.owner}/{gh_pr.repository}` | **Pull Request**: #{gh_pr.number}")
            lines.append(f"**Base -> Head**: `{gh_pr.base_ref}` (`{gh_pr.base_sha[:8]}`) → `{gh_pr.head_ref}` (`{gh_pr.head_sha[:8]}`)")
        else:
            lines.append(f"**Base -> Head**: `{pr_analysis.base_ref}` (`{pr_analysis.resolved_base_sha[:8]}`) → `{pr_analysis.head_ref}` (`{pr_analysis.resolved_head_sha[:8]}`)")
        lines.append("")

        # Section 1: ML Defect Risk Predictions
        lines.append("### 📊 Per-Commit Historical ML Defect Risk Predictions")
        lines.append("")

        if pr_analysis.commits:
            lines.append("| Commit | Author | ML Risk Score | Risk Level | Details |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")

            for c in pr_analysis.commits:
                commit_short = f"`{c.commit_hash}`"
                author_short = c.author.split('<')[0].strip() if '<' in c.author else c.author

                if c.prediction:
                    p = c.prediction
                    score_str = f"**{p.raw_probability:.4f}**"
                    level_badge = f"`{p.risk_level}`"
                    if p.is_above_threshold:
                        level_badge = f"⚠️ `{p.risk_level}`"
                    else:
                        level_badge = f"✅ `{p.risk_level}`"
                    
                    details = f"Threshold: {p.decision_threshold:.2f}"
                    lines.append(f"| {commit_short} | {author_short} | {score_str} | {level_badge} | {details} |")
                elif c.is_merge_commit:
                    lines.append(f"| {commit_short} | {author_short} | *N/A* | ℹ️ `MERGE` | Merge commit skipped |")
                else:
                    lines.append(f"| {commit_short} | {author_short} | *N/A* | ⚠️ `SKIPPED` | {c.error or 'ML prediction unavailable'} |")
            lines.append("")
        else:
            lines.append("*No commits introduced in this pull request comparison.*")
            lines.append("")

        # Section 2: LLM Qualitative Analysis
        lines.append("### 🧠 Cumulative PR Code-Change Analysis (Phase 6 LLM)")
        lines.append("")

        if pr_analysis.is_empty or pr_analysis.cumulative_analysis is None:
            lines.append("*No code changes detected between base and head references.*")
            lines.append("")
        else:
            ca = pr_analysis.cumulative_analysis
            lines.append(f"**Summary**: {ca.summary}")
            lines.append("")

            if ca.key_changes:
                lines.append("**Key Changes:**")
                for item in ca.key_changes:
                    lines.append(f"- {item}")
                lines.append("")

            if ca.potential_risk_factors:
                lines.append("**Potential Risk Factors:**")
                for rf in ca.potential_risk_factors:
                    pri_badge = f"[{rf.review_priority}]"
                    line_info = f" ({rf.line_range})" if rf.line_range else ""
                    verified_tag = " [Verified Grounding]" if rf.evidence_verified else " [Ungrounded]"
                    lines.append(f"- **{pri_badge} {rf.category}**: {rf.description}")
                    lines.append(f"  - *Location*: `{rf.file_path}`{line_info}")
                    if rf.evidence_snippet:
                        lines.append(f"  - *Evidence*: `{rf.evidence_snippet}`{verified_tag}")
                lines.append("")

            if ca.affected_areas:
                lines.append(f"**Affected Areas**: {', '.join(ca.affected_areas)}")
                lines.append("")

            if ca.testing_observations:
                lines.append("**Testing Observations:**")
                for obs in ca.testing_observations:
                    lines.append(f"- {obs}")
                lines.append("")

            if ca.recommended_review_actions:
                lines.append("**Recommended Review Actions:**")
                for ra in ca.recommended_review_actions:
                    target_str = f" (`{ra.target_file}`)" if ra.target_file else ""
                    lines.append(f"- {ra.action}{target_str}")
                lines.append("")

        # Section 3: Disclaimer
        lines.append("---")
        lines.append(f"> ℹ️ **PatchGuard Disclaimer**: {pr_analysis.analysis_disclaimer}")
        lines.append("")

        return "\n".join(lines)

    def _execute_http(
        self,
        url: str,
        headers: Dict[str, str],
        method: str = "GET",
        payload_dict: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Helper for executing HTTP requests via http_fetcher or urllib."""
        if self.http_fetcher is not None:
            try:
                return self.http_fetcher(url, headers, method, payload_dict)
            except GitHubIntegrationError:
                raise
            except Exception as e:
                raise GitHubReportingError(f"HTTP request failed: {e.__class__.__name__}")

        body_bytes = json.dumps(payload_dict).encode("utf-8") if payload_dict is not None else None
        req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                resp_bytes = resp.read()
                if not resp_bytes:
                    return {}
                return json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as he:
            raise GitHubReportingError(f"GitHub PR comment API failed with HTTP {he.code}.")
        except Exception as e:
            raise GitHubReportingError(f"GitHub PR comment API network error: {e.__class__.__name__}")

    def post_or_update_comment(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        comment_body: str
    ) -> Dict[str, Any]:
        """
        Posts or updates a GitHub PR comment with the given report body.
        Searches existing issue comments for IDEMPOTENCY_MARKER.
        If found, updates via PATCH; otherwise creates via POST.
        """
        token = self._get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "PatchGuard-Risk-Prediction",
            "Content-Type": "application/json"
        }

        # 1. Search existing comments on PR issue
        comments_url = f"{self.base_url}/repos/{owner}/{repo}/issues/{pull_number}/comments"
        comments_data = self._execute_http(comments_url, headers, method="GET")

        existing_comment_id = None
        if isinstance(comments_data, list):
            for c in comments_data:
                if isinstance(c, dict) and IDEMPOTENCY_MARKER in c.get("body", ""):
                    existing_comment_id = c.get("id")
                    break

        # 2. Update existing comment or Post new comment
        if existing_comment_id:
            patch_url = f"{self.base_url}/repos/{owner}/{repo}/issues/comments/{existing_comment_id}"
            res = self._execute_http(patch_url, headers, method="PATCH", payload_dict={"body": comment_body})
            return {
                "action": "updated",
                "comment_id": existing_comment_id,
                "html_url": res.get("html_url", "") if isinstance(res, dict) else ""
            }
        else:
            res = self._execute_http(comments_url, headers, method="POST", payload_dict={"body": comment_body})
            comment_id = res.get("id") if isinstance(res, dict) else None
            html_url = res.get("html_url", "") if isinstance(res, dict) else ""
            return {
                "action": "created",
                "comment_id": comment_id,
                "html_url": html_url
            }
