"""
Automated GitHub PR Workflow Orchestrator for PatchGuard Phase 7.3.
Orchestrates Webhook Event Validation -> PR Metadata Retrieval -> Repository Acquisition
-> PRAnalyzer Execution -> GitHub PR Reporting.
"""

from typing import Optional, Dict, Any

from src.github.schema import GitHubPullRequest
from src.github.client import GitHubClient
from src.github.adapter import GitHubPRAdapter
from src.github.acquisition import RepositoryAcquisitionManager
from src.github.reporter import GitHubPRReporter
from src.github.webhook import WebhookEventHandler, WebhookEventData
from src.analysis.pr_analyzer import PRAnalyzer
from src.analysis.pr_schema import PRAnalysisResult
from src.github.exceptions import (
    GitHubIntegrationError,
    PRAnalysisWorkflowError,
    UnsupportedGitHubEvent,
    WebhookAuthenticationError,
    WebhookValidationError
)


class GitHubPRWorkflow:
    """
    Orchestrates automated risk analysis for incoming GitHub Pull Requests.
    Strictly isolated from ML retraining and LLM prompting details.
    """
    def __init__(
        self,
        client: Optional[GitHubClient] = None,
        acquisition_manager: Optional[RepositoryAcquisitionManager] = None,
        pr_analyzer: Optional[PRAnalyzer] = None,
        reporter: Optional[GitHubPRReporter] = None
    ):
        self.client = client if client is not None else GitHubClient()
        self.acquisition_manager = acquisition_manager if acquisition_manager is not None else RepositoryAcquisitionManager()
        self.pr_analyzer = pr_analyzer if pr_analyzer is not None else PRAnalyzer()
        self.reporter = reporter if reporter is not None else GitHubPRReporter()

    def process_webhook(
        self,
        payload_bytes: bytes,
        headers: Dict[str, str],
        webhook_secret: Optional[str] = None,
        local_repo_override: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processes raw GitHub webhook payload bytes and headers.
        Executes end-to-end PR risk analysis pipeline and publishes PR comment.
        """
        # 1. Parse and validate webhook event & HMAC signature
        event_data: WebhookEventData = WebhookEventHandler.parse_event(
            payload_bytes=payload_bytes,
            headers=headers,
            webhook_secret=webhook_secret
        )

        gh_pr = event_data.pull_request
        delivery_id = event_data.delivery_id
        action = event_data.action

        # 2. Retrieve latest PR metadata using GitHubClient if available (or use payload PR metadata)
        latest_gh_pr = gh_pr
        try:
            latest_gh_pr = self.client.get_pull_request(
                owner=gh_pr.owner,
                repo=gh_pr.repository,
                pull_number=gh_pr.number
            )
        except Exception:
            # If API client fails or in offline mock test mode without credentials, fallback to payload gh_pr
            latest_gh_pr = gh_pr

        # 3. Get access token for Git transport if available
        token = None
        try:
            token = self.client._get_token()
        except Exception:
            pass

        # 4. Acquire repository workspace and execute PR analysis
        try:
            with self.acquisition_manager.acquire(
                owner=latest_gh_pr.owner,
                repo=latest_gh_pr.repository,
                base_sha=latest_gh_pr.base_sha,
                head_sha=latest_gh_pr.head_sha,
                token=token,
                local_repo_override=local_repo_override
            ) as acquired_repo_path:
                # 5. Adapt to PRAnalysisInput
                pr_input = GitHubPRAdapter.to_analysis_input(latest_gh_pr, repo_path=acquired_repo_path)

                # 6. Execute PRAnalyzer (Phase 5 ML + Phase 6 LLM)
                analysis_result: PRAnalysisResult = self.pr_analyzer.analyze(pr_input, threshold=threshold)

                # 7. Format markdown report and publish/update GitHub comment
                report_text = self.reporter.format_report(analysis_result, gh_pr=latest_gh_pr)
                report_res = self.reporter.post_or_update_comment(
                    owner=latest_gh_pr.owner,
                    repo=latest_gh_pr.repository,
                    pull_number=latest_gh_pr.number,
                    comment_body=report_text
                )

                return {
                    "status": "success",
                    "delivery_id": delivery_id,
                    "action": action,
                    "github_pr": latest_gh_pr.to_dict(),
                    "analysis_summary": {
                        "commit_count": analysis_result.commit_count,
                        "is_empty": analysis_result.is_empty,
                        "resolved_base_sha": analysis_result.resolved_base_sha,
                        "resolved_head_sha": analysis_result.resolved_head_sha
                    },
                    "reporting": report_res
                }
        except GitHubIntegrationError:
            raise
        except Exception as ex:
            raise PRAnalysisWorkflowError(f"Automated PR workflow failed: {ex}")
