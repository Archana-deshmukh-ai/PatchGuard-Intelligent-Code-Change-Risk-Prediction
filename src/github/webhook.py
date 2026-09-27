"""
GitHub Webhook signature verification and event handling for PatchGuard Phase 7.3.
Verifies HMAC-SHA256 signatures over raw request body bytes and parses pull_request events.
"""

import hmac
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Dict, Any, Optional

from src.github.schema import GitHubPullRequest
from src.github.exceptions import (
    WebhookAuthenticationError,
    WebhookValidationError,
    UnsupportedGitHubEvent
)

SHA1_REGEX = re.compile(r'^[0-9a-fA-F]{40}$')
REPO_NAME_REGEX = re.compile(r'^[a-zA-Z0-9_.-]+$')

SUPPORTED_ACTIONS = {"opened", "synchronize", "reopened"}


@dataclass(frozen=True)
class WebhookEventData:
    """Structured representation of a parsed GitHub webhook event."""
    delivery_id: str
    event_type: str
    action: str
    pull_request: GitHubPullRequest


def verify_webhook_signature(
    payload_bytes: bytes,
    signature_header: Optional[str],
    secret: str
) -> bool:
    """
    Verifies GitHub webhook payload signature using HMAC-SHA256 and constant-time comparison.
    
    :param payload_bytes: Raw HTTP request body bytes before JSON decoding.
    :param signature_header: Value of X-Hub-Signature-256 header (e.g. 'sha256=...').
    :param secret: Webhook secret key.
    :return: True if signature is valid, False otherwise.
    """
    if not secret or not signature_header or not payload_bytes:
        return False

    if not signature_header.startswith("sha256="):
        return False

    provided_sig = signature_header[7:].strip()
    if not provided_sig:
        return False

    try:
        expected_sig = hmac.new(
            secret.encode("utf-8"),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected_sig.lower(), provided_sig.lower())
    except Exception:
        return False


class WebhookEventHandler:
    """
    Parses and validates incoming GitHub webhook events for PatchGuard automated workflow.
    """
    @staticmethod
    def parse_event(
        payload_bytes: bytes,
        headers: Dict[str, str],
        webhook_secret: Optional[str] = None
    ) -> WebhookEventData:
        """
        Parses raw HTTP request payload and headers into structured WebhookEventData metadata.
        Enforces HMAC-SHA256 signature verification if webhook_secret is provided.
        Filters for supported pull_request events ('opened', 'synchronize', 'reopened').
        """
        # Normalize header keys to lowercase
        norm_headers = {k.lower(): v for k, v in headers.items()}

        # 1. HMAC Signature Verification (BEFORE JSON Parsing)
        if webhook_secret:
            sig_header = norm_headers.get("x-hub-signature-256")
            if not sig_header:
                raise WebhookAuthenticationError(
                    "Missing 'X-Hub-Signature-256' header in webhook request."
                )
            if not verify_webhook_signature(payload_bytes, sig_header, webhook_secret):
                raise WebhookAuthenticationError(
                    "GitHub webhook HMAC-SHA256 signature verification failed."
                )

        # 2. GitHub Event Type Verification & Delivery ID extraction
        delivery_id = norm_headers.get("x-github-delivery", "")
        event_type = norm_headers.get("x-github-event", "")
        if event_type != "pull_request":
            raise UnsupportedGitHubEvent(
                f"Unsupported GitHub event type: '{event_type}'. Only 'pull_request' events are supported."
            )

        # 3. Payload JSON Decoding
        try:
            data = json.loads(payload_bytes.decode("utf-8"))
        except Exception as ex:
            raise WebhookValidationError(
                f"Malformed JSON payload in webhook request: {ex}"
            )

        if not isinstance(data, dict):
            raise WebhookValidationError("Webhook payload expected JSON object.")

        # 4. Action & State Filtering
        action = data.get("action", "")
        if action not in SUPPORTED_ACTIONS:
            raise UnsupportedGitHubEvent(
                f"Pull request action '{action}' is not supported for risk analysis. "
                f"Supported actions: {', '.join(sorted(SUPPORTED_ACTIONS))}."
            )

        pr_data = data.get("pull_request")
        if not isinstance(pr_data, dict):
            raise WebhookValidationError("Webhook payload missing 'pull_request' object.")

        pr_state = pr_data.get("state", "open")
        if pr_state != "open":
            raise UnsupportedGitHubEvent(
                f"Pull request state is '{pr_state}'. Only 'open' pull requests are analyzed."
            )

        # 5. Extract PR Metadata
        base_data = pr_data.get("base", {})
        head_data = pr_data.get("head", {})

        owner = (
            base_data.get("repo", {}).get("owner", {}).get("login") or
            head_data.get("repo", {}).get("owner", {}).get("login") or
            ""
        )
        repo = base_data.get("repo", {}).get("name") or head_data.get("repo", {}).get("name") or ""
        number = pr_data.get("number", 0)
        title = pr_data.get("title", "")
        html_url = pr_data.get("html_url", "")
        base_ref = base_data.get("ref", "")
        base_sha = base_data.get("sha", "")
        head_ref = head_data.get("ref", "")
        head_sha = head_data.get("sha", "")

        # 6. Validate Metadata Attributes
        if not owner or not REPO_NAME_REGEX.match(owner):
            raise WebhookValidationError(f"Invalid or missing repository owner in webhook payload: '{owner}'")
        if not repo or not REPO_NAME_REGEX.match(repo):
            raise WebhookValidationError(f"Invalid or missing repository name in webhook payload: '{repo}'")
        if not isinstance(number, int) or number <= 0:
            raise WebhookValidationError(f"Invalid PR number in webhook payload: '{number}'")

        if not base_sha or not SHA1_REGEX.match(base_sha):
            raise WebhookValidationError(f"Invalid base commit SHA in webhook payload: '{base_sha}'")
        if not head_sha or not SHA1_REGEX.match(head_sha):
            raise WebhookValidationError(f"Invalid head commit SHA in webhook payload: '{head_sha}'")

        gh_pr = GitHubPullRequest(
            owner=owner,
            repository=repo,
            number=number,
            title=title,
            html_url=html_url,
            base_ref=base_ref,
            base_sha=base_sha,
            head_ref=head_ref,
            head_sha=head_sha,
            state=pr_state
        )

        return WebhookEventData(
            delivery_id=delivery_id,
            event_type=event_type,
            action=action,
            pull_request=gh_pr
        )
