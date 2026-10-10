"""
Pure-Python helper functions for PatchGuard Developer Dashboard (Phase 9.1).
Transforms domain schemas, model signals, and exceptions into presentation-ready
data structures without any Streamlit or UI framework runtime dependencies.
"""

import re
from typing import Dict, List, Any, Optional, Union

from src.engine.schema import FEATURE_COLUMNS, ModelSignals
from src.analysis.schema import RiskFactor, ALLOWED_REVIEW_PRIORITIES
from src.engine.exceptions import (
    PatchGuardError,
    RepositoryError,
    RepositoryNotFoundError,
    InvalidGitRepositoryError,
    CommitError,
    CommitNotFoundError,
    UnsupportedCommitTypeError,
    EmptyCommitError,
    FeatureExtractionError,
    GitExtractionError,
    ModelError,
    ModelArtifactNotFoundError,
    ScalerArtifactNotFoundError,
    CorruptedArtifactError,
    IncompatibleSchemaError,
    InvalidFeatureValueError,
)
from src.analysis.exceptions import (
    AnalysisError,
    MalformedLLMResponseError,
    LLMProviderError,
    OpenAIProviderError,
    MissingAPIKeyError,
    PRAnalysisError,
    InvalidPRRefError,
)
from src.github.exceptions import (
    GitHubIntegrationError,
    GitHubConfigError,
    GitHubAuthError,
    GitHubAPIError,
    GitHubPRNotFoundError,
    GitHubPRValidationError,
    RepositoryAcquisitionError,
    WebhookAuthenticationError,
    WebhookValidationError,
    UnsupportedGitHubEvent,
    PRAnalysisWorkflowError,
    GitHubReportingError,
)


def prepare_signals_chart_data(
    model_signals: Optional[Union[ModelSignals, Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Converts ModelSignals quantitative feature contributions into a list of
    chart-ready dictionary records.

    Preserves exact feature names and numeric contribution values (beta_i * z_i).
    Distinguishes positive contributions (pushing linear score up) from negative
    contributions (pushing linear score down).

    Args:
        model_signals: A ModelSignals instance or equivalent dictionary. If None
                       or empty, an empty list is returned.

    Returns:
        List[Dict[str, Any]]: Records sorted by absolute contribution descending,
        each containing:
            - 'feature': feature name string
            - 'contribution': float log-odds contribution value
            - 'abs_contribution': absolute value of contribution
            - 'direction': 'Positive', 'Negative', or 'Neutral'
            - 'impact': 'Increases Risk', 'Decreases Risk', or 'Neutral'

    Raises:
        TypeError: If model_signals is neither None, a ModelSignals instance, nor a dict.
    """
    if model_signals is None:
        return []

    if isinstance(model_signals, ModelSignals):
        contributions = model_signals.log_odds_contributions
        if not contributions:
            # Fallback to merging top_positive and top_negative if log_odds_contributions is empty
            contributions = {}
            if model_signals.top_positive_factors:
                contributions.update(model_signals.top_positive_factors)
            if model_signals.top_negative_factors:
                contributions.update(model_signals.top_negative_factors)
    elif isinstance(model_signals, dict):
        contributions = model_signals.get("log_odds_contributions")
        if not contributions or not isinstance(contributions, dict):
            contributions = {}
            pos = model_signals.get("top_positive_factors", {})
            neg = model_signals.get("top_negative_factors", {})
            if isinstance(pos, dict):
                contributions.update(pos)
            if isinstance(neg, dict):
                contributions.update(neg)
    else:
        raise TypeError(
            f"Expected ModelSignals or dict, got {type(model_signals).__name__}"
        )

    if not contributions:
        return []

    records: List[Dict[str, Any]] = []
    for feat_name, val in contributions.items():
        val_float = float(val)
        if val_float > 0.0:
            direction = "Positive"
            impact = "Increases Risk"
        elif val_float < 0.0:
            direction = "Negative"
            impact = "Decreases Risk"
        else:
            direction = "Neutral"
            impact = "Neutral"

        records.append(
            {
                "feature": str(feat_name),
                "contribution": val_float,
                "abs_contribution": abs(val_float),
                "direction": direction,
                "impact": impact,
            }
        )

    # Sort primarily by absolute magnitude descending to highlight dominant drivers
    records.sort(key=lambda r: r["abs_contribution"], reverse=True)
    return records


def format_features_dataframe(
    features: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Prepares the canonical 10 commit-time features into an ordered, tabular format
    suitable for UI table rendering or DataFrame conversion.

    Guarantees that the canonical 10 features appear in the exact production
    schema order (1 to 10). If a canonical feature is missing in the input,
    its value is set to None (never silently invented). Any extra unexpected
    keys in features are appended after the canonical 10.

    Canonical 10 Features Order:
        1. lines_added
        2. lines_deleted
        3. code_churn
        4. files_changed
        5. functions_changed
        6. num_directories_touched
        7. is_test_file_modified
        8. avg_lines_changed_per_file
        9. max_lines_changed_in_single_file
        10. num_source_files_changed

    Args:
        features: Dictionary mapping feature names to numerical values.

    Returns:
        List[Dict[str, Any]]: Ordered list of dictionaries with:
            - 'feature': feature name string
            - 'value': feature value or None if missing
            - 'is_canonical': True for canonical 10 features, False for extra keys

    Raises:
        TypeError: If features is None or not a dictionary.
    """
    if features is None or not isinstance(features, dict):
        raise TypeError(f"Features must be a dictionary, got {type(features).__name__}")

    records: List[Dict[str, Any]] = []

    # 1. Enforce exact canonical 10 features and order
    for col in FEATURE_COLUMNS:
        val = features.get(col, None)
        records.append({"feature": col, "value": val, "is_canonical": True})

    # 2. Preserve any extra unexpected keys at the end
    extra_keys = [k for k in features.keys() if k not in FEATURE_COLUMNS]
    for extra_k in extra_keys:
        records.append(
            {"feature": extra_k, "value": features[extra_k], "is_canonical": False}
        )

    return records


def get_risk_level_badge(
    risk_level: Optional[str],
    is_above_threshold: Optional[bool]
) -> Dict[str, Any]:
    """
    Provides UI presentation metadata for risk classification and threshold status.

    Clearly separates the presentation risk level ('LOW', 'MEDIUM', 'HIGH')
    from the production decision threshold cutoff (0.35). Does NOT calculate
    or alter ML predictions.

    Args:
        risk_level: Risk level presentation string ('LOW', 'MEDIUM', 'HIGH').
        is_above_threshold: Boolean indicating whether raw_probability >= 0.35.

    Returns:
        Dict[str, Any]: Presentation dictionary containing:
            - 'risk_level': Normalized uppercase risk level string
            - 'is_above_threshold': Boolean threshold comparison flag
            - 'label': 'Elevated Defect Risk' or 'Standard Risk Change'
            - 'badge_text': Combined icon and risk level text (e.g. '⚠️ HIGH')
            - 'color': 'red', 'amber', 'green', or 'gray'
            - 'icon': '⚠️', '✅', or 'ℹ️'
            - 'threshold_status': Human-readable threshold relation description
            - 'disclaimer': Presentation disclaimer string
    """
    if isinstance(risk_level, str) and risk_level.strip():
        norm_level = risk_level.strip().upper()
    else:
        norm_level = "UNKNOWN"

    above = bool(is_above_threshold) if is_above_threshold is not None else False

    if norm_level == "HIGH":
        color = "red"
        icon = "⚠️" if above else "ℹ️"
    elif norm_level == "MEDIUM":
        color = "amber"
        icon = "⚠️" if above else "ℹ️"
    elif norm_level == "LOW":
        color = "green"
        icon = "✅" if not above else "⚠️"
    else:
        norm_level = "UNKNOWN"
        color = "gray"
        icon = "ℹ️"

    label = "Elevated Defect Risk" if above else "Standard Risk Change"
    badge_text = f"{icon} {norm_level}"
    threshold_status = (
        "Above Threshold (>= 0.35)" if above else "Below Threshold (< 0.35)"
    )
    disclaimer = (
        "Model probabilities represent estimated risk scores and are not calibrated "
        "real-world defect probabilities. Risk levels are presentation labels only."
    )

    return {
        "risk_level": norm_level,
        "is_above_threshold": above,
        "label": label,
        "badge_text": badge_text,
        "color": color,
        "icon": icon,
        "threshold_status": threshold_status,
        "disclaimer": disclaimer,
    }


def _sanitize_message(raw_msg: str) -> str:
    """Removes potential secrets, authorization tokens, or internal path prefixes."""
    if not raw_msg:
        return ""
    # Strip any authorization header mentions
    if "Authorization" in raw_msg:
        raw_msg = raw_msg.split("Authorization")[0].strip()
    # Strip bearer tokens or long hex/base64 hashes if present
    raw_msg = re.sub(r'Bearer\s+[A-Za-z0-9_\-\.]+', '[REDACTED_TOKEN]', raw_msg)
    return raw_msg.strip()


def format_error_message(exception: Exception) -> Dict[str, str]:
    """
    Maps PatchGuard domain exceptions and general errors to user-friendly messages
    with actionable remediation advice.

    Ensures no stack traces, authorization tokens, or sensitive filesystem details
    are leaked to the presentation layer.

    Args:
        exception: The exception instance to format.

    Returns:
        Dict[str, str]: Dictionary containing:
            - 'title': Short descriptive error title
            - 'message': Clear, user-friendly explanation
            - 'remediation': Actionable advice for the user
            - 'error_type': Name of the exception class
    """
    if not isinstance(exception, Exception):
        return {
            "title": "Invalid Error Object",
            "message": "An invalid error object was passed to error formatter.",
            "remediation": "Check application logging for details.",
            "error_type": type(exception).__name__,
        }

    raw_msg = _sanitize_message(str(exception))
    error_type = exception.__class__.__name__

    # 1. Repository Errors
    if isinstance(exception, RepositoryNotFoundError):
        return {
            "title": "Repository Not Found",
            "message": "The specified repository path does not exist on disk.",
            "remediation": "Verify the repository directory path and ensure it is accessible.",
            "error_type": error_type,
        }
    if isinstance(exception, InvalidGitRepositoryError):
        return {
            "title": "Invalid Git Repository",
            "message": "The target directory is not an initialized Git repository.",
            "remediation": "Ensure the directory contains a '.git' directory or clone a valid repository.",
            "error_type": error_type,
        }

    # 2. Commit Errors
    if isinstance(exception, CommitNotFoundError):
        return {
            "title": "Commit Not Found",
            "message": "The specified commit hash, branch, or ref could not be found.",
            "remediation": "Check the commit SHA or branch name (e.g. 'HEAD') and ensure git history is fetched.",
            "error_type": error_type,
        }
    if isinstance(exception, UnsupportedCommitTypeError):
        return {
            "title": "Unsupported Commit Type",
            "message": "Merge commits are excluded from commit-level risk prediction.",
            "remediation": "Select a single-parent non-merge commit for risk evaluation.",
            "error_type": error_type,
        }
    if isinstance(exception, EmptyCommitError):
        return {
            "title": "Empty Commit",
            "message": "The commit contains no modified or touched files to analyze.",
            "remediation": "Select a commit that contains file modifications or code additions/deletions.",
            "error_type": error_type,
        }

    # 3. Model & Artifact Errors
    if isinstance(exception, ModelArtifactNotFoundError):
        return {
            "title": "Model Artifact Missing",
            "message": "The production machine learning model artifact file was not found.",
            "remediation": "Verify that 'models/logistic_regression_real.pkl' exists in the models directory.",
            "error_type": error_type,
        }
    if isinstance(exception, ScalerArtifactNotFoundError):
        return {
            "title": "Scaler Artifact Missing",
            "message": "The production feature preprocessor scaler artifact was not found.",
            "remediation": "Verify that 'models/real_data_scaler.pkl' exists in the models directory.",
            "error_type": error_type,
        }
    if isinstance(exception, CorruptedArtifactError):
        return {
            "title": "Corrupted Artifact",
            "message": "Failed to load model or scaler pickle artifact from disk.",
            "remediation": "Restore the model and scaler pickle artifacts from the verified repository checkpoint.",
            "error_type": error_type,
        }
    if isinstance(exception, IncompatibleSchemaError):
        return {
            "title": "Incompatible Feature Schema",
            "message": "The model or scaler feature schema does not match the canonical 10-feature schema.",
            "remediation": "Ensure feature names and column ordering match the canonical 10 features.",
            "error_type": error_type,
        }
    if isinstance(exception, InvalidFeatureValueError):
        return {
            "title": "Invalid Feature Values",
            "message": "Extracted feature vector contains NaN or non-finite numerical values.",
            "remediation": "Inspect git diff extraction logs for corrupted diffs or unparseable churn values.",
            "error_type": error_type,
        }

    # 4. Feature Extraction & Git Errors
    if isinstance(exception, GitExtractionError):
        return {
            "title": "Git Extraction Failed",
            "message": "An error occurred while executing git commands to extract commit features.",
            "remediation": "Verify that Git is installed, accessible on PATH, and the repository is not locked.",
            "error_type": error_type,
        }

    # 5. LLM Provider & Analysis Errors
    if isinstance(exception, MissingAPIKeyError):
        return {
            "title": "Missing OpenAI API Key",
            "message": "OPENAI_API_KEY environment variable is missing or empty.",
            "remediation": "Set OPENAI_API_KEY in your environment or use the offline 'mock' provider.",
            "error_type": error_type,
        }
    if isinstance(exception, OpenAIProviderError):
        return {
            "title": "OpenAI Provider Error",
            "message": "An error occurred while communicating with the OpenAI API.",
            "remediation": "Check network connectivity, API key validity, rate limits, or use the 'mock' provider.",
            "error_type": error_type,
        }
    if isinstance(exception, MalformedLLMResponseError):
        return {
            "title": "Malformed LLM Response",
            "message": "The qualitative analysis response could not be parsed as valid JSON.",
            "remediation": "Retry the analysis or verify LLM model instructions.",
            "error_type": error_type,
        }
    if isinstance(exception, InvalidPRRefError):
        return {
            "title": "Invalid Pull Request Reference",
            "message": "The base or head git reference is invalid, or no common merge-base was found.",
            "remediation": "Ensure both base and head references exist and share a common commit ancestor.",
            "error_type": error_type,
        }

    # 6. GitHub Integration Errors
    if isinstance(exception, GitHubPRNotFoundError):
        return {
            "title": "GitHub PR Not Found",
            "message": raw_msg or "The requested GitHub Pull Request was not found.",
            "remediation": "Check the repository owner, name, and pull request number.",
            "error_type": error_type,
        }
    if isinstance(exception, (GitHubConfigError, GitHubAuthError)):
        return {
            "title": "GitHub Authentication Failed",
            "message": "GitHub App credentials or installation token authentication failed.",
            "remediation": "Check GITHUB_APP_ID, GITHUB_APP_PRIVATE_KEY, and GITHUB_INSTALLATION_ID settings.",
            "error_type": error_type,
        }
    if isinstance(exception, GitHubIntegrationError):
        return {
            "title": "GitHub Integration Error",
            "message": raw_msg or "A GitHub API or repository integration error occurred.",
            "remediation": "Check GitHub connectivity, permissions, and request arguments.",
            "error_type": error_type,
        }

    # 7. Other PatchGuard Domain Errors
    if isinstance(exception, PatchGuardError):
        return {
            "title": "PatchGuard Domain Error",
            "message": raw_msg or "A domain-level PatchGuard error occurred.",
            "remediation": "Review the input parameters and repository state.",
            "error_type": error_type,
        }

    # 8. Generic / Unknown Exceptions
    return {
        "title": "Unexpected Error",
        "message": "An unexpected error occurred during operation.",
        "remediation": "Check system logs and retry the operation.",
        "error_type": error_type,
    }


def format_risk_factor_display(
    risk_factor: Union[RiskFactor, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Formats a qualitative RiskFactor observation for presentation in the UI.

    Preserves finding description, file path, line range, evidence snippet,
    review priority, and grounding status. Clearly derives grounding from the
    evidence_verified field, ensuring unverified citations are never marked as verified.

    Args:
        risk_factor: A RiskFactor dataclass instance or dictionary conforming to
                     the RiskFactor schema.

    Returns:
        Dict[str, Any]: Presentation dictionary containing:
            - 'category': Risk category string (e.g. 'Error Handling')
            - 'description': Risk description string
            - 'review_priority': 'LOW', 'MEDIUM', or 'HIGH'
            - 'priority_badge': Formatted string (e.g. '[HIGH]')
            - 'file_path': File path string
            - 'line_range': Line range string (e.g. 'L10-L15') or None
            - 'location_display': Combined file path and line range string
            - 'evidence_snippet': Snippet string or None
            - 'evidence_verified': Boolean grounding flag
            - 'verification_badge': '[Verified Grounding]' or '[Ungrounded Citation]'
            - 'verification_status': 'Verified' or 'Ungrounded'

    Raises:
        TypeError: If risk_factor is neither a RiskFactor instance nor a dict.
        ValueError: If review_priority is invalid or required fields are missing.
    """
    if isinstance(risk_factor, RiskFactor):
        category = risk_factor.category
        description = risk_factor.description
        priority = risk_factor.review_priority
        file_path = risk_factor.file_path
        line_range = risk_factor.line_range
        evidence_snippet = risk_factor.evidence_snippet
        evidence_verified = risk_factor.evidence_verified
    elif isinstance(risk_factor, dict):
        category = str(risk_factor.get("category", "General"))
        description = str(risk_factor.get("description", ""))
        priority = str(risk_factor.get("review_priority", "MEDIUM")).strip().upper()
        if priority not in ALLOWED_REVIEW_PRIORITIES:
            raise ValueError(
                f"Invalid review_priority: '{priority}'. Must be one of {sorted(ALLOWED_REVIEW_PRIORITIES)}"
            )
        file_path = str(risk_factor.get("file_path", ""))
        line_range = risk_factor.get("line_range")
        if line_range is not None:
            line_range = str(line_range).strip() or None
        evidence_snippet = str(risk_factor.get("evidence_snippet", ""))
        evidence_verified = bool(risk_factor.get("evidence_verified", True))
    else:
        raise TypeError(
            f"Expected RiskFactor or dict, got {type(risk_factor).__name__}"
        )

    # Derive verification grounding
    is_verified = bool(evidence_verified)
    verification_badge = (
        "[Verified Grounding]" if is_verified else "[Ungrounded Citation]"
    )
    verification_status = "Verified" if is_verified else "Ungrounded"

    # Format location display
    if line_range:
        location_display = f"{file_path} ({line_range})"
    else:
        location_display = file_path

    # Clean evidence snippet
    clean_snippet = evidence_snippet.strip() if evidence_snippet else None

    return {
        "category": category,
        "description": description,
        "review_priority": priority,
        "priority_badge": f"[{priority}]",
        "file_path": file_path,
        "line_range": line_range,
        "location_display": location_display,
        "evidence_snippet": clean_snippet,
        "evidence_verified": is_verified,
        "verification_badge": verification_badge,
        "verification_status": verification_status,
    }
