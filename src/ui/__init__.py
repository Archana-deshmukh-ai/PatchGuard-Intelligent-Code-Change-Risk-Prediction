"""
PatchGuard UI Package — Presentation Layer for Developer Dashboard (Phase 9).
"""

from .helpers import (
    prepare_signals_chart_data,
    format_features_dataframe,
    get_risk_level_badge,
    format_error_message,
    format_risk_factor_display,
)
from .benchmark_helpers import (
    load_benchmark_data,
    format_benchmark_comparison_records,
    prepare_benchmark_chart_records,
    get_split_integrity_summary,
    get_decision_justification,
)
from .pr_helpers import (
    validate_local_pr_inputs,
    parse_github_pr_identifier,
    is_github_app_configured,
    get_pr_risk_summary_stats,
    format_pr_commits_table,
    format_pr_error,
)

__version__ = "1.0.0"

__all__ = [
    "prepare_signals_chart_data",
    "format_features_dataframe",
    "get_risk_level_badge",
    "format_error_message",
    "format_risk_factor_display",
    "load_benchmark_data",
    "format_benchmark_comparison_records",
    "prepare_benchmark_chart_records",
    "get_split_integrity_summary",
    "get_decision_justification",
    "validate_local_pr_inputs",
    "parse_github_pr_identifier",
    "is_github_app_configured",
    "get_pr_risk_summary_stats",
    "format_pr_commits_table",
    "format_pr_error",
]
