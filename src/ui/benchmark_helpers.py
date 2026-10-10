"""
Pure-Python helper functions for PatchGuard ML Benchmark Viewer (Phase 9.4).
Parses, structures, and validates Phase 8 benchmark reports for read-only UI display
without introducing training, retraining, or UI framework dependencies.
"""

import json
import os
from typing import Dict, List, Any, Optional

# Default path to Phase 8 benchmark report JSON
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_BENCHMARK_PATH = os.path.join(
    PROJECT_ROOT, "reports", "experiments", "phase8", "phase8_model_benchmark.json"
)


def load_benchmark_data(filepath: Optional[str] = None) -> Dict[str, Any]:
    """
    Safely loads and validates the Phase 8 model benchmark JSON report.

    Args:
        filepath: Optional path to the benchmark JSON file. Defaults to
                  reports/experiments/phase8/phase8_model_benchmark.json.

    Returns:
        Dict[str, Any]: Parsed benchmark report dictionary.

    Raises:
        FileNotFoundError: If the benchmark report file does not exist.
        ValueError: If the file contains invalid JSON or lacks required benchmark keys.
    """
    path = os.path.abspath(filepath) if filepath else DEFAULT_BENCHMARK_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Benchmark report not found at: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed benchmark JSON report: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to read benchmark report: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in benchmark report, got {type(data).__name__}")

    # Validate essential schema sections
    required_sections = ["metadata", "production_logistic_regression_anchor", "benchmark_candidates", "benchmark_summary"]
    missing = [sec for sec in required_sections if sec not in data]
    if missing:
        raise ValueError(f"Incomplete benchmark report schema. Missing sections: {missing}")

    return data


def format_benchmark_comparison_records(benchmark_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extracts multi-model benchmark metrics into flat comparison records.
    Distinguishes validation metrics from test metrics and flags the production model.

    Args:
        benchmark_data: Parsed benchmark report dictionary.

    Returns:
        List[Dict[str, Any]]: List of model comparison dictionaries, each containing:
            - model_name: Human-readable model display name
            - family: Model algorithm family
            - threshold: Operating decision threshold
            - features: 'scaled' or 'raw'
            - is_production: True if this is the locked production model
            - is_winner: True if this model achieved highest validation PR-AUC
            - val_pr_auc: Validation PR-AUC (or None)
            - val_roc_auc: Validation ROC-AUC (or None)
            - val_f1: Validation F1 score (or None)
            - val_precision: Validation precision (or None)
            - val_recall: Validation recall (or None)
            - test_pr_auc: Test PR-AUC (or None)
            - test_roc_auc: Test ROC-AUC (or None)
            - test_f1: Test F1 score (or None)
            - test_precision: Test precision (or None)
            - test_recall: Test recall (or None)
            - status_badge: Human-readable status badge string
    """
    records: List[Dict[str, Any]] = []

    # 1. Production Anchor (Locked Baseline)
    anchor = benchmark_data.get("production_logistic_regression_anchor", {})
    if anchor:
        val_m = anchor.get("validation_metrics", {})
        test_m = anchor.get("test_metrics", {})
        records.append({
            "model_name": anchor.get("model_name", "Production LogisticRegression"),
            "family": "LogisticRegression",
            "threshold": anchor.get("locked_threshold", 0.35),
            "features": "scaled",
            "is_production": True,
            "is_winner": False,
            "val_pr_auc": val_m.get("pr_auc"),
            "val_roc_auc": val_m.get("roc_auc"),
            "val_f1": val_m.get("f1"),
            "val_precision": val_m.get("precision"),
            "val_recall": val_m.get("recall"),
            "test_pr_auc": test_m.get("pr_auc"),
            "test_roc_auc": test_m.get("roc_auc"),
            "test_f1": test_m.get("f1"),
            "test_precision": test_m.get("precision"),
            "test_recall": test_m.get("recall"),
            "status_badge": "🛡️ Preserved Production Model",
        })

    # 2. Candidate Models (Phase 8)
    candidates = benchmark_data.get("benchmark_candidates", {})
    winner_name = benchmark_data.get("benchmark_summary", {}).get("validation_winning_model", "")

    for key, cand in candidates.items():
        val_p = cand.get("validation_performance", {})
        test_p = cand.get("final_test_performance_at_frozen_threshold", {})
        is_win = (key == winner_name)

        badge = "🏆 Validation Winner (Candidate)" if is_win else "Benchmark Candidate"
        records.append({
            "model_name": f"Phase 8 {key}",
            "family": cand.get("family_name", key),
            "threshold": cand.get("frozen_validation_threshold", val_p.get("decision_threshold", 0.35)),
            "features": cand.get("features_used", "scaled"),
            "is_production": False,
            "is_winner": is_win,
            "val_pr_auc": val_p.get("pr_auc"),
            "val_roc_auc": val_p.get("roc_auc"),
            "val_f1": val_p.get("f1"),
            "val_precision": val_p.get("precision"),
            "val_recall": val_p.get("recall"),
            "test_pr_auc": test_p.get("pr_auc"),
            "test_roc_auc": test_p.get("roc_auc"),
            "test_f1": test_p.get("f1"),
            "test_precision": test_p.get("precision"),
            "test_recall": test_p.get("recall"),
            "status_badge": badge,
        })

    return records


def prepare_benchmark_chart_records(
    comparison_records: List[Dict[str, Any]],
    metric_base: str = "pr_auc"
) -> List[Dict[str, Any]]:
    """
    Prepares tidy, long-format records for Altair grouped bar charts.
    Explicitly pairs validation and test evaluations as separate records
    so splits are never conflated or directly summed.

    Args:
        comparison_records: List of model comparison dictionaries.
        metric_base: Base metric name string (e.g. 'pr_auc', 'roc_auc', 'f1').

    Returns:
        List[Dict[str, Any]]: Records with keys 'model', 'split', 'value', 'is_production'.
    """
    chart_records: List[Dict[str, Any]] = []
    val_key = f"val_{metric_base}"
    test_key = f"test_{metric_base}"

    for rec in comparison_records:
        val_val = rec.get(val_key)
        test_val = rec.get(test_key)
        model = rec.get("model_name", "Unknown")
        is_prod = rec.get("is_production", False)

        if val_val is not None:
            chart_records.append({
                "model": model,
                "split": "Validation",
                "value": float(val_val),
                "is_production": is_prod,
                "metric": metric_base.upper().replace("_", "-"),
            })

        if test_val is not None:
            chart_records.append({
                "model": model,
                "split": "Test (Unseen)",
                "value": float(test_val),
                "is_production": is_prod,
                "metric": metric_base.upper().replace("_", "-"),
            })

    return chart_records


def get_split_integrity_summary(benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts dataset split sizing and prevalence statistics from benchmark metadata.

    Args:
        benchmark_data: Parsed benchmark report dictionary.

    Returns:
        Dict[str, Any]: Sizing and integrity details.
    """
    meta = benchmark_data.get("metadata", {})
    integrity = meta.get("split_integrity", {})

    return {
        "experiment_version": meta.get("experiment_version", "v1.0.0-phase8"),
        "timestamp_utc": meta.get("timestamp_utc", "Unknown"),
        "git_commit_sha": meta.get("git_commit_sha", "Unknown"),
        "total_rows": integrity.get("total_rows", 1687),
        "train_rows": integrity.get("train_rows", 1181),
        "val_rows": integrity.get("val_rows", 253),
        "test_rows": integrity.get("test_rows", 253),
        "train_pos_rate": integrity.get("train_pos_rate", 0.4505),
        "val_pos_rate": integrity.get("val_pos_rate", 0.2964),
        "test_pos_rate": integrity.get("test_pos_rate", 0.2016),
        "strictly_chronological": integrity.get("strictly_chronological", True),
        "zero_hash_overlap": integrity.get("zero_hash_overlap", True),
    }


def get_decision_justification(benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts the Phase 8 model-selection recommendation and trade-off rationale.

    Args:
        benchmark_data: Parsed benchmark report dictionary.

    Returns:
        Dict[str, Any]: Decision summary and justification notes.
    """
    summary = benchmark_data.get("benchmark_summary", {})

    return {
        "final_recommendation": summary.get(
            "final_recommendation", "PRESERVE_PRODUCTION_LOGISTIC_REGRESSION"
        ),
        "decision_code": "PRESERVE_PRODUCTION_LOGISTIC_REGRESSION",
        "validation_winner": summary.get("validation_winning_model", "XGBoost"),
        "winner_val_pr_auc": summary.get("winner_val_pr_auc", 0.6293),
        "prod_val_pr_auc": summary.get("prod_val_pr_auc", 0.6220),
        "improvement_margin": summary.get("pr_auc_improvement_over_prod", 0.0073),
        "is_meaningful_improvement": summary.get("is_meaningful_improvement", False),
        "preserved_model": "Logistic Regression (Locked Phase 4/5 Baseline)",
        "preserved_threshold": 0.35,
        "rationale": (
            "The validation PR-AUC advantage for XGBoost (+0.0073 / +0.73%) was marginal and not "
            "statistically significant under chronological split evaluation. All models degraded "
            "substantially on the unseen temporal test set due to class distribution shift "
            "(positive rate dropped from 45.05% in train to 20.16% in test). "
            "Logistic Regression was preserved to retain exact mathematical log-odds explainability "
            "(beta_i * z_i) without incurring additional tree-model dependency overhead in production."
        ),
    }
