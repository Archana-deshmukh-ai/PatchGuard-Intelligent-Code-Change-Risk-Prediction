"""
Canonical feature schema and structured prediction result dataclasses for PatchGuard.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Union

# Exact canonical 10 feature names in mandatory ordering
FEATURE_COLUMNS: List[str] = [
    'lines_added',
    'lines_deleted',
    'code_churn',
    'files_changed',
    'functions_changed',
    'num_directories_touched',
    'is_test_file_modified',
    'avg_lines_changed_per_file',
    'max_lines_changed_in_single_file',
    'num_source_files_changed'
]

@dataclass(frozen=True)
class ModelSignals:
    """
    Quantitative feature contributions to the Logistic Regression model score.
    NOTE: These signals represent mathematical feature contributions (beta_i * z_i) 
    to the model risk score and do NOT constitute causal explanations of code bugs.
    """
    top_positive_factors: Dict[str, float]
    top_negative_factors: Dict[str, float]
    log_odds_contributions: Dict[str, float]

@dataclass(frozen=True)
class PredictionResult:
    """
    Structured, LLM-agnostic output for a commit defect risk prediction.
    """
    # 1. Target Commit Metadata
    full_hash: str
    commit_hash: str
    repo_path: str
    
    # 2. Machine Model Output
    raw_probability: float  # Estimated model risk score in [0.0, 1.0]
    is_above_threshold: bool
    decision_threshold: float
    
    # 3. Developer Presentation (UI Display Labels Only)
    prediction_label: str  # "Elevated Defect Risk" or "Standard Risk Change"
    risk_level: str  # "LOW", "MEDIUM", "HIGH" (Presentation label only)
    presentation_disclaimer: str
    
    # 4. Model Signals & Feature Breakdown
    features: Dict[str, Union[int, float]]
    model_signals: ModelSignals
    
    # 5. Provenance Metadata
    model_id: str
    model_version: str
    feature_schema_version: str
    prediction_timestamp: str
    model_artifact_hash: str
    scaler_artifact_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the prediction result to a standard dictionary format.
        """
        d = asdict(self)
        return d
