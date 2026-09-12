"""
Risk Prediction Engine implementation for PatchGuard.
Inference-only module for predicting commit risk using trained Logistic Regression models.
"""

import os
import hashlib
import datetime
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

from src.git_extractor import GitRepositoryExtractor
from .schema import FEATURE_COLUMNS, ModelSignals, PredictionResult
from .exceptions import (
    RepositoryNotFoundError,
    InvalidGitRepositoryError,
    CommitNotFoundError,
    UnsupportedCommitTypeError,
    EmptyCommitError,
    ModelArtifactNotFoundError,
    ScalerArtifactNotFoundError,
    CorruptedArtifactError,
    IncompatibleSchemaError,
    InvalidFeatureValueError
)

def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

class RiskPredictionEngine:
    """
    Inference-only engine that converts a Git commit into a structured risk prediction.
    """
    MODEL_ID = "logistic_regression_real"
    MODEL_VERSION = "v1.0.0-phase4"
    FEATURE_SCHEMA_VERSION = "v1.0-10features"
    PRESENTATION_DISCLAIMER = (
        "Model probabilities represent estimated risk scores and are not calibrated real-world defect probabilities. "
        "Risk levels (LOW/MEDIUM/HIGH) are presentation labels only for UI display."
    )

    def __init__(self, models_dir: Optional[str] = None, default_threshold: float = 0.35):
        if models_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'models'))
            
        self.models_dir = os.path.abspath(models_dir)
        self.default_threshold = default_threshold
        
        self.model_path = os.path.join(self.models_dir, 'logistic_regression_real.pkl')
        self.scaler_path = os.path.join(self.models_dir, 'real_data_scaler.pkl')
        
        if not os.path.exists(self.model_path):
            raise ModelArtifactNotFoundError(f"Model artifact not found at: {self.model_path}")
            
        if not os.path.exists(self.scaler_path):
            raise ScalerArtifactNotFoundError(f"Scaler artifact not found at: {self.scaler_path}")
            
        # Compute SHA-256 hashes for provenance
        self.model_artifact_hash = compute_file_sha256(self.model_path)
        self.scaler_artifact_hash = compute_file_sha256(self.scaler_path)
        
        # Load artifacts
        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            raise CorruptedArtifactError(f"Failed to load model artifact from {self.model_path}: {e}")
            
        try:
            self.scaler = joblib.load(self.scaler_path)
        except Exception as e:
            raise CorruptedArtifactError(f"Failed to load scaler artifact from {self.scaler_path}: {e}")
            
        # Validate feature counts
        if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != len(FEATURE_COLUMNS):
            raise IncompatibleSchemaError(
                f"Scaler feature count mismatch: expected {len(FEATURE_COLUMNS)}, found {self.scaler.n_features_in_}"
            )
            
        if hasattr(self.model, 'n_features_in_') and self.model.n_features_in_ != len(FEATURE_COLUMNS):
            raise IncompatibleSchemaError(
                f"Model feature count mismatch: expected {len(FEATURE_COLUMNS)}, found {self.model.n_features_in_}"
            )

        # Validate exact canonical feature names if stored in scaler
        if hasattr(self.scaler, 'feature_names_in_'):
            scaler_cols = list(self.scaler.feature_names_in_)
            if scaler_cols != FEATURE_COLUMNS:
                raise IncompatibleSchemaError(
                    f"Scaler feature names mismatch. Expected {FEATURE_COLUMNS}, found {scaler_cols}"
                )

    def predict(
        self,
        repo_path: str,
        commit_hash: str = "HEAD",
        threshold: Optional[float] = None
    ) -> PredictionResult:
        """
        Executes risk prediction for a specific commit in a Git repository.
        """
        repo_path_abs = os.path.abspath(repo_path)
        if not os.path.exists(repo_path_abs):
            raise RepositoryNotFoundError(f"Repository directory does not exist: {repo_path_abs}")
            
        try:
            extractor = GitRepositoryExtractor(repo_path_abs)
        except ValueError as ve:
            raise InvalidGitRepositoryError(str(ve))
            
        # Extract single commit details
        try:
            extracted = extractor.extract_single_commit(commit_hash)
        except ValueError as ve:
            raise CommitNotFoundError(str(ve))
            
        # Check merge commit status
        parents = extracted.get('parents', [])
        if len(parents) > 1:
            raise UnsupportedCommitTypeError(
                f"Commit {extracted['commit_hash']} is a merge commit (parents: {len(parents)}). "
                "Merge commits are excluded from risk prediction."
            )
            
        # Check empty commit status
        if extracted['files_changed'] == 0:
            raise EmptyCommitError(f"Commit {extracted['commit_hash']} contains no file changes to evaluate.")
            
        # Extract raw features and enforce exact canonical schema & ordering
        raw_features_dict = {col: extracted[col] for col in FEATURE_COLUMNS}
        df_raw = pd.DataFrame([raw_features_dict])[FEATURE_COLUMNS]
        
        # Verify numerical validity
        if df_raw.isna().any().any() or not np.isfinite(df_raw.values).all():
            raise InvalidFeatureValueError("Extracted feature vector contains NaN or non-finite values.")
            
        # Scale features using training-fitted StandardScaler
        X_scaled = self.scaler.transform(df_raw)
        
        # Inference
        raw_prob = float(self.model.predict_proba(X_scaled)[0, 1])
        
        # Decision threshold evaluation
        decision_threshold = threshold if threshold is not None else self.default_threshold
        is_above_thresh = raw_prob >= decision_threshold
        
        # Calculate quantitative log-odds contributions (beta_i * z_i)
        coefs = self.model.coef_[0]
        scaled_vals = X_scaled[0]
        contributions = {col: float(coefs[i] * scaled_vals[i]) for i, col in enumerate(FEATURE_COLUMNS)}
        
        # Top positive (pushing risk score up) and negative (pushing risk score down)
        pos_factors = {k: round(v, 4) for k, v in sorted(contributions.items(), key=lambda item: item[1], reverse=True) if v > 0}
        neg_factors = {k: round(v, 4) for k, v in sorted(contributions.items(), key=lambda item: item[1]) if v < 0}
        
        model_signals = ModelSignals(
            top_positive_factors=pos_factors,
            top_negative_factors=neg_factors,
            log_odds_contributions={k: round(v, 4) for k, v in contributions.items()}
        )
        
        # Presentation Risk Level (UI Display Labels Only)
        if raw_prob < 0.35:
            risk_level = "LOW"
        elif raw_prob < 0.60:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
            
        prediction_label = "Elevated Defect Risk" if is_above_thresh else "Standard Risk Change"
        timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        return PredictionResult(
            full_hash=extracted['full_hash'],
            commit_hash=extracted['commit_hash'],
            repo_path=repo_path_abs,
            raw_probability=round(raw_prob, 4),
            is_above_threshold=is_above_thresh,
            decision_threshold=decision_threshold,
            prediction_label=prediction_label,
            risk_level=risk_level,
            presentation_disclaimer=self.PRESENTATION_DISCLAIMER,
            features=raw_features_dict,
            model_signals=model_signals,
            model_id=self.MODEL_ID,
            model_version=self.MODEL_VERSION,
            feature_schema_version=self.FEATURE_SCHEMA_VERSION,
            prediction_timestamp=timestamp_utc,
            model_artifact_hash=self.model_artifact_hash,
            scaler_artifact_hash=self.scaler_artifact_hash
        )
