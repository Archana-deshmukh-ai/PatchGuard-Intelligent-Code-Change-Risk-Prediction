"""
CLI module for PatchGuard Risk Prediction Engine.
"""

import sys
import os
import json
import argparse
from typing import List, Optional

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.engine import RiskPredictionEngine, PatchGuardError, PredictionResult

def render_text_card(res: PredictionResult) -> str:
    """Renders human-readable text report card for a prediction result."""
    lines = []
    lines.append("==========================================================================")
    lines.append("                 PATCHGUARD CODE CHANGE RISK PREDICTION                   ")
    lines.append("==========================================================================")
    lines.append(f"  Target Commit          : {res.commit_hash} ({res.full_hash[:12]})")
    lines.append(f"  Repository Path        : {res.repo_path}")
    lines.append(f"  Prediction Timestamp   : {res.prediction_timestamp}")
    lines.append("--------------------------------------------------------------------------")
    lines.append("--- MODEL INFERENCE & DECISION ---")
    lines.append(f"  Estimated Risk Score   : {res.raw_probability:.4f}")
    lines.append(f"  Decision Threshold     : {res.decision_threshold:.2f}")
    lines.append(f"  Above Threshold        : {res.is_above_threshold}")
    lines.append(f"  Prediction Label       : {res.prediction_label}")
    lines.append(f"  Risk Level (UI Label)  : {res.risk_level}")
    lines.append("--------------------------------------------------------------------------")
    lines.append("--- QUANTITATIVE MODEL SIGNALS (Feature Contributions to Score) ---")
    if res.model_signals.top_positive_factors:
        lines.append("  Top Factors Pushing Risk Score Up:")
        for feat, val in res.model_signals.top_positive_factors.items():
            lines.append(f"    - {feat:<34} : +{val:.4f}")
    else:
        lines.append("  Top Factors Pushing Risk Score Up: None")
        
    if res.model_signals.top_negative_factors:
        lines.append("  Top Factors Pushing Risk Score Down:")
        for feat, val in res.model_signals.top_negative_factors.items():
            lines.append(f"    - {feat:<34} : {val:.4f}")
    else:
        lines.append("  Top Factors Pushing Risk Score Down: None")
    lines.append("--------------------------------------------------------------------------")
    lines.append("--- EXTRACTED COMMIT FEATURES ---")
    for feat_name, feat_val in res.features.items():
        lines.append(f"  - {feat_name:<36} : {feat_val}")
    lines.append("--------------------------------------------------------------------------")
    lines.append("--- AUDIT PROVENANCE ---")
    lines.append(f"  Model ID               : {res.model_id} ({res.model_version})")
    lines.append(f"  Feature Schema Version : {res.feature_schema_version}")
    lines.append(f"  Model Artifact SHA-256 : {res.model_artifact_hash[:16]}...")
    lines.append(f"  Scaler Artifact SHA-256: {res.scaler_artifact_hash[:16]}...")
    lines.append("==========================================================================")
    lines.append(f"DISCLAIMER: {res.presentation_disclaimer}")
    lines.append("==========================================================================\n")
    return "\n".join(lines)

def main(args_list: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="patchguard",
        description="PatchGuard: Intelligent Code Change Risk Prediction CLI"
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")
    
    predict_parser = subparsers.add_parser("predict", help="Predict risk for a specific commit")
    predict_parser.add_argument("--repo", "-r", required=True, help="Path to local Git repository")
    predict_parser.add_argument("--commit", "-c", default="HEAD", help="Commit SHA, branch, or ref (default: HEAD)")
    predict_parser.add_argument("--threshold", "-t", type=float, default=None, help="Decision threshold override (default: 0.35)")
    predict_parser.add_argument("--format", "-f", choices=["text", "json"], default="text", help="Output format (default: text)")
    predict_parser.add_argument("--models-dir", default=None, help="Directory containing model artifacts")
    
    parsed = parser.parse_args(args_list)
    
    if not parsed.subcommand:
        parser.print_help()
        return 1
        
    if parsed.subcommand == "predict":
        try:
            engine = RiskPredictionEngine(models_dir=parsed.models_dir)
            result = engine.predict(
                repo_path=parsed.repo,
                commit_hash=parsed.commit,
                threshold=parsed.threshold
            )
            
            if parsed.format == "json":
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print(render_text_card(result))
            return 0
            
        except PatchGuardError as pge:
            sys.stderr.write(f"[PatchGuard Error] {pge.__class__.__name__}: {pge}\n")
            return 1
        except Exception as e:
            sys.stderr.write(f"[Unexpected Error] {e.__class__.__name__}: {e}\n")
            return 2

    return 0

if __name__ == "__main__":
    sys.exit(main())
