"""
CLI module for PatchGuard Risk Prediction & Code Change Analysis Engine.
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
from src.analysis import (
    LLMCodeAnalyzer,
    MockLLMProvider,
    OpenAIProvider,
    LLMAnalysisResult,
    LLMProvider
)

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


def render_analysis_text_card(prediction: PredictionResult, analysis: LLMAnalysisResult) -> str:
    """Renders human-readable text report card for combined ML risk prediction and LLM analysis."""
    lines = []
    lines.append("==========================================================================")
    lines.append("                 PATCHGUARD INTELLIGENT CODE CHANGE ANALYSIS               ")
    lines.append("==========================================================================")
    lines.append(f"  Target Commit          : {prediction.commit_hash} ({prediction.full_hash[:12]})")
    lines.append(f"  Repository Path        : {prediction.repo_path}")
    lines.append(f"  Prediction Timestamp   : {prediction.prediction_timestamp}")
    lines.append("--------------------------------------------------------------------------")
    lines.append("--- ML DEFECT RISK PREDICTION (Phase 5 Engine) ---")
    lines.append(f"  Estimated Risk Score   : {prediction.raw_probability:.4f}")
    lines.append(f"  Decision Threshold     : {prediction.decision_threshold:.2f}")
    lines.append(f"  Above Threshold        : {prediction.is_above_threshold}")
    lines.append(f"  Prediction Label       : {prediction.prediction_label}")
    lines.append(f"  Risk Level (UI Label)  : {prediction.risk_level}")
    lines.append("--------------------------------------------------------------------------")
    lines.append("--- QUALITATIVE CODE CHANGE ANALYSIS (Phase 6 LLM Engine) ---")
    lines.append(f"  Model Provider         : {analysis.model_provider} ({analysis.model_name})")
    lines.append("")
    lines.append("  Summary:")
    lines.append(f"    {analysis.summary}")
    lines.append("")
    lines.append("  Key Changes:")
    if analysis.key_changes:
        for item in analysis.key_changes:
            lines.append(f"    - {item}")
    else:
        lines.append("    - None reported.")
    lines.append("")
    lines.append("  Potential Risk Factors:")
    if analysis.potential_risk_factors:
        for rf in analysis.potential_risk_factors:
            line_str = f" (Line Range: {rf.line_range})" if rf.line_range else ""
            verified_str = "Verified" if rf.evidence_verified else "Unverified/Ungrounded"
            lines.append(f"    - [{rf.review_priority}] {rf.category}: {rf.description}")
            lines.append(f"      File    : {rf.file_path}{line_str}")
            if rf.evidence_snippet:
                lines.append(f"      Evidence: \"{rf.evidence_snippet}\" [{verified_str}]")
            else:
                lines.append(f"      Evidence: [No snippet provided] [{verified_str}]")
    else:
        lines.append("    - None reported.")
    lines.append("")
    lines.append("  Affected Areas:")
    if analysis.affected_areas:
        for area in analysis.affected_areas:
            lines.append(f"    - {area}")
    else:
        lines.append("    - None reported.")
    lines.append("")
    lines.append("  Testing Observations:")
    if analysis.testing_observations:
        for obs in analysis.testing_observations:
            lines.append(f"    - {obs}")
    else:
        lines.append("    - None reported.")
    lines.append("")
    lines.append("  Recommended Review Actions:")
    if analysis.recommended_review_actions:
        for ra in analysis.recommended_review_actions:
            target_str = f" (Target File: {ra.target_file})" if ra.target_file else ""
            lines.append(f"    - {ra.action}{target_str}")
    else:
        lines.append("    - None reported.")
    lines.append("")
    lines.append("  Confidence & Audit Notes:")
    lines.append(f"    {analysis.confidence_notes or 'None.'}")
    lines.append("==========================================================================")
    lines.append(f"DISCLAIMER: {prediction.presentation_disclaimer}")
    lines.append("==========================================================================\n")
    return "\n".join(lines)


def main(args_list: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="patchguard",
        description="PatchGuard: Intelligent Code Change Risk Prediction & Analysis CLI"
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")
    
    # Subcommand: predict
    predict_parser = subparsers.add_parser("predict", help="Predict risk for a specific commit using ML model")
    predict_parser.add_argument("--repo", "-r", required=True, help="Path to local Git repository")
    predict_parser.add_argument("--commit", "-c", default="HEAD", help="Commit SHA, branch, or ref (default: HEAD)")
    predict_parser.add_argument("--threshold", "-t", type=float, default=None, help="Decision threshold override (default: 0.35)")
    predict_parser.add_argument("--format", "-f", choices=["text", "json"], default="text", help="Output format (default: text)")
    predict_parser.add_argument("--models-dir", default=None, help="Directory containing model artifacts")

    # Subcommand: analyze
    analyze_parser = subparsers.add_parser("analyze", help="Run combined ML risk prediction and qualitative LLM code analysis")
    analyze_parser.add_argument("--repo", "-r", required=True, help="Path to local Git repository")
    analyze_parser.add_argument("--commit", "-c", default="HEAD", help="Commit SHA, branch, or ref (default: HEAD)")
    analyze_parser.add_argument("--provider", "-p", choices=["mock", "openai"], default="mock", help="LLM provider to use (default: mock)")
    analyze_parser.add_argument("--model", "-m", default=None, help="Optional model name override for provider (e.g. gpt-4o)")
    analyze_parser.add_argument("--threshold", "-t", type=float, default=None, help="ML decision threshold override (default: 0.35)")
    analyze_parser.add_argument("--format", "-f", choices=["text", "json"], default="text", help="Output format (default: text)")
    analyze_parser.add_argument("--models-dir", default=None, help="Directory containing ML model artifacts")
    
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

    elif parsed.subcommand == "analyze":
        try:
            # 1. Run ML Prediction Engine
            engine = RiskPredictionEngine(models_dir=parsed.models_dir)
            prediction = engine.predict(
                repo_path=parsed.repo,
                commit_hash=parsed.commit,
                threshold=parsed.threshold
            )

            # 2. Instantiate selected LLM Provider
            if parsed.provider == "mock":
                provider: LLMProvider = MockLLMProvider()
            elif parsed.provider == "openai":
                provider_kwargs = {}
                if parsed.model:
                    provider_kwargs["model"] = parsed.model
                provider = OpenAIProvider(**provider_kwargs)
            else:
                sys.stderr.write(f"[PatchGuard Error] Unsupported provider: {parsed.provider}\n")
                return 1

            # 3. Execute Qualitative LLM Analysis Pipeline
            analyzer = LLMCodeAnalyzer(provider=provider)
            analysis = analyzer.analyze_repo_commit(
                repo_path=parsed.repo,
                commit_ref=parsed.commit,
                prediction_result=prediction
            )

            # 4. Render Output
            if parsed.format == "json":
                combined_output = {
                    "commit": {
                        "hash": prediction.commit_hash,
                        "full_hash": prediction.full_hash,
                        "repository": prediction.repo_path
                    },
                    "ml_prediction": prediction.to_dict(),
                    "code_change_analysis": analysis.to_dict()
                }
                print(json.dumps(combined_output, indent=2))
            else:
                print(render_analysis_text_card(prediction, analysis))

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
