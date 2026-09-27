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
    LLMProvider,
    PRAnalyzer,
    PRAnalysisInput,
    PRAnalysisResult
)
from src.github import (
    GitHubAppConfig,
    GitHubAppAuthenticator,
    GitHubClient,
    GitHubPRAdapter,
    GitHubPullRequest,
    GitHubIntegrationError,
    GitHubPRValidationError,
    RepositoryAcquisitionManager
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


def render_pr_analysis_text_card(res: PRAnalysisResult) -> str:
    """Renders human-readable text report card for Pull Request analysis."""
    lines = []
    lines.append("==========================================================================")
    lines.append("                 PATCHGUARD LOCAL PULL REQUEST ANALYSIS                   ")
    lines.append("==========================================================================")
    lines.append(f"  Repository Path        : {res.repo_path}")
    lines.append(f"  Base Reference         : {res.base_ref} ({res.resolved_base_sha[:12]})")
    lines.append(f"  Head Reference         : {res.head_ref} ({res.resolved_head_sha[:12]})")
    lines.append(f"  Merge Base SHA         : {res.resolved_merge_base_sha[:12]}")
    lines.append(f"  PR Commits Included    : {res.commit_count}")
    lines.append("--------------------------------------------------------------------------")
    lines.append("--- PER-COMMIT HISTORICAL ML DEFECT RISK PREDICTIONS ---")
    if res.commits:
        for c in res.commits:
            msg_first_line = c.commit_message.splitlines()[0] if c.commit_message else ''
            lines.append(f"  Commit {c.commit_hash} ({c.full_hash[:12]}): {msg_first_line}")
            if c.prediction:
                p = c.prediction
                lines.append(f"    - Estimated Risk Score : {p.raw_probability:.4f} (Threshold: {p.decision_threshold:.2f})")
                lines.append(f"    - Above Threshold      : {p.is_above_threshold} | Risk Level: {p.risk_level}")
            else:
                lines.append(f"    - ML Prediction Skipped: {c.error or 'N/A'}")
            lines.append("")
    else:
        lines.append("  No commits introduced in this pull request comparison.")
    lines.append("--------------------------------------------------------------------------")
    lines.append("--- CUMULATIVE PULL REQUEST CODE CHANGE ANALYSIS (Phase 6 LLM) ---")
    if res.is_empty or res.cumulative_analysis is None:
        lines.append("  No code changes detected between base and head references.")
    else:
        analysis = res.cumulative_analysis
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
    lines.append(f"DISCLAIMER: {res.analysis_disclaimer}")
    lines.append("==========================================================================\n")
    return "\n".join(lines)


def render_github_pr_analysis_text_card(gh_pr: GitHubPullRequest, res: PRAnalysisResult) -> str:
    """Renders human-readable text report card for GitHub Pull Request risk analysis."""
    lines = []
    lines.append("==========================================================================")
    lines.append("              PATCHGUARD GITHUB PULL REQUEST RISK ANALYSIS                ")
    lines.append("==========================================================================")
    lines.append(f"  Target Repository      : {gh_pr.owner}/{gh_pr.repository}")
    lines.append(f"  Pull Request Number    : #{gh_pr.number}")
    lines.append(f"  Title                  : {gh_pr.title}")
    lines.append(f"  PR State               : {gh_pr.state}")
    lines.append(f"  HTML URL               : {gh_pr.html_url or 'N/A'}")
    lines.append("--------------------------------------------------------------------------")
    lines.append("--- RETRIEVED GITHUB COMMIT REFERENCES ---")
    lines.append(f"  Base Branch / Ref      : {gh_pr.base_ref} ({gh_pr.base_sha[:12]})")
    lines.append(f"  Head Branch / Ref      : {gh_pr.head_ref} ({gh_pr.head_sha[:12]})")
    lines.append("--------------------------------------------------------------------------")
    lines.append("--- PER-COMMIT HISTORICAL ML DEFECT RISK PREDICTIONS ---")
    if res.commits:
        for c in res.commits:
            msg_first_line = c.commit_message.splitlines()[0] if c.commit_message else ''
            lines.append(f"  Commit {c.commit_hash} ({c.full_hash[:12]}): {msg_first_line}")
            if c.prediction:
                p = c.prediction
                lines.append(f"    - Estimated Risk Score : {p.raw_probability:.4f} (Threshold: {p.decision_threshold:.2f})")
                lines.append(f"    - Above Threshold      : {p.is_above_threshold} | Risk Level: {p.risk_level}")
            else:
                lines.append(f"    - ML Prediction Skipped: {c.error or 'N/A'}")
            lines.append("")
    else:
        lines.append("  No commits introduced in this pull request comparison.")
    lines.append("--------------------------------------------------------------------------")
    lines.append("--- CUMULATIVE PULL REQUEST CODE CHANGE ANALYSIS (Phase 6 LLM) ---")
    if res.is_empty or res.cumulative_analysis is None:
        lines.append("  No code changes detected between base and head references.")
    else:
        analysis = res.cumulative_analysis
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
    lines.append(f"DISCLAIMER: {res.analysis_disclaimer}")
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

    # Subcommand: pr-analyze
    pr_parser = subparsers.add_parser("pr-analyze", help="Run local Pull Request risk analysis (per-commit ML + cumulative LLM)")
    pr_parser.add_argument("--repo", "-r", required=True, help="Path to local Git repository")
    pr_parser.add_argument("--base", "-b", required=True, help="Base ref or branch (e.g. main)")
    pr_parser.add_argument("--head", required=True, help="Head ref or branch (e.g. feature/test-change)")
    pr_parser.add_argument("--provider", "-p", choices=["mock", "openai"], default="mock", help="LLM provider to use (default: mock)")
    pr_parser.add_argument("--model", "-m", default=None, help="Optional model name override for provider")
    pr_parser.add_argument("--threshold", "-t", type=float, default=None, help="ML decision threshold override (default: 0.35)")
    pr_parser.add_argument("--format", "-f", choices=["text", "json"], default="text", help="Output format (default: text)")
    pr_parser.add_argument("--models-dir", default=None, help="Directory containing ML model artifacts")

    # Subcommand: github-pr-analyze
    gh_parser = subparsers.add_parser("github-pr-analyze", help="Retrieve GitHub PR, acquire repository workspace, and run PR risk analysis")
    gh_parser.add_argument("--repo", "-r", required=True, help="GitHub repository in 'owner/repository' format (e.g. bottlepy/bottle)")
    gh_parser.add_argument("--pr", "-p", type=int, required=True, help="Pull Request number (e.g. 42)")
    gh_parser.add_argument("--local-repo", default=None, help="Optional existing local repository path override")
    gh_parser.add_argument("--provider", choices=["mock", "openai"], default="mock", help="LLM provider to use (default: mock)")
    gh_parser.add_argument("--model", "-m", default=None, help="Optional model name override for provider")
    gh_parser.add_argument("--threshold", "-t", type=float, default=None, help="ML decision threshold override (default: 0.35)")
    gh_parser.add_argument("--format", "-f", choices=["text", "json"], default="text", help="Output format (default: text)")
    gh_parser.add_argument("--models-dir", default=None, help="Directory containing ML model artifacts")
    
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

    elif parsed.subcommand == "pr-analyze":
        try:
            # 1. Instantiate RiskPredictionEngine
            engine = RiskPredictionEngine(models_dir=parsed.models_dir)

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

            # 3. Instantiate LLMCodeAnalyzer and PRAnalyzer
            llm_analyzer = LLMCodeAnalyzer(provider=provider)
            pr_analyzer = PRAnalyzer(predictor=engine, llm_analyzer=llm_analyzer)

            # 4. Execute PR Analysis
            pr_input = PRAnalysisInput(
                repo_path=parsed.repo,
                base_ref=parsed.base,
                head_ref=parsed.head
            )
            result = pr_analyzer.analyze(pr_input, threshold=parsed.threshold)

            # 5. Render Output
            if parsed.format == "json":
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print(render_pr_analysis_text_card(result))

            return 0

        except PatchGuardError as pge:
            sys.stderr.write(f"[PatchGuard Error] {pge.__class__.__name__}: {pge}\n")
            return 1
        except Exception as e:
            sys.stderr.write(f"[Unexpected Error] {e.__class__.__name__}: {e}\n")
            return 2

    elif parsed.subcommand == "github-pr-analyze":
        try:
            # 1. Parse repository owner and name
            repo_arg = parsed.repo.strip()
            if "/" not in repo_arg:
                raise GitHubPRValidationError("Repository argument must be in 'owner/repository' format (e.g. octocat/Hello-World).")
            parts = repo_arg.split('/')
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise GitHubPRValidationError("Repository argument must be in 'owner/repository' format.")
            owner, repo = parts[0].strip(), parts[1].strip()

            # 2. Retrieve environment configuration and client
            config = GitHubAppConfig.from_env()
            client = GitHubClient(config=config)

            # 3. Fetch GitHub Pull Request metadata
            gh_pr = client.get_pull_request(owner=owner, repo=repo, pull_number=parsed.pr)

            # 4. Acquire access token for Git transport
            token = client._get_token()

            # 5. Acquire repository workspace and execute PR analysis
            acq_mgr = RepositoryAcquisitionManager()
            with acq_mgr.acquire(
                owner=gh_pr.owner,
                repo=gh_pr.repository,
                base_sha=gh_pr.base_sha,
                head_sha=gh_pr.head_sha,
                token=token,
                local_repo_override=parsed.local_repo
            ) as acquired_repo_path:
                # 6. Adapt to PRAnalysisInput
                pr_input = PRAnalysisInput(
                    repo_path=acquired_repo_path,
                    base_ref=gh_pr.base_sha,
                    head_ref=gh_pr.head_sha
                )

                # 7. Instantiate RiskPredictionEngine and LLM Provider
                engine = RiskPredictionEngine(models_dir=parsed.models_dir)
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

                llm_analyzer = LLMCodeAnalyzer(provider=provider)
                pr_analyzer = PRAnalyzer(predictor=engine, llm_analyzer=llm_analyzer)

                # 8. Run PR analysis
                result = pr_analyzer.analyze(pr_input, threshold=parsed.threshold)

                # 9. Render Output
                if parsed.format == "json":
                    out_dict = {
                        "github_pr": gh_pr.to_dict(),
                        "pr_analysis": result.to_dict()
                    }
                    print(json.dumps(out_dict, indent=2))
                else:
                    print(render_github_pr_analysis_text_card(gh_pr, result))

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
