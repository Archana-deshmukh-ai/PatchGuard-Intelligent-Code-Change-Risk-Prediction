"""
PatchGuard Developer Dashboard — Commit Risk Inspector, Qualitative Analysis & ML Benchmark Viewer (Phase 9.4).
Provides an interactive, local-first Streamlit UI for:
1. Commit Risk Inspector: analyzing code change risk for a single commit using the production ML model.
2. Qualitative LLM Analysis: optional structured code-change explanation and review recommendations.
3. ML Benchmark Viewer: read-only comparative analysis of Phase 8 model benchmark results.
"""

import os
import sys
import json

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import altair as alt
import streamlit as st

from src.engine import RiskPredictionEngine, PredictionResult
from src.analysis.analyzer import LLMCodeAnalyzer
from src.analysis.providers import MockLLMProvider, OpenAIProvider
from src.analysis.schema import LLMAnalysisResult
from src.ui.helpers import (
    prepare_signals_chart_data,
    format_features_dataframe,
    get_risk_level_badge,
    format_error_message,
    format_risk_factor_display,
)
from src.ui.benchmark_helpers import (
    load_benchmark_data,
    format_benchmark_comparison_records,
    prepare_benchmark_chart_records,
    get_split_integrity_summary,
    get_decision_justification,
)
from src.analysis.pr_schema import PRAnalysisInput, PRAnalysisResult, PRCommitPrediction
from src.analysis.pr_analyzer import PRAnalyzer
from src.analysis.diff_extractor import GitDiffExtractor
from src.github.schema import GitHubPullRequest
from src.github.client import GitHubClient
from src.github.auth import GitHubAppConfig
from src.github.adapter import GitHubPRAdapter
from src.github.acquisition import RepositoryAcquisitionManager
from src.ui.pr_helpers import (
    validate_local_pr_inputs,
    parse_github_pr_identifier,
    is_github_app_configured,
    get_pr_risk_summary_stats,
    format_pr_commits_table,
    format_pr_error,
)


class _NoOpLLMAnalyzer:
    """Pass-through dummy analyzer ensuring PRAnalyzer skips automatic LLM diff analysis."""
    def analyze_commit(self, commit_diff, prediction_result=None):
        return None


@st.cache_resource
def get_prediction_engine() -> RiskPredictionEngine:
    """
    Initializes and caches the production RiskPredictionEngine as a singleton.
    Verifies model and scaler artifact SHA-256 integrity upon startup.
    """
    return RiskPredictionEngine()


def render_benchmark_viewer(metric_choice: str = "PR-AUC"):
    """
    Renders the read-only ML Benchmark Viewer tab for Phase 8 experimental results.
    """
    st.title("📊 PatchGuard ML Benchmark Viewer")
    st.subheader("Historical Model Evaluation & Production Decision Audit (Phase 8)")
    st.caption(
        "Read-only comparative analysis of candidate algorithms across chronological validation "
        "and unseen test sets. Zero retraining or automatic model promotion."
    )

    try:
        benchmark_data = load_benchmark_data()
    except Exception as exc:
        st.error(f"### ❌ Failed to Load Benchmark Data\n\n{exc}")
        st.info("💡 **Remediation**: Ensure `reports/experiments/phase8/phase8_model_benchmark.json` exists and contains valid benchmark data.")
        return

    decision = get_decision_justification(benchmark_data)
    integrity = get_split_integrity_summary(benchmark_data)
    records = format_benchmark_comparison_records(benchmark_data)

    # 1. Decision & Policy Banner
    st.info(
        f"🛡️ **Production Model Decision**: `{decision['decision_code']}`\n\n"
        f"**Preserved Model**: `{decision['preserved_model']}` (@ locked threshold `{decision['preserved_threshold']:.2f}`)\n\n"
        f"**Validation Winner**: `{decision['validation_winner']}` (Val PR-AUC: `{decision['winner_val_pr_auc']:.4f}` vs Production `{decision['prod_val_pr_auc']:.4f}`, Margin: `+{decision['improvement_margin']:.4f}`)\n\n"
        f"**Rationale**: {decision['rationale']}"
    )

    st.markdown("---")

    # 2. Dataset Split & Class Prevalence Overview
    st.markdown("### 🧱 Dataset Splits & Temporal Distribution Shift")
    st.caption(
        "Chronological 70/15/15 train/val/test split. Notice the natural defect prevalence decline "
        "over time as the codebase matured, which impacts fixed-threshold precision on unseen commits."
    )
    scol1, scol2, scol3, scol4 = st.columns(4)
    with scol1:
        st.metric(
            label="Total Mined Commits",
            value=f"{integrity['total_rows']:,}",
            help="Total dataset size across all splits.",
        )
    with scol2:
        st.metric(
            label="Train Split (70%)",
            value=f"{integrity['train_rows']:,}",
            delta=f"{integrity['train_pos_rate']*100:.1f}% defect rate",
            delta_color="off",
            help="Chronological training set.",
        )
    with scol3:
        st.metric(
            label="Validation Split (15%)",
            value=f"{integrity['val_rows']:,}",
            delta=f"{integrity['val_pos_rate']*100:.1f}% defect rate",
            delta_color="off",
            help="Chronological validation set used for hyperparameter & threshold tuning.",
        )
    with scol4:
        st.metric(
            label="Unseen Test Split (15%)",
            value=f"{integrity['test_rows']:,}",
            delta=f"{integrity['test_pos_rate']*100:.1f}% defect rate",
            delta_color="off",
            help="Chronological held-out test set evaluated once at frozen thresholds.",
        )

    st.markdown("---")

    # 3. Model Comparison Table
    st.markdown("### 📋 Multi-Model Benchmark Comparison Table")
    st.caption(
        "Direct comparison of all benchmarked algorithms. Validation metrics reflect tuning performance; "
        "Test metrics evaluate generalization on strictly unseen future commits."
    )

    table_rows = []
    for r in records:
        table_rows.append({
            "Model": r["model_name"],
            "Role": r["status_badge"],
            "Operating Threshold": f"{r['threshold']:.2f}",
            "Features": r["features"],
            "Val PR-AUC": f"{r['val_pr_auc']:.4f}" if r["val_pr_auc"] is not None else "N/A",
            "Val ROC-AUC": f"{r['val_roc_auc']:.4f}" if r["val_roc_auc"] is not None else "N/A",
            "Val F1": f"{r['val_f1']:.4f}" if r["val_f1"] is not None else "N/A",
            "Test PR-AUC": f"{r['test_pr_auc']:.4f}" if r["test_pr_auc"] is not None else "N/A",
            "Test ROC-AUC": f"{r['test_roc_auc']:.4f}" if r["test_roc_auc"] is not None else "N/A",
            "Test F1": f"{r['test_f1']:.4f}" if r["test_f1"] is not None else "N/A",
        })

    df_table = pd.DataFrame(table_rows)
    st.dataframe(df_table, hide_index=True)

    st.markdown("---")

    # 4. Comparative Visualizations
    st.markdown(f"### 📈 {metric_choice} Comparison: Validation vs Unseen Test")
    st.caption(
        "Side-by-side comparison across Validation and Test splits clearly illustrating temporal performance shift. "
        "Validation and Test results are shown separately to avoid misleading cross-split averaging."
    )

    metric_key = "pr_auc"
    if "ROC" in metric_choice:
        metric_key = "roc_auc"
    elif "F1" in metric_choice:
        metric_key = "f1"

    chart_data = prepare_benchmark_chart_records(records, metric_base=metric_key)
    if chart_data:
        df_chart = pd.DataFrame(chart_data)
        chart = (
            alt.Chart(df_chart)
            .mark_bar()
            .encode(
                x=alt.X("split:N", title=None, axis=alt.Axis(labels=True)),
                y=alt.Y("value:Q", title=f"{metric_choice} Value", scale=alt.Scale(domain=[0, 1.0])),
                color=alt.Color(
                    "split:N",
                    scale=alt.Scale(domain=["Validation", "Test (Unseen)"], range=["#1f77b4", "#ff7f0e"]),
                    legend=alt.Legend(title="Evaluation Split"),
                ),
                column=alt.Column("model:N", title="Algorithm / Model"),
                tooltip=["model", "split", "value"],
            )
            .properties(width=160, height=280)
        )
        st.altair_chart(chart)

    st.markdown("---")

    # 5. Model Deep Dive & Confusion Matrices
    st.markdown("### 🔍 Model Deep Dive & Confusion Matrices")

    anchor_data = benchmark_data.get("production_logistic_regression_anchor", {})
    if anchor_data:
        with st.expander("🛡️ Production Logistic Regression (Locked Baseline Anchor)", expanded=False):
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.markdown("**Validation Performance (@ threshold 0.35)**")
                vm = anchor_data.get("validation_metrics", {})
                st.text(f"PR-AUC: {vm.get('pr_auc', 'N/A')} | ROC-AUC: {vm.get('roc_auc', 'N/A')} | F1: {vm.get('f1', 'N/A')}")
                st.text(f"Precision: {vm.get('precision', 'N/A')} | Recall: {vm.get('recall', 'N/A')}")
                st.text(f"Confusion Matrix: TP={vm.get('tp')}, FP={vm.get('fp')}, TN={vm.get('tn')}, FN={vm.get('fn')}")
            with mcol2:
                st.markdown("**Test Performance (@ threshold 0.35)**")
                tm = anchor_data.get("test_metrics", {})
                st.text(f"PR-AUC: {tm.get('pr_auc', 'N/A')} | ROC-AUC: {tm.get('roc_auc', 'N/A')} | F1: {tm.get('f1', 'N/A')}")
                st.text(f"Precision: {tm.get('precision', 'N/A')} | Recall: {tm.get('recall', 'N/A')}")
                st.text(f"Confusion Matrix: TP={tm.get('tp')}, FP={tm.get('fp')}, TN={tm.get('tn')}, FN={tm.get('fn')}")

            deg = anchor_data.get("degradation", {})
            st.caption(f"Degradation (Val -> Test): PR-AUC Diff: {deg.get('pr_auc_diff', 'N/A')} | ROC-AUC Diff: {deg.get('roc_auc_diff', 'N/A')} | F1 Diff: {deg.get('f1_diff', 'N/A')}")

    candidates = benchmark_data.get("benchmark_candidates", {})
    for cname, cdata in candidates.items():
        with st.expander(f"Candidate: Phase 8 {cname}", expanded=False):
            ccol1, ccol2 = st.columns(2)
            with ccol1:
                st.markdown(f"**Validation Performance (@ threshold {cdata.get('frozen_validation_threshold')})**")
                cvm = cdata.get("validation_performance", {})
                st.text(f"PR-AUC: {cvm.get('pr_auc', 'N/A')} | ROC-AUC: {cvm.get('roc_auc', 'N/A')} | F1: {cvm.get('f1', 'N/A')}")
                st.text(f"Precision: {cvm.get('precision', 'N/A')} | Recall: {cvm.get('recall', 'N/A')}")
                st.text(f"Confusion Matrix: TP={cvm.get('tp')}, FP={cvm.get('fp')}, TN={cvm.get('tn')}, FN={cvm.get('fn')}")
            with ccol2:
                st.markdown(f"**Test Performance (@ threshold {cdata.get('frozen_validation_threshold')})**")
                ctm = cdata.get("final_test_performance_at_frozen_threshold", {})
                st.text(f"PR-AUC: {ctm.get('pr_auc', 'N/A')} | ROC-AUC: {ctm.get('roc_auc', 'N/A')} | F1: {ctm.get('f1', 'N/A')}")
                st.text(f"Precision: {ctm.get('precision', 'N/A')} | Recall: {ctm.get('recall', 'N/A')}")
                st.text(f"Confusion Matrix: TP={ctm.get('tp')}, FP={ctm.get('fp')}, TN={ctm.get('tn')}, FN={ctm.get('fn')}")

            st.caption(f"Best Hyperparameters: `{json.dumps(cdata.get('best_hyperparameters', {}))}` | Features: `{cdata.get('features_used')}`")


def render_pr_explorer():
    """
    Renders the Pull Request Explorer view (Phase 9.5).
    Evaluates PR constituent commits using production ML defect predictions (@ threshold 0.35)
    and provides optional cumulative PR-level qualitative analysis via Phase 6 LLM pipeline.
    """
    st.title("🔀 PatchGuard Pull Request Explorer")
    st.subheader("Multi-Commit PR Risk Inspection & Cumulative Code Change Analysis")
    st.caption(
        "Evaluates constituent commits using production ML defect predictions (@ threshold 0.35) "
        "and provides optional cumulative PR-level qualitative analysis. Reuses existing Phase 7 engine."
    )

    # 1. Session State Initialization for PR Explorer
    if "pr_analysis_result" not in st.session_state:
        st.session_state.pr_analysis_result = None
    if "github_pr_metadata" not in st.session_state:
        st.session_state.github_pr_metadata = None
    if "pr_analysis_error" not in st.session_state:
        st.session_state.pr_analysis_error = None
    if "pr_llm_analysis" not in st.session_state:
        st.session_state.pr_llm_analysis = None
    if "pr_llm_error" not in st.session_state:
        st.session_state.pr_llm_error = None
    if "last_pr_input_key" not in st.session_state:
        st.session_state.last_pr_input_key = None

    # 2. Sidebar: PR Configuration & Inputs
    with st.sidebar:
        st.header("🔀 PR Configuration")
        pr_mode = st.radio(
            "Input Source",
            options=["Local Git Repository", "GitHub Pull Request"],
            index=0,
            key="pr_input_mode",
            help="Choose between local Git branch/ref comparison and remote GitHub Pull Request retrieval.",
        )

        if pr_mode == "Local Git Repository":
            pr_repo_path = st.text_input(
                "Local Repository Path",
                value=PROJECT_ROOT,
                key="pr_local_repo_path",
                help="Path to an existing local Git repository directory.",
            )
            pr_base_ref = st.text_input(
                "Base Reference",
                value="main",
                key="pr_base_ref",
                help="Base branch or commit reference (e.g., 'main').",
            )
            pr_head_ref = st.text_input(
                "Head Reference",
                value="HEAD",
                key="pr_head_ref",
                help="Head branch or commit reference (e.g., 'HEAD' or feature branch).",
            )
            gh_repo_input = ""
            gh_pr_num_input = 1
            gh_local_override = ""
            st.caption(
                "🔒 **Offline & Local**: Analyzes Git commit differences directly on disk without network access."
            )
        else:
            gh_repo_input = st.text_input(
                "GitHub Repository / PR URL",
                value="owner/repository",
                key="gh_repo_input",
                help="Target repository in 'owner/repository' format or full GitHub PR URL.",
            )
            gh_pr_num_input = st.number_input(
                "PR Number",
                min_value=1,
                value=1,
                step=1,
                key="gh_pr_num_input",
                help="Pull request number (e.g., 42). Ignored if full PR URL is entered.",
            )
            gh_local_override = st.text_input(
                "Local Repo Path Override (Optional)",
                value="",
                key="gh_local_override",
                help="Optional path to existing local clone to avoid cloning from remote GitHub.",
            )
            pr_repo_path = ""
            pr_base_ref = ""
            pr_head_ref = ""

            # Display environment auth status without leaking secrets
            if is_github_app_configured():
                st.success("✅ GitHub App credentials detected in environment.")
            else:
                st.warning(
                    "⚠️ GitHub App credentials not detected. Remote repository acquisition "
                    "requires GITHUB_APP_ID/INSTALLATION_ID/KEY or providing a Local Repo Path Override."
                )

        st.markdown("---")
        analyze_pr_button = st.button(
            "Analyze Pull Request", type="primary", key="analyze_pr_button"
        )
        st.markdown("---")
        st.caption(
            "🔒 **Security Boundary**: Read-only Git inspection. Never executes repository code or builds."
        )

    # 3. Detect Input Invalidation
    current_input_key = (
        pr_mode,
        pr_repo_path.strip(),
        pr_base_ref.strip(),
        pr_head_ref.strip(),
        gh_repo_input.strip(),
        int(gh_pr_num_input) if gh_pr_num_input else 1,
        gh_local_override.strip(),
    )
    if st.session_state.last_pr_input_key is not None and st.session_state.last_pr_input_key != current_input_key:
        st.session_state.pr_analysis_result = None
        st.session_state.github_pr_metadata = None
        st.session_state.pr_analysis_error = None
        st.session_state.pr_llm_analysis = None
        st.session_state.pr_llm_error = None

    # 4. Handle "Analyze Pull Request" Click
    if analyze_pr_button:
        st.session_state.last_pr_input_key = current_input_key
        st.session_state.pr_analysis_error = None
        st.session_state.pr_llm_analysis = None
        st.session_state.pr_llm_error = None

        if pr_mode == "Local Git Repository":
            try:
                validate_local_pr_inputs(pr_repo_path, pr_base_ref, pr_head_ref)
                engine = get_prediction_engine()
                analyzer = PRAnalyzer(predictor=engine, llm_analyzer=_NoOpLLMAnalyzer())
                pr_input = PRAnalysisInput(
                    repo_path=pr_repo_path.strip(),
                    base_ref=pr_base_ref.strip(),
                    head_ref=pr_head_ref.strip(),
                )
                result = analyzer.analyze(pr_input)
                st.session_state.pr_analysis_result = result
                st.session_state.github_pr_metadata = None
            except Exception as exc:
                st.session_state.pr_analysis_result = None
                st.session_state.github_pr_metadata = None
                st.session_state.pr_analysis_error = format_pr_error(exc)
        else:
            try:
                owner, repo, pr_num = parse_github_pr_identifier(gh_repo_input, gh_pr_num_input)
                config = GitHubAppConfig.from_env()
                client = GitHubClient(config=config)
                gh_pr = client.get_pull_request(owner=owner, repo=repo, pull_number=pr_num)
                token = client._get_token()

                acq_mgr = RepositoryAcquisitionManager()
                local_override_clean = gh_local_override.strip() if gh_local_override.strip() else None
                with acq_mgr.acquire(
                    owner=gh_pr.owner,
                    repo=gh_pr.repository,
                    base_sha=gh_pr.base_sha,
                    head_sha=gh_pr.head_sha,
                    token=token,
                    local_repo_override=local_override_clean,
                ) as acquired_path:
                    engine = get_prediction_engine()
                    analyzer = PRAnalyzer(predictor=engine, llm_analyzer=_NoOpLLMAnalyzer())
                    pr_input = GitHubPRAdapter.to_analysis_input(gh_pr, repo_path=acquired_path)
                    result = analyzer.analyze(pr_input)
                    st.session_state.pr_analysis_result = result
                    st.session_state.github_pr_metadata = gh_pr
            except Exception as exc:
                st.session_state.pr_analysis_result = None
                st.session_state.github_pr_metadata = None
                st.session_state.pr_analysis_error = format_pr_error(exc)

    # 5. Render Output
    if st.session_state.pr_analysis_error:
        err_title, err_msg = st.session_state.pr_analysis_error
        st.error(f"### ❌ {err_title}\n\n{err_msg}")
        return

    if st.session_state.pr_analysis_result is not None:
        pr_res: PRAnalysisResult = st.session_state.pr_analysis_result
        gh_meta: Optional[GitHubPullRequest] = st.session_state.github_pr_metadata

        # Section A: PR Identity & Branch Overview
        if gh_meta:
            st.markdown(f"### 📋 GitHub Pull Request #{gh_meta.number}: {gh_meta.title}")
            url_link = f"[{gh_meta.html_url}]({gh_meta.html_url})" if gh_meta.html_url else "N/A"
            st.markdown(
                f"**Repository**: `{gh_meta.owner}/{gh_meta.repository}` | "
                f"**State**: `{gh_meta.state.upper()}` | **Link**: {url_link}\n\n"
                f"**Base Ref**: `{gh_meta.base_ref}` (`{gh_meta.base_sha[:8]}`) → "
                f"**Head Ref**: `{gh_meta.head_ref}` (`{gh_meta.head_sha[:8]}`)"
            )
        else:
            st.markdown(f"### 📋 Local Pull Request: `{pr_res.base_ref}` → `{pr_res.head_ref}`")
            st.markdown(
                f"**Repository**: `{pr_res.repo_path}`\n\n"
                f"**Resolved Base**: `{pr_res.resolved_base_sha[:8]}` | "
                f"**Merge Base**: `{pr_res.resolved_merge_base_sha[:8]}` | "
                f"**Resolved Head**: `{pr_res.resolved_head_sha[:8]}`"
            )

        # Section B: Summary Metric Cards
        stats = get_pr_risk_summary_stats(pr_res.commits)
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Total Commits", f"{stats['total_commits']}")
        with mcol2:
            st.metric("Analyzed Commits", f"{stats['analyzed_commits']}")
        with mcol3:
            st.metric(
                "Flagged Defect-Prone",
                f"{stats['defect_prone_commits']}",
                help="Commits evaluated as defect-prone (P(defect) >= 0.35)",
            )
        with mcol4:
            st.metric(
                "Skipped / Merges",
                f"{stats['merge_commits'] + stats['skipped_commits']}",
                help="Commits skipped from ML risk prediction (merge commits or unanalyzable)",
            )

        # Section C: Scientific Disclaimer & Policy Note
        st.info(f"ℹ️ **Model Governance Note**: {pr_res.analysis_disclaimer}")

        if pr_res.is_empty or not pr_res.commits:
            st.warning("⚠️ No commits or code changes detected between base and head references.")
            return

        # Section D: Constituent Commits Table
        st.markdown("### 📊 Constituent Commits Risk Inventory")
        st.caption(
            "Individual commit risk predictions evaluated by the production Logistic Regression model (@ threshold 0.35)."
        )
        table_rows = format_pr_commits_table(pr_res.commits)
        if table_rows:
            df_display = pd.DataFrame(table_rows)[
                ["Commit", "Author", "Message", "Risk Score", "Classification", "Risk Level", "Status"]
            ]
            st.dataframe(df_display)

        # Section E: Commit Deep Dive Expander
        analyzed_commits = [c for c in pr_res.commits if c.prediction is not None]
        if analyzed_commits:
            with st.expander("🔍 Commit Model Signals & Feature Deep Dive", expanded=False):
                commit_dict = {c.commit_hash: c for c in analyzed_commits}
                selected_hash = st.selectbox(
                    "Select Commit to Inspect",
                    options=list(commit_dict.keys()),
                    format_func=lambda h: f"{h} — {commit_dict[h].commit_message.splitlines()[0][:50]}",
                    key="pr_commit_inspect_select",
                )
                sel_commit = commit_dict[selected_hash]
                sel_pred = sel_commit.prediction

                st.markdown(f"**Full Commit SHA**: `{sel_commit.full_hash}`")
                st.markdown(f"**Author**: `{sel_commit.author}` | **Timestamp**: `{sel_commit.timestamp}`")
                st.markdown(f"**Message**: {sel_commit.commit_message}")

                # Signals Chart
                signals_data = prepare_signals_chart_data(sel_pred.model_signals)
                if signals_data:
                    df_signals = pd.DataFrame(signals_data)
                    st.markdown("##### Quantitative Model Signals (Log-Odds Contributions)")
                    chart = (
                        alt.Chart(df_signals)
                        .mark_bar()
                        .encode(
                            x=alt.X("contribution:Q", title="Log-Odds Contribution"),
                            y=alt.Y("feature:N", sort=None, title="Feature"),
                            color=alt.Color(
                                "direction:N",
                                scale=alt.Scale(
                                    domain=["Increases Risk", "Decreases Risk"],
                                    range=["#d9534f", "#5cb85c"],
                                ),
                                title="Signal Direction",
                            ),
                        )
                        .properties(height=240)
                    )
                    st.altair_chart(chart)

                # Canonical Features
                st.markdown("##### Canonical 10-Feature Data")
                feat_df = format_features_dataframe(sel_pred.features)
                st.dataframe(feat_df)

        # Section F: Optional Cumulative Qualitative LLM Analysis
        st.markdown("---")
        st.markdown("### 🧠 Cumulative Pull Request Code-Change Analysis (Phase 6 LLM)")
        st.caption(
            "Optional qualitative risk analysis of the cumulative diff (BASE → HEAD via merge-base). "
            "Evaluates code changes using the Phase 6 LLM analysis pipeline without modifying ML predictions."
        )

        col_pr_p, col_pr_m = st.columns(2)
        with col_pr_p:
            pr_provider_choice = st.selectbox(
                "LLM Provider",
                options=["mock", "openai"],
                index=0,
                key="pr_llm_provider_select",
                help="Select LLM provider. 'mock' runs offline locally; 'openai' requires OPENAI_API_KEY.",
            )
        with col_pr_m:
            pr_model_name = st.text_input(
                "Model Override",
                value="gpt-4o-mini",
                key="pr_llm_model_input",
                help="Target model identifier for OpenAI API.",
            )

        if pr_provider_choice == "openai":
            st.warning(
                "⚠️ **External Privacy Boundary**: Qualitative analysis transmits the cumulative "
                "code diff to `api.openai.com`. Requires `OPENAI_API_KEY` configured in the environment."
            )

        run_pr_llm_button = st.button(
            "Generate Cumulative Qualitative Analysis",
            key="run_pr_llm_button",
        )

        if run_pr_llm_button:
            st.session_state.pr_llm_error = None
            try:
                # 1. Instantiate selected provider
                if pr_provider_choice == "mock":
                    llm_prov = MockLLMProvider()
                elif pr_provider_choice == "openai":
                    llm_prov = OpenAIProvider(model=pr_model_name.strip() or "gpt-4o-mini")
                else:
                    raise ValueError(f"Unsupported provider: {pr_provider_choice}")

                # 2. Extract cumulative diff and analyze
                analyzer = LLMCodeAnalyzer(provider=llm_prov)
                diff_extractor = GitDiffExtractor(pr_res.repo_path)
                cum_diff = diff_extractor.extract_cumulative_diff(pr_res.base_ref, pr_res.head_ref)

                if cum_diff.is_empty_commit:
                    st.session_state.pr_llm_analysis = None
                    st.info("No code changes detected in the cumulative diff.")
                else:
                    analysis = analyzer.analyze_commit(commit_diff=cum_diff, prediction_result=None)
                    st.session_state.pr_llm_analysis = analysis

            except Exception as exc:
                st.session_state.pr_llm_analysis = None
                st.session_state.pr_llm_error = format_error_message(exc)

        # Render LLM Results or Error
        if st.session_state.pr_llm_error:
            llm_err = st.session_state.pr_llm_error
            st.error(
                f"### ❌ Qualitative Analysis Failed: {llm_err.get('title', 'Error')} (`{llm_err.get('error_type', 'Error')}`)\n\n"
                f"{llm_err.get('message', '')}"
            )
            st.info(f"💡 **Suggested Action**: {llm_err.get('remediation', 'ML defect risk predictions remain intact and unaffected.')}")

        elif st.session_state.pr_llm_analysis is not None:
            llm_res: LLMAnalysisResult = st.session_state.pr_llm_analysis

            st.markdown("#### 📝 Cumulative Code Change Summary")
            st.write(llm_res.summary)

            if llm_res.key_changes:
                st.markdown("#### 🔑 Key Code Changes")
                for ch in llm_res.key_changes:
                    st.markdown(f"- {ch}")

            st.markdown("#### ⚠️ Potential Risk Factors")
            if llm_res.potential_risk_factors:
                for rf in llm_res.potential_risk_factors:
                    rf_d = format_risk_factor_display(rf)
                    st.markdown(f"**{rf_d['priority_badge']} {rf_d['category']}**: {rf_d['description']}")
                    st.caption(f"📍 Location: `{rf_d['location_display']}`")
                    if rf_d["evidence_snippet"]:
                        st.code(rf_d["evidence_snippet"], language="text")
                    if rf_d["evidence_verified"]:
                        st.caption(f"✅ **{rf_d['verification_badge']}**: Grounded in cumulative diff.")
                    else:
                        st.caption(f"⚠️ **{rf_d['verification_badge']}**: Unverified by cumulative diff.")
            else:
                st.info("No qualitative risk factors identified.")

            col_a, col_t = st.columns(2)
            with col_a:
                if llm_res.affected_areas:
                    st.markdown("#### 🗺️ Affected Areas")
                    for a in llm_res.affected_areas:
                        st.markdown(f"- `{a}`")
            with col_t:
                if llm_res.testing_observations:
                    st.markdown("#### 🧪 Testing Observations")
                    for to in llm_res.testing_observations:
                        st.markdown(f"- {to}")

            if llm_res.recommended_review_actions:
                st.markdown("#### 📋 Recommended Review Actions")
                for ra in llm_res.recommended_review_actions:
                    target_str = f" (Target: `{ra.target_file}`)" if ra.target_file else ""
                    st.markdown(f"- {ra.action}{target_str}")

            st.markdown("---")
            st.caption(
                f"ℹ️ **Confidence & Audit Notes**: {llm_res.confidence_notes or 'None reported.'}\n\n"
                f"Model Provider: `{llm_res.model_provider}` | Model: `{llm_res.model_name}`\n\n"
                "**Verification Boundary**: Grounded citations are verified against cumulative diff hunks."
            )

    else:
        st.info(
            "👈 Select a local Git repository or enter a GitHub PR in the sidebar, "
            "then click **Analyze Pull Request** to inspect risks."
        )


def main():
    st.set_page_config(
        page_title="PatchGuard — Developer Dashboard",
        page_icon="🛡️",
        layout="wide",
    )

    # Sidebar Navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        view_mode = st.radio(
            "Dashboard View",
            options=["Commit Risk Inspector", "ML Benchmark Viewer", "Pull Request Explorer"],
            index=0,
            key="dashboard_view_select",
            help="Switch between single commit risk analysis, multi-model benchmark evaluation, and pull request exploration.",
        )
        st.markdown("---")

    if view_mode == "ML Benchmark Viewer":
        with st.sidebar:
            st.header("📊 Benchmark Settings")
            metric_choice = st.selectbox(
                "Metric Comparison Focus",
                options=["PR-AUC", "ROC-AUC", "F1 Score"],
                index=0,
                key="benchmark_metric_select",
                help="Select evaluation metric for side-by-side comparison across models.",
            )
            st.markdown("---")
            st.caption(
                "🔒 **Read-Only Mode**: Displays already-recorded Phase 8 benchmark evaluations. "
                "No models are trained or retrained. Production model is strictly preserved."
            )
        render_benchmark_viewer(metric_choice=metric_choice)
        return

    if view_mode == "Pull Request Explorer":
        render_pr_explorer()
        return

    # --- COMMIT RISK INSPECTOR VIEW ---
    # 1. Header & Identity
    st.title("🛡️ PatchGuard")
    st.subheader("Intelligent Code Change Risk Prediction — Commit Risk Inspector")
    st.caption(
        "Quantitative defect-risk inference powered by production Logistic Regression (@ threshold 0.35) "
        "with optional qualitative code-change analysis."
    )

    # 2. Session State Initialization
    if "last_analysis" not in st.session_state:
        st.session_state.last_analysis = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "last_llm_analysis" not in st.session_state:
        st.session_state.last_llm_analysis = None
    if "last_llm_error" not in st.session_state:
        st.session_state.last_llm_error = None

    # 3. Sidebar: Configuration & Repository Inputs
    with st.sidebar:
        st.header("🎯 Target Selection")
        repo_path_input = st.text_input(
            "Local Repository Path",
            value=PROJECT_ROOT,
            help="Path to an existing local Git repository directory.",
            key="repo_path_input",
        )
        commit_ref_input = st.text_input(
            "Commit Reference / SHA",
            value="HEAD",
            help="Commit SHA, branch, tag, or ref (default: HEAD).",
            key="commit_ref_input",
        )

        st.markdown("---")
        analyze_button = st.button(
            "Analyze Commit", type="primary", key="analyze_button"
        )

        st.markdown("---")
        st.caption(
            "🔒 **Security Boundary**: Analysis is strictly offline and read-only. "
            "No repository code or build scripts are ever executed."
        )

    # Cleaned input paths
    repo_clean = repo_path_input.strip() if repo_path_input else ""
    commit_clean = commit_ref_input.strip() if commit_ref_input else ""

    # Invalidate stale results if user modified target selection in sidebar
    if st.session_state.last_analysis is not None:
        if (
            st.session_state.last_analysis.get("repo_path") != repo_clean
            or st.session_state.last_analysis.get("commit_ref") != commit_clean
        ):
            st.session_state.last_analysis = None
            st.session_state.last_error = None
            st.session_state.last_llm_analysis = None
            st.session_state.last_llm_error = None

    if st.session_state.last_llm_analysis is not None:
        if (
            st.session_state.last_llm_analysis.get("repo_path") != repo_clean
            or st.session_state.last_llm_analysis.get("commit_ref") != commit_clean
        ):
            st.session_state.last_llm_analysis = None
            st.session_state.last_llm_error = None

    # 4. Trigger Analysis on Button Click
    if analyze_button:
        st.session_state.last_analysis = None
        st.session_state.last_error = None
        st.session_state.last_llm_analysis = None
        st.session_state.last_llm_error = None

        if not repo_clean:
            st.session_state.last_error = {
                "title": "Missing Repository Path",
                "message": "Repository path cannot be empty.",
                "remediation": "Enter a valid local Git repository path in the sidebar.",
                "error_type": "ValueError",
            }
        elif not commit_clean:
            st.session_state.last_error = {
                "title": "Missing Commit Reference",
                "message": "Commit reference cannot be empty.",
                "remediation": "Enter a commit SHA, branch, or ref (e.g. 'HEAD') in the sidebar.",
                "error_type": "ValueError",
            }
        else:
            with st.spinner("Extracting commit features and executing model inference..."):
                try:
                    engine = get_prediction_engine()
                    result: PredictionResult = engine.predict(
                        repo_path=repo_clean,
                        commit_hash=commit_clean,
                    )
                    st.session_state.last_analysis = {
                        "result": result,
                        "repo_path": repo_clean,
                        "commit_ref": commit_clean,
                    }
                    st.session_state.last_error = None
                    st.session_state.last_llm_analysis = None
                    st.session_state.last_llm_error = None
                except Exception as exc:
                    st.session_state.last_error = format_error_message(exc)

    # 5. Display Error (if any)
    if st.session_state.last_error:
        err = st.session_state.last_error
        st.error(f"### ❌ {err['title']} (`{err['error_type']}`)\n\n{err['message']}")
        st.info(f"💡 **Suggested Action**: {err['remediation']}")

    # 6. Display Analysis Results (if available)
    elif st.session_state.last_analysis:
        analysis_data = st.session_state.last_analysis
        result: PredictionResult = analysis_data["result"]
        badge = get_risk_level_badge(result.risk_level, result.is_above_threshold)

        st.markdown(
            f"### 📊 Analysis for Commit: `{result.commit_hash}` (`{result.full_hash[:12]}`)"
        )
        st.text(f"Repository: {result.repo_path}")

        # Section A: Primary Risk Summary Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Estimated Risk Score",
                value=f"{result.raw_probability:.4f}",
                delta=f"{result.raw_probability - result.decision_threshold:+.4f} vs threshold",
                delta_color="inverse",
            )
        with col2:
            st.metric(
                label="Production Threshold",
                value=f"{result.decision_threshold:.2f}",
                help="Locked production decision boundary selected in Phase 4/8.",
            )
        with col3:
            st.metric(
                label="Risk Classification",
                value=result.prediction_label,
            )
        with col4:
            st.metric(
                label="Presentation Risk Level",
                value=badge["badge_text"],
                help=badge["disclaimer"],
            )

        st.caption(
            "ℹ️ **Scientific Disclaimer**: Model score is an uncalibrated statistical risk estimate based on "
            "commit-time engineering features. It is not a guarantee that a defect exists."
        )

        st.markdown("---")

        # Section B & C: Model Signals Chart & Canonical Features Side-by-Side
        col_signals, col_features = st.columns([1, 1])

        with col_signals:
            st.markdown("#### 📈 Mathematical Model Signals")
            st.caption(
                "Linear log-odds feature contributions ($C_i = \\beta_i \\cdot z_i$) relative to training mean. "
                "These contributions describe the linear model's statistical behavior, not causal explanations."
            )
            signals_data = prepare_signals_chart_data(result.model_signals)
            if signals_data:
                df_signals = pd.DataFrame(signals_data)
                chart = (
                    alt.Chart(df_signals)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "contribution:Q",
                            title="Log-Odds Contribution (β_i * z_i)",
                        ),
                        y=alt.Y("feature:N", sort=None, title="Feature"),
                        color=alt.Color(
                            "direction:N",
                            scale=alt.Scale(
                                domain=["Positive", "Negative", "Neutral"],
                                range=["#d9534f", "#5cb85c", "#777777"],
                            ),
                            legend=alt.Legend(title="Risk Impact"),
                        ),
                        tooltip=["feature", "contribution", "impact"],
                    )
                    .properties(height=max(240, len(signals_data) * 26))
                )
                st.altair_chart(chart)
            else:
                st.info("No quantitative feature contributions available.")

        with col_features:
            st.markdown("#### 📋 Canonical 10 Commit Features")
            st.caption("Extracted commit-time features in exact production schema order.")
            features_records = format_features_dataframe(result.features)
            df_features = pd.DataFrame(features_records)
            st.dataframe(
                df_features[["feature", "value"]],
                hide_index=True,
                column_config={
                    "feature": st.column_config.TextColumn("Feature Name"),
                    "value": st.column_config.NumberColumn("Extracted Value"),
                },
            )

        st.markdown("---")

        # Section D: Model Provenance & Artifact Integrity
        with st.expander("🛡️ Model Provenance & Verification Hashes", expanded=False):
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.markdown(
                    f"**Model ID**: `{result.model_id}` (`{result.model_version}`)"
                )
                st.markdown(
                    f"**Feature Schema Version**: `{result.feature_schema_version}` (10 canonical features)"
                )
                st.markdown(f"**Decision Threshold**: `{result.decision_threshold:.2f}`")
                st.markdown(
                    f"**Prediction Timestamp (UTC)**: `{result.prediction_timestamp}`"
                )
            with pcol2:
                st.markdown(
                    f"**Model Artifact File**: `logistic_regression_real.pkl`"
                )
                st.markdown(f"**Model SHA-256**: `{result.model_artifact_hash}`")
                st.markdown(
                    f"**Scaler Artifact File**: `real_data_scaler.pkl`"
                )
                st.markdown(
                    f"**Scaler SHA-256**: `{result.scaler_artifact_hash}`"
                )

        st.markdown("---")

        # Section E: Qualitative LLM Analysis (Phase 9.3)
        st.markdown("### 🔍 Qualitative LLM Analysis")
        st.caption(
            "Optional qualitative code-change explanation and review recommendations. "
            "Reuses the ML prediction above without modifying or recalculating risk scores."
        )

        col_prov, col_cfg = st.columns([1, 2])
        with col_prov:
            provider_choice = st.selectbox(
                "LLM Provider",
                options=["mock", "openai"],
                index=0,
                key="llm_provider_select",
                help="Select LLM provider for qualitative analysis (default: mock).",
            )

        with col_cfg:
            if provider_choice == "openai":
                st.text_input(
                    "OpenAI Model",
                    value="gpt-4o-mini",
                    key="llm_model_name_input",
                    help="OpenAI chat completion model name (e.g. gpt-4o-mini).",
                )
                st.caption(
                    "🌐 **Privacy & Security Notice**: The OpenAI provider sends commit diff content "
                    "to the external OpenAI API (`api.openai.com`). Requires `OPENAI_API_KEY` set in the environment. "
                    "No API keys are stored in session state or logs."
                )
            else:
                st.caption(
                    "🔒 **Offline Mock Provider**: Operates strictly locally and offline. "
                    "Generates deterministic structural observations without API keys or network requests."
                )

        run_llm_button = st.button(
            "Generate Qualitative Analysis",
            key="run_llm_button",
            help="Run qualitative LLM code-change analysis for this commit.",
        )

        if run_llm_button:
            st.session_state.last_llm_error = None
            st.session_state.last_llm_analysis = None

            with st.spinner(f"Running qualitative analysis using '{provider_choice}' provider..."):
                try:
                    if provider_choice == "mock":
                        provider = MockLLMProvider()
                    elif provider_choice == "openai":
                        selected_model = (
                            st.session_state.get("llm_model_name_input", "gpt-4o-mini").strip()
                            or "gpt-4o-mini"
                        )
                        provider = OpenAIProvider(model=selected_model)
                    else:
                        raise ValueError(f"Unsupported provider: {provider_choice}")

                    analyzer = LLMCodeAnalyzer(provider=provider)
                    llm_res: LLMAnalysisResult = analyzer.analyze_repo_commit(
                        repo_path=analysis_data["repo_path"],
                        commit_ref=analysis_data["commit_ref"],
                        prediction_result=result,
                    )
                    st.session_state.last_llm_analysis = {
                        "result": llm_res,
                        "repo_path": analysis_data["repo_path"],
                        "commit_ref": analysis_data["commit_ref"],
                        "provider": provider_choice,
                    }
                    st.session_state.last_llm_error = None
                except Exception as exc:
                    st.session_state.last_llm_error = format_error_message(exc)
                    st.session_state.last_llm_analysis = None

        # Display LLM Error (isolated from ML prediction)
        if st.session_state.last_llm_error:
            llm_err = st.session_state.last_llm_error
            st.error(
                f"### ❌ Qualitative Analysis Failed: {llm_err['title']} (`{llm_err['error_type']}`)\n\n"
                f"{llm_err['message']}"
            )
            st.info(f"💡 **Suggested Action**: {llm_err['remediation']}")

        # Display LLM Analysis Results
        elif st.session_state.last_llm_analysis:
            llm_data: LLMAnalysisResult = st.session_state.last_llm_analysis["result"]

            # 1. Summary
            st.markdown("#### 📝 Commit Summary")
            st.markdown(llm_data.summary)

            # 2. Key Changes
            if llm_data.key_changes:
                st.markdown("#### 🔑 Key Code Changes")
                for change in llm_data.key_changes:
                    st.markdown(f"- {change}")

            # 3. Structured Risk Factors
            st.markdown("#### ⚠️ Potential Risk Factors")
            if llm_data.potential_risk_factors:
                for rf in llm_data.potential_risk_factors:
                    rf_disp = format_risk_factor_display(rf)
                    st.markdown(
                        f"**{rf_disp['priority_badge']} {rf_disp['category']}**: "
                        f"{rf_disp['description']}"
                    )
                    st.caption(f"📍 Location: `{rf_disp['location_display']}`")
                    if rf_disp["evidence_snippet"]:
                        st.code(rf_disp["evidence_snippet"], language="text")

                    if rf_disp["evidence_verified"]:
                        st.caption(
                            f"✅ **{rf_disp['verification_badge']}**: Grounded in commit diff."
                        )
                    else:
                        st.caption(
                            f"⚠️ **{rf_disp['verification_badge']}**: Unverified by commit diff."
                        )
            else:
                st.info("No qualitative risk factors identified.")

            # 4. Affected Areas & Testing Observations
            col_areas, col_tests = st.columns(2)
            with col_areas:
                if llm_data.affected_areas:
                    st.markdown("#### 🗺️ Affected Areas")
                    for area in llm_data.affected_areas:
                        st.markdown(f"- `{area}`")
            with col_tests:
                if llm_data.testing_observations:
                    st.markdown("#### 🧪 Testing Observations")
                    for obs in llm_data.testing_observations:
                        st.markdown(f"- {obs}")

            # 5. Recommended Review Actions
            if llm_data.recommended_review_actions:
                st.markdown("#### 📋 Recommended Review Actions")
                for action in llm_data.recommended_review_actions:
                    target = f" (Target: `{action.target_file}`)" if action.target_file else ""
                    st.markdown(f"- {action.action}{target}")

            # 6. Confidence & Grounding Audit Notes
            st.markdown("---")
            st.caption(
                f"ℹ️ **Confidence & Audit Notes**: {llm_data.confidence_notes or 'None reported.'}\n\n"
                f"Model Provider: `{llm_data.model_provider}` | Model: `{llm_data.model_name}`\n\n"
                "**Verification Boundary**: Grounded citations are verified against the commit diff. "
                "Unverified citations represent model assertions not matched by the evidence validator."
            )

    else:
        st.info(
            "👈 Enter a local Git repository path and commit reference in the sidebar, "
            "then click **Analyze Commit** to inspect risk."
        )


if __name__ == "__main__":
    main()
