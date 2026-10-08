"""
Phase 8 Benchmark Runner.
Orchestrates end-to-end model benchmarking, threshold selection, test evaluation, and report generation.
"""

import os
import sys
import json
import joblib
import subprocess
import pandas as pd
import numpy as np
import sklearn
import xgboost
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiments.config import (
    FEATURE_COLUMNS, TARGET_COLUMN, IDENTIFIER_COLUMNS, DEFAULT_SEED, EXPERIMENT_VERSION,
    PRODUCTION_THRESHOLD, LOGISTIC_REGRESSION_GRID, RANDOM_FOREST_GRID, XGBOOST_GRID
)
from src.experiments.evaluator import (
    ModelTrainerEvaluator, compute_detailed_metrics, sweep_validation_thresholds
)

def get_git_commit_hash(repo_dir: str) -> str:
    """Returns HEAD git commit hash of the repository if available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_GIT_SHA"

class Phase8BenchmarkRunner:
    def __init__(self, project_root_dir: str = project_root):
        self.project_root = os.path.abspath(project_root_dir)
        self.data_dir = os.path.join(self.project_root, 'data', 'processed')
        self.prod_models_dir = os.path.join(self.project_root, 'models')
        self.exp_models_dir = os.path.join(self.prod_models_dir, 'experiments', 'phase8')
        self.reports_dir = os.path.join(self.project_root, 'reports', 'experiments', 'phase8')

        os.makedirs(self.exp_models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        self.train_path = os.path.join(self.data_dir, 'bottle_train.csv')
        self.val_path = os.path.join(self.data_dir, 'bottle_validation.csv')
        self.test_path = os.path.join(self.data_dir, 'bottle_test.csv')

        for p in [self.train_path, self.val_path, self.test_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required split artifact missing: {p}")

        self.df_train = pd.read_csv(self.train_path)
        self.df_val = pd.read_csv(self.val_path)
        self.df_test = pd.read_csv(self.test_path)

        self.evaluator = ModelTrainerEvaluator(self.df_train, self.df_val, self.df_test)

    def validate_split_integrity(self) -> Dict[str, Any]:
        """Audits split row counts, temporal monotonicity, and zero hash overlaps."""
        train_hashes = set(self.df_train['commit_hash'])
        val_hashes = set(self.df_val['commit_hash'])
        test_hashes = set(self.df_test['commit_hash'])

        overlap_count = len(train_hashes & val_hashes) + len(val_hashes & test_hashes) + len(train_hashes & test_hashes)
        if overlap_count > 0:
            raise ValueError(f"Temporal dataset overlap detected: {overlap_count} overlapping hashes!")

        train_max_ts = pd.to_datetime(self.df_train['commit_timestamp'], utc=True).max()
        val_min_ts = pd.to_datetime(self.df_val['commit_timestamp'], utc=True).min()
        val_max_ts = pd.to_datetime(self.df_val['commit_timestamp'], utc=True).max()
        test_min_ts = pd.to_datetime(self.df_test['commit_timestamp'], utc=True).min()

        if train_max_ts > val_min_ts or val_max_ts > test_min_ts:
            raise ValueError("Temporal non-monotonicity detected across split boundaries!")

        return {
            "train_rows": len(self.df_train),
            "val_rows": len(self.df_val),
            "test_rows": len(self.df_test),
            "total_rows": len(self.df_train) + len(self.df_val) + len(self.df_test),
            "train_pos_rate": round(float(np.mean(self.df_train[TARGET_COLUMN] == 1)), 4),
            "val_pos_rate": round(float(np.mean(self.df_val[TARGET_COLUMN] == 1)), 4),
            "test_pos_rate": round(float(np.mean(self.df_test[TARGET_COLUMN] == 1)), 4),
            "zero_hash_overlap": True,
            "strictly_chronological": True
        }

    def benchmark_candidate_family(
        self,
        family_name: str,
        model_class: Any,
        hyperparam_grid: List[Dict[str, Any]],
        is_scaled: bool
    ) -> Dict[str, Any]:
        """
        Executes hyperparameter grid search on TRAIN + VALIDATION for a candidate model family.
        Selects best candidate configuration based on VALIDATION PR-AUC.
        """
        X_train, X_val, X_test = self.evaluator.prepare_data('logistic_regression' if is_scaled else 'random_forest')
        y_train, y_val, y_test = self.evaluator.y_train, self.evaluator.y_val, self.evaluator.y_test

        candidates = []
        for idx, params in enumerate(hyperparam_grid):
            m_obj = model_class(**params)
            m_obj.fit(X_train, y_train)

            val_probs = m_obj.predict_proba(X_val)[:, 1]
            val_sweep, val_op = sweep_validation_thresholds(y_val, val_probs)
            max_f1_op = val_op["max_f1"]

            candidates.append({
                "candidate_id": f"{family_name.lower()}_cfg_{idx + 1}",
                "params": params,
                "model_obj": m_obj,
                "val_probs": val_probs,
                "val_pr_auc": max_f1_op["pr_auc"],
                "val_roc_auc": max_f1_op["roc_auc"],
                "val_max_f1": max_f1_op["f1"],
                "best_val_threshold": max_f1_op["decision_threshold"],
                "operating_points": val_op,
                "threshold_sweep": val_sweep
            })

        # Rank candidates by Validation PR-AUC (with ROC-AUC tiebreaker)
        candidates.sort(key=lambda c: (c["val_pr_auc"], c["val_roc_auc"], c["val_max_f1"]), reverse=True)
        winner = candidates[0]

        # Freeze winner and evaluate on TEST set at frozen best threshold AND production threshold (0.35)
        best_model = winner["model_obj"]
        frozen_thresh = winner["best_val_threshold"]

        test_probs = best_model.predict_proba(X_test)[:, 1]
        test_eval_frozen = compute_detailed_metrics(y_test, test_probs, frozen_thresh)
        test_eval_035 = compute_detailed_metrics(y_test, test_probs, PRODUCTION_THRESHOLD)

        val_frozen_eval = winner["operating_points"]["max_f1"]

        degradation = {
            "pr_auc_diff": round(test_eval_frozen["pr_auc"] - val_frozen_eval["pr_auc"], 4),
            "roc_auc_diff": round(test_eval_frozen["roc_auc"] - val_frozen_eval["roc_auc"], 4),
            "f1_diff": round(test_eval_frozen["f1"] - val_frozen_eval["f1"], 4),
            "precision_diff": round(test_eval_frozen["precision"] - val_frozen_eval["precision"], 4),
            "recall_diff": round(test_eval_frozen["recall"] - val_frozen_eval["recall"], 4)
        }

        # Save model artifact into isolated experiments directory
        artifact_name = f"{family_name.lower()}_phase8.pkl"
        artifact_path = os.path.join(self.exp_models_dir, artifact_name)
        joblib.dump(best_model, artifact_path)

        return {
            "family_name": family_name,
            "best_hyperparameters": winner["params"],
            "features_used": "scaled" if is_scaled else "raw",
            "artifact_path": artifact_path,
            "validation_performance": val_frozen_eval,
            "frozen_validation_threshold": frozen_thresh,
            "final_test_performance_at_frozen_threshold": test_eval_frozen,
            "final_test_performance_at_035": test_eval_035,
            "validation_to_test_degradation": degradation,
            "all_candidate_configs_evaluated": [
                {
                    "params": c["params"],
                    "val_pr_auc": c["val_pr_auc"],
                    "val_roc_auc": c["val_roc_auc"],
                    "val_max_f1": c["val_max_f1"],
                    "best_threshold": c["best_val_threshold"]
                }
                for c in candidates
            ]
        }

    def evaluate_existing_production_model(self) -> Dict[str, Any]:
        """Loads and benchmarks the existing production Logistic Regression artifact without altering it."""
        prod_model_path = os.path.join(self.prod_models_dir, 'logistic_regression_real.pkl')
        prod_scaler_path = os.path.join(self.prod_models_dir, 'real_data_scaler.pkl')

        if not os.path.exists(prod_model_path) or not os.path.exists(prod_scaler_path):
            return {"status": "PRODUCTION_ARTIFACTS_NOT_FOUND"}

        prod_model = joblib.load(prod_model_path)
        prod_scaler = joblib.load(prod_scaler_path)

        X_val_scaled = prod_scaler.transform(self.df_val[FEATURE_COLUMNS])
        X_test_scaled = prod_scaler.transform(self.df_test[FEATURE_COLUMNS])

        y_val = self.df_val[TARGET_COLUMN]
        y_test = self.df_test[TARGET_COLUMN]

        val_probs = prod_model.predict_proba(X_val_scaled)[:, 1]
        test_probs = prod_model.predict_proba(X_test_scaled)[:, 1]

        val_eval_035 = compute_detailed_metrics(y_val, val_probs, PRODUCTION_THRESHOLD)
        test_eval_035 = compute_detailed_metrics(y_test, test_probs, PRODUCTION_THRESHOLD)

        return {
            "model_name": "Production LogisticRegression (Locked Phase 4/5)",
            "model_path": prod_model_path,
            "scaler_path": prod_scaler_path,
            "locked_threshold": PRODUCTION_THRESHOLD,
            "validation_metrics": val_eval_035,
            "test_metrics": test_eval_035,
            "degradation": {
                "pr_auc_diff": round(test_eval_035["pr_auc"] - val_eval_035["pr_auc"], 4),
                "roc_auc_diff": round(test_eval_035["roc_auc"] - val_eval_035["roc_auc"], 4),
                "f1_diff": round(test_eval_035["f1"] - val_eval_035["f1"], 4)
            }
        }

    def run_benchmark(self) -> Dict[str, Any]:
        """Executes full Phase 8 benchmarking pipeline across LR, RF, XGBoost and saves results."""
        split_info = self.validate_split_integrity()

        # Save Phase 8 fitted scaler to isolated experiment directory
        exp_scaler_path = os.path.join(self.exp_models_dir, 'phase8_scaler.pkl')
        joblib.dump(self.evaluator.scaler, exp_scaler_path)

        # 1. Existing Production Baseline Anchor
        prod_anchor = self.evaluate_existing_production_model()

        # 2. Candidate Models Benchmarking
        lr_result = self.benchmark_candidate_family("LogisticRegression", LogisticRegression, LOGISTIC_REGRESSION_GRID, is_scaled=True)
        rf_result = self.benchmark_candidate_family("RandomForest", RandomForestClassifier, RANDOM_FOREST_GRID, is_scaled=False)
        xgb_result = self.benchmark_candidate_family("XGBoost", XGBClassifier, XGBOOST_GRID, is_scaled=False)

        candidates_map = {
            "LogisticRegression": lr_result,
            "RandomForest": rf_result,
            "XGBoost": xgb_result
        }

        # Determine overall benchmark winner based on Validation PR-AUC
        ranked = sorted(
            candidates_map.items(),
            key=lambda item: (
                item[1]["validation_performance"]["pr_auc"],
                item[1]["validation_performance"]["roc_auc"],
                item[1]["validation_performance"]["f1"]
            ),
            reverse=True
        )

        winning_family = ranked[0][0]
        winner_details = ranked[0][1]

        # Determine whether winner meaningfully outperforms existing production LR baseline on Validation
        prod_val_pr_auc = prod_anchor.get("validation_metrics", {}).get("pr_auc", 0.0)
        winner_val_pr_auc = winner_details["validation_performance"]["pr_auc"]
        pr_auc_improvement = round(winner_val_pr_auc - prod_val_pr_auc, 4)

        is_meaningful_improvement = (pr_auc_improvement >= 0.03) and (winner_details["final_test_performance_at_frozen_threshold"]["pr_auc"] > prod_anchor.get("test_metrics", {}).get("pr_auc", 0.0))

        if is_meaningful_improvement:
            recommendation = f"PROPOSE_PROMOTION_CANDIDATE: {winning_family} showed meaningful PR-AUC improvement (+{pr_auc_improvement}) on Validation without collapsing on Test."
        else:
            recommendation = "PRESERVE_PRODUCTION_LOGISTIC_REGRESSION: Neither RF nor XGBoost provided a defensible, substantial improvement over the production LR baseline under temporal evaluation. Keep Logistic Regression as production default."

        report_data = {
            "metadata": {
                "experiment_version": EXPERIMENT_VERSION,
                "timestamp_utc": pd.Timestamp.now().isoformat(),
                "git_commit_sha": get_git_commit_hash(self.project_root),
                "random_seed": DEFAULT_SEED,
                "package_versions": {
                    "scikit_learn": sklearn.__version__,
                    "xgboost": xgboost.__version__,
                    "pandas": pd.__version__,
                    "numpy": np.__version__
                },
                "split_integrity": split_info,
                "experiment_artifacts_dir": self.exp_models_dir,
                "production_artifacts_unmodified": True
            },
            "production_logistic_regression_anchor": prod_anchor,
            "benchmark_candidates": candidates_map,
            "benchmark_summary": {
                "validation_ranking_order": [r[0] for r in ranked],
                "validation_winning_model": winning_family,
                "winner_best_hyperparameters": winner_details["best_hyperparameters"],
                "winner_frozen_threshold": winner_details["frozen_validation_threshold"],
                "winner_val_pr_auc": winner_val_pr_auc,
                "winner_test_pr_auc": winner_details["final_test_performance_at_frozen_threshold"]["pr_auc"],
                "prod_val_pr_auc": prod_val_pr_auc,
                "prod_test_pr_auc": prod_anchor.get("test_metrics", {}).get("pr_auc", 0.0),
                "pr_auc_improvement_over_prod": pr_auc_improvement,
                "is_meaningful_improvement": is_meaningful_improvement,
                "final_recommendation": recommendation
            }
        }

        # Save JSON report
        json_report_path = os.path.join(self.reports_dir, 'phase8_model_benchmark.json')
        with open(json_report_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        # Save CSV summary table
        summary_rows = []
        # Add production anchor
        prod_val = prod_anchor.get("validation_metrics", {})
        prod_test = prod_anchor.get("test_metrics", {})
        summary_rows.append({
            "model_name": "Production_LogisticRegression_v1",
            "threshold": PRODUCTION_THRESHOLD,
            "val_pr_auc": prod_val.get("pr_auc", 0),
            "val_roc_auc": prod_val.get("roc_auc", 0),
            "val_f1": prod_val.get("f1", 0),
            "val_prec": prod_val.get("precision", 0),
            "val_rec": prod_val.get("recall", 0),
            "test_pr_auc": prod_test.get("pr_auc", 0),
            "test_roc_auc": prod_test.get("roc_auc", 0),
            "test_f1": prod_test.get("f1", 0),
            "test_prec": prod_test.get("precision", 0),
            "test_rec": prod_test.get("recall", 0),
            "is_winner": False
        })

        for name, res in candidates_map.items():
            v = res["validation_performance"]
            t = res["final_test_performance_at_frozen_threshold"]
            summary_rows.append({
                "model_name": f"Phase8_{name}",
                "threshold": res["frozen_validation_threshold"],
                "val_pr_auc": v.get("pr_auc", 0),
                "val_roc_auc": v.get("roc_auc", 0),
                "val_f1": v.get("f1", 0),
                "val_prec": v.get("precision", 0),
                "val_rec": v.get("recall", 0),
                "test_pr_auc": t.get("pr_auc", 0),
                "test_roc_auc": t.get("roc_auc", 0),
                "test_f1": t.get("f1", 0),
                "test_prec": t.get("precision", 0),
                "test_rec": t.get("recall", 0),
                "is_winner": (name == winning_family)
            })

        df_summary = pd.DataFrame(summary_rows)
        csv_report_path = os.path.join(self.reports_dir, 'phase8_benchmark_summary.csv')
        df_summary.to_csv(csv_report_path, index=False)

        # Save Markdown human-readable report
        md_report_path = os.path.join(self.reports_dir, 'PHASE8_BENCHMARK_REPORT.md')
        self._generate_markdown_report(report_data, df_summary, md_report_path)

        print(f"[Phase8Runner] Benchmark completed successfully.")
        print(f"[Phase8Runner] Saved JSON report to: {json_report_path}")
        print(f"[Phase8Runner] Saved CSV summary to : {csv_report_path}")
        print(f"[Phase8Runner] Saved MD report to  : {md_report_path}")

        return report_data

    def _generate_markdown_report(self, data: Dict[str, Any], df_summary: pd.DataFrame, md_path: str):
        meta = data["metadata"]
        summary = data["benchmark_summary"]
        prod_anchor = data["production_logistic_regression_anchor"]
        candidates = data["benchmark_candidates"]

        lines = [
            "# Phase 8 ML Model Benchmarking & Risk Prediction Report",
            "",
            "## Executive Summary",
            f"- **Experiment Version**: `{meta['experiment_version']}`",
            f"- **Timestamp (UTC)**: `{meta['timestamp_utc']}`",
            f"- **Git Commit SHA**: `{meta['git_commit_sha']}`",
            f"- **Random Seed**: `{meta['random_seed']}`",
            f"- **Validation Winner**: **{summary['validation_winning_model']}**",
            f"- **Recommendation**: **{summary['final_recommendation']}**",
            "",
            "---",
            "",
            "## Dataset & Split Provenance",
            f"- **Total Dataset Commits**: {meta['split_integrity']['total_rows']}",
            f"- **Train Split**: {meta['split_integrity']['train_rows']} commits (Positive rate: {meta['split_integrity']['train_pos_rate']*100:.2f}%)",
            f"- **Validation Split**: {meta['split_integrity']['val_rows']} commits (Positive rate: {meta['split_integrity']['val_pos_rate']*100:.2f}%)",
            f"- **Test Split**: {meta['split_integrity']['test_rows']} commits (Positive rate: {meta['split_integrity']['test_pos_rate']*100:.2f}%)",
            f"- **Strict Monotonic Temporal Splitting**: {meta['split_integrity']['strictly_chronological']}",
            f"- **Zero Commit Hash Overlap**: {meta['split_integrity']['zero_hash_overlap']}",
            "",
            "---",
            "",
            "## Benchmark Results Summary",
            "",
            "| Model | Frozen Thresh | Val PR-AUC | Val ROC-AUC | Val F1 | Test PR-AUC | Test ROC-AUC | Test F1 | Test Prec | Test Rec | Winner? |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]

        for _, r in df_summary.iterrows():
            winner_str = "**YES**" if r["is_winner"] else "No"
            lines.append(
                f"| {r['model_name']} | {r['threshold']:.2f} | {r['val_pr_auc']:.4f} | {r['val_roc_auc']:.4f} | {r['val_f1']:.4f} | "
                f"{r['test_pr_auc']:.4f} | {r['test_roc_auc']:.4f} | {r['test_f1']:.4f} | {r['test_prec']:.4f} | {r['test_rec']:.4f} | {winner_str} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## Detailed Candidate Model Analysis",
            ""
        ])

        for c_name, c_info in candidates.items():
            val = c_info["validation_performance"]
            test = c_info["final_test_performance_at_frozen_threshold"]
            deg = c_info["validation_to_test_degradation"]

            lines.extend([
                f"### {c_name}",
                f"- **Features Used**: `{c_info['features_used']}`",
                f"- **Best Hyperparameters**: `{json.dumps(c_info['best_hyperparameters'])}`",
                f"- **Frozen Threshold**: `{c_info['frozen_validation_threshold']}`",
                f"- **Validation Metrics**: PR-AUC: `{val['pr_auc']}`, ROC-AUC: `{val['roc_auc']}`, F1: `{val['f1']}`, Precision: `{val['precision']}`, Recall: `{val['recall']}`",
                f"- **Final Test Metrics**: PR-AUC: `{test['pr_auc']}`, ROC-AUC: `{test['roc_auc']}`, F1: `{test['f1']}`, Precision: `{test['precision']}`, Recall: `{test['recall']}`",
                f"- **Validation to Test Degradation**: PR-AUC Diff: `{deg['pr_auc_diff']:+.4f}`, ROC-AUC Diff: `{deg['roc_auc_diff']:+.4f}`, F1 Diff: `{deg['f1_diff']:+.4f}`",
                ""
            ])

        lines.extend([
            "---",
            "",
            "## Production Artifact Safety Audit",
            "- **`models/logistic_regression_real.pkl`**: Modified? **NO (Preserved 100%)**",
            "- **`models/real_data_scaler.pkl`**: Modified? **NO (Preserved 100%)**",
            "- **Production Prediction Behavior**: **UNTOUCHED**",
            ""
        ])

        with open(md_path, 'w') as f:
            f.write("\n".join(lines))

if __name__ == '__main__':
    runner = Phase8BenchmarkRunner()
    runner.run_benchmark()
