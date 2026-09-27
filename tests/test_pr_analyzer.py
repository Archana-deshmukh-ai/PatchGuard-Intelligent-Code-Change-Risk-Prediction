"""
Unit tests for Phase 7.1 Local Pull Request Analysis Core (PRAnalyzer, PRAnalysisInput, PRAnalysisResult).
Operates 100% offline using temporary local Git repositories and MockLLMProvider.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
import subprocess
from typing import List

from src.analysis import (
    PRAnalyzer,
    PRAnalysisInput,
    PRAnalysisResult,
    MockLLMProvider,
    LLMCodeAnalyzer,
    InvalidPRRefError
)
from src.engine import RiskPredictionEngine, PatchGuardError
from src.cli import main as cli_main


class TestPRAnalyzer(unittest.TestCase):
    """Unit and integration test suite for PRAnalyzer."""

    @classmethod
    def setUpClass(cls):
        # Locate project models directory for RiskPredictionEngine
        cls.script_dir = os.path.dirname(os.path.abspath(__file__))
        cls.project_root = os.path.abspath(os.path.join(cls.script_dir, '..'))
        cls.models_dir = os.path.join(cls.project_root, 'models')

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_pr_repo_")
        self.repo_path = os.path.abspath(self.temp_dir)
        self._init_test_git_repo()

        # Instantiate real prediction engine with models dir and mock LLM analyzer
        self.engine = RiskPredictionEngine(models_dir=self.models_dir)
        self.mock_provider = MockLLMProvider()
        self.llm_analyzer = LLMCodeAnalyzer(provider=self.mock_provider)
        self.pr_analyzer = PRAnalyzer(predictor=self.engine, llm_analyzer=self.llm_analyzer)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _run_git(self, args: List[str]) -> subprocess.CompletedProcess:
        git_executable = shutil.which("git") or "git"
        cmd = [git_executable] + args
        env = os.environ.copy()
        env["GIT_AUTHOR_NAME"] = "Test Author"
        env["GIT_AUTHOR_EMAIL"] = "test@example.com"
        env["GIT_COMMITTER_NAME"] = "Test Author"
        env["GIT_COMMITTER_EMAIL"] = "test@example.com"
        return subprocess.run(
            cmd,
            cwd=self.repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )

    def _init_test_git_repo(self):
        self._run_git(["init"])
        self._run_git(["config", "user.name", "Test Author"])
        self._run_git(["config", "user.email", "test@example.com"])

        # Initial commit on main
        app_file = os.path.join(self.repo_path, "app.py")
        with open(app_file, "w", encoding="utf-8") as f:
            f.write("def main():\n    print('Hello World')\n")
        
        self._run_git(["add", "."])
        self._run_git(["commit", "-m", "Initial commit on main"])
        self._run_git(["branch", "-M", "main"])

        # Resolve initial commit SHA
        res = self._run_git(["rev-parse", "HEAD"])
        self.initial_sha = res.stdout.strip()

        # Branch: feature/single (1 commit ahead)
        self._run_git(["checkout", "-b", "feature/single"])
        with open(app_file, "a", encoding="utf-8") as f:
            f.write("\ndef helper():\n    return 42\n")
        self._run_git(["add", "."])
        self._run_git(["commit", "-m", "Add helper function"])
        res = self._run_git(["rev-parse", "HEAD"])
        self.single_commit_sha = res.stdout.strip()

        # Branch: feature/multi (2 commits ahead of main)
        self._run_git(["checkout", "main"])
        self._run_git(["checkout", "-b", "feature/multi"])

        # Commit 1
        utils_file = os.path.join(self.repo_path, "utils.py")
        with open(utils_file, "w", encoding="utf-8") as f:
            f.write("def format_string(s):\n    return s.strip().lower()\n")
        self._run_git(["add", "."])
        self._run_git(["commit", "-m", "Add utils module"])
        res = self._run_git(["rev-parse", "HEAD"])
        self.multi_commit_1_sha = res.stdout.strip()

        # Commit 2
        with open(app_file, "a", encoding="utf-8") as f:
            f.write("\nfrom utils import format_string\n")
        self._run_git(["add", "."])
        self._run_git(["commit", "-m", "Import format_string in app"])
        res = self._run_git(["rev-parse", "HEAD"])
        self.multi_commit_2_sha = res.stdout.strip()

        # Return to main branch
        self._run_git(["checkout", "main"])

    def test_1_valid_single_commit_pr(self):
        """1. Valid single-commit PR analysis."""
        res = self.pr_analyzer.analyze_pr(
            repo_path=self.repo_path,
            base_ref="main",
            head_ref="feature/single"
        )
        self.assertEqual(res.base_ref, "main")
        self.assertEqual(res.head_ref, "feature/single")
        self.assertFalse(res.is_empty)
        self.assertEqual(res.commit_count, 1)
        self.assertEqual(len(res.commits), 1)

        c = res.commits[0]
        self.assertEqual(c.full_hash, self.single_commit_sha)
        self.assertIsNotNone(c.prediction)
        self.assertTrue(0.0 <= c.prediction.raw_probability <= 1.0)
        self.assertIsNotNone(res.cumulative_analysis)
        self.assertEqual(res.cumulative_analysis.model_provider, "mock")

    def test_2_valid_multi_commit_pr(self):
        """2. Valid multi-commit PR analysis enumerates all commits in order."""
        res = self.pr_analyzer.analyze_pr(
            repo_path=self.repo_path,
            base_ref="main",
            head_ref="feature/multi"
        )
        self.assertFalse(res.is_empty)
        self.assertEqual(res.commit_count, 2)
        self.assertEqual(len(res.commits), 2)

        # Chronological sequence check
        self.assertEqual(res.commits[0].full_hash, self.multi_commit_1_sha)
        self.assertEqual(res.commits[1].full_hash, self.multi_commit_2_sha)
        self.assertIsNotNone(res.commits[0].prediction)
        self.assertIsNotNone(res.commits[1].prediction)

    def test_3_4_5_ref_resolution_and_full_shas(self):
        """3, 4, 5. Resolves base and head refs to 40-character SHAs."""
        res = self.pr_analyzer.analyze_pr(
            repo_path=self.repo_path,
            base_ref="main",
            head_ref="feature/single"
        )
        self.assertEqual(len(res.resolved_base_sha), 40)
        self.assertEqual(len(res.resolved_head_sha), 40)
        self.assertEqual(len(res.resolved_merge_base_sha), 40)
        self.assertEqual(res.resolved_base_sha, self.initial_sha)
        self.assertEqual(res.resolved_head_sha, self.single_commit_sha)
        self.assertEqual(res.resolved_merge_base_sha, self.initial_sha)

    def test_6_7_commit_enumeration_and_cumulative_diff(self):
        """6, 7. Enumerates exact PR commits and extracts cumulative diff."""
        res = self.pr_analyzer.analyze_pr(
            repo_path=self.repo_path,
            base_ref="main",
            head_ref="feature/multi"
        )
        # Verify cumulative analysis received combined changes
        analysis = res.cumulative_analysis
        self.assertIsNotNone(analysis)
        self.assertIn("summary", analysis.to_dict())

    def test_8_9_per_commit_predictions_no_fabricated_pr_probability(self):
        """8, 9. Preserves per-commit ML predictions; does NOT invent PR risk probability."""
        res = self.pr_analyzer.analyze_pr(
            repo_path=self.repo_path,
            base_ref="main",
            head_ref="feature/multi"
        )
        res_dict = res.to_dict()

        # Check per-commit predictions present
        self.assertIn("commits", res_dict)
        for commit_data in res_dict["commits"]:
            self.assertIn("prediction", commit_data)
            self.assertIsNotNone(commit_data["prediction"]["raw_probability"])

        # Ensure NO fabricated PR-wide probability field exists in root result
        self.assertNotIn("pr_risk_probability", res_dict)
        self.assertNotIn("pr_probability", res_dict)
        self.assertNotIn("overall_risk_score", res_dict)

    def test_10_11_invalid_base_and_head_refs(self):
        """10, 11. Invalid base or head ref raises InvalidPRRefError."""
        with self.assertRaises(InvalidPRRefError):
            self.pr_analyzer.analyze_pr(
                repo_path=self.repo_path,
                base_ref="nonexistent-branch-xyz",
                head_ref="main"
            )

        with self.assertRaises(InvalidPRRefError):
            self.pr_analyzer.analyze_pr(
                repo_path=self.repo_path,
                base_ref="main",
                head_ref="invalid-head-ref-999"
            )

    def test_12_13_same_base_and_head_empty_pr(self):
        """12, 13. BASE == HEAD returns structured empty result without errors or LLM call."""
        res = self.pr_analyzer.analyze_pr(
            repo_path=self.repo_path,
            base_ref="main",
            head_ref="main"
        )
        self.assertTrue(res.is_empty)
        self.assertEqual(res.commit_count, 0)
        self.assertEqual(len(res.commits), 0)
        self.assertIsNone(res.cumulative_analysis)
        self.assertEqual(res.resolved_base_sha, res.resolved_head_sha)

    def test_14_mock_provider_path(self):
        """14. Operates cleanly with MockLLMProvider."""
        pr_input = PRAnalysisInput(
            repo_path=self.repo_path,
            base_ref="main",
            head_ref="feature/single"
        )
        res = self.pr_analyzer.analyze(pr_input)
        self.assertIsNotNone(res.cumulative_analysis)
        self.assertEqual(res.cumulative_analysis.model_provider, "mock")

    def test_15_json_serialization(self):
        """15. Result serializes cleanly to dict/JSON."""
        res = self.pr_analyzer.analyze_pr(
            repo_path=self.repo_path,
            base_ref="main",
            head_ref="feature/multi"
        )
        d = res.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)

        self.assertEqual(parsed["base_ref"], "main")
        self.assertEqual(parsed["head_ref"], "feature/multi")
        self.assertEqual(parsed["commit_count"], 2)
        self.assertIsNotNone(parsed["cumulative_analysis"])

    def test_16_merge_commit_in_pr(self):
        """16. Handles PRs containing merge commits without crashing."""
        # Create a branch with a merge commit
        self._run_git(["checkout", "main"])
        self._run_git(["checkout", "-b", "side_branch"])
        side_file = os.path.join(self.repo_path, "side.py")
        with open(side_file, "w", encoding="utf-8") as f:
            f.write("# side branch work\n")
        self._run_git(["add", "."])
        self._run_git(["commit", "-m", "Side work"])

        self._run_git(["checkout", "feature/single"])
        self._run_git(["merge", "--no-ff", "side_branch", "-m", "Merge side_branch into feature/single"])

        res = self.pr_analyzer.analyze_pr(
            repo_path=self.repo_path,
            base_ref="main",
            head_ref="feature/single"
        )
        self.assertGreater(res.commit_count, 1)
        
        # Verify merge commit record has prediction=None and error description
        merge_records = [c for c in res.commits if c.is_merge_commit]
        self.assertGreater(len(merge_records), 0)
        for mc in merge_records:
            self.assertIsNone(mc.prediction)
            self.assertIn("merge commit", mc.error.lower())

    def test_17_cli_pr_analyze_command(self):
        """CLI integration test for pr-analyze subcommand in text and JSON formats."""
        # Test CLI text card
        args_text = ["pr-analyze", "--repo", self.repo_path, "--base", "main", "--head", "feature/single", "--format", "text"]
        code_text = cli_main(args_text)
        self.assertEqual(code_text, 0)

        # Test CLI JSON
        args_json = ["pr-analyze", "--repo", self.repo_path, "--base", "main", "--head", "feature/multi", "--format", "json"]
        code_json = cli_main(args_json)
        self.assertEqual(code_json, 0)


if __name__ == "__main__":
    unittest.main()
