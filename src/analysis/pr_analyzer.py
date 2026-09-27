"""
Local Pull Request analysis orchestrator for Phase 7.1.
Combines per-commit Phase 5 ML defect predictions with cumulative BASE->HEAD Phase 6 LLM analysis.
"""

import os
import subprocess
import shutil
from typing import List, Optional

from src.engine import RiskPredictionEngine
from src.engine.exceptions import (
    RepositoryNotFoundError,
    InvalidGitRepositoryError
)
from src.analysis.diff_extractor import GitDiffExtractor
from src.analysis.analyzer import LLMCodeAnalyzer
from src.analysis.pr_schema import PRAnalysisInput, PRCommitPrediction, PRAnalysisResult
from src.analysis.exceptions import InvalidPRRefError


def _find_git_executable() -> str:
    path = shutil.which("git")
    if path:
        return path
    for fallback in [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe"
    ]:
        if os.path.exists(fallback):
            return fallback
    return "git"


class PRAnalyzer:
    """
    Orchestrates local PR-level analysis.
    - Resolves base and head references and determines merge-base.
    - Enumerates PR commits and evaluates per-commit ML risk via RiskPredictionEngine.
    - Extracts cumulative diff and evaluates qualitative LLM analysis via LLMCodeAnalyzer.
    - Reuses existing Phase 5 and Phase 6 components without duplicating extraction or ML logic.
    """
    def __init__(
        self,
        predictor: Optional[RiskPredictionEngine] = None,
        llm_analyzer: Optional[LLMCodeAnalyzer] = None
    ):
        self.predictor = predictor if predictor is not None else RiskPredictionEngine()
        self.llm_analyzer = llm_analyzer if llm_analyzer is not None else LLMCodeAnalyzer()

    def _run_git(self, repo_path: str, args: List[str]) -> subprocess.CompletedProcess:
        git_executable = _find_git_executable()
        cmd = [git_executable] + args
        return subprocess.run(
            cmd,
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

    def analyze(
        self,
        pr_input: PRAnalysisInput,
        threshold: Optional[float] = None
    ) -> PRAnalysisResult:
        """
        Executes PR analysis using a PRAnalysisInput object.
        """
        return self.analyze_pr(
            repo_path=pr_input.repo_path,
            base_ref=pr_input.base_ref,
            head_ref=pr_input.head_ref,
            threshold=threshold
        )

    def analyze_pr(
        self,
        repo_path: str,
        base_ref: str,
        head_ref: str,
        threshold: Optional[float] = None
    ) -> PRAnalysisResult:
        """
        Executes PR analysis for specified base and head references in a repository.
        """
        repo_abs = os.path.abspath(repo_path)
        if not os.path.exists(repo_abs):
            raise RepositoryNotFoundError(f"Repository directory does not exist: {repo_abs}")

        rev_check = self._run_git(repo_abs, ["rev-parse", "--is-inside-work-tree"])
        if rev_check.returncode != 0:
            raise InvalidGitRepositoryError(f"Directory is not a valid Git repository: {repo_abs}")

        # 1. Resolve base and head references to full 40-character SHAs
        res_base = self._run_git(repo_abs, ["rev-parse", "--verify", f"{base_ref}^{{commit}}"])
        if res_base.returncode != 0:
            raise InvalidPRRefError(f"Invalid base reference: '{base_ref}'")
        resolved_base_sha = res_base.stdout.strip()

        res_head = self._run_git(repo_abs, ["rev-parse", "--verify", f"{head_ref}^{{commit}}"])
        if res_head.returncode != 0:
            raise InvalidPRRefError(f"Invalid head reference: '{head_ref}'")
        resolved_head_sha = res_head.stdout.strip()

        # 2. Compute merge base
        mb_res = self._run_git(repo_abs, ["merge-base", resolved_base_sha, resolved_head_sha])
        if mb_res.returncode != 0 or not mb_res.stdout.strip():
            raise InvalidPRRefError(f"No common ancestor (merge-base) found between '{base_ref}' and '{head_ref}'")
        resolved_merge_base_sha = mb_res.stdout.strip()

        # 3. Check for empty PR / Same base & head
        if resolved_base_sha == resolved_head_sha or resolved_merge_base_sha == resolved_head_sha:
            return PRAnalysisResult(
                repo_path=repo_abs,
                base_ref=base_ref,
                head_ref=head_ref,
                resolved_base_sha=resolved_base_sha,
                resolved_head_sha=resolved_head_sha,
                resolved_merge_base_sha=resolved_merge_base_sha,
                is_empty=True,
                commit_count=0,
                commits=[],
                cumulative_analysis=None
            )

        # 4. Enumerate commits introduced by the PR (merge_base..head)
        rev_list_res = self._run_git(repo_abs, ["rev-list", "--reverse", f"{resolved_merge_base_sha}..{resolved_head_sha}"])
        commit_shas = [s.strip() for s in rev_list_res.stdout.splitlines() if s.strip()]

        commit_records: List[PRCommitPrediction] = []

        for commit_sha in commit_shas:
            # Get commit log info
            log_res = self._run_git(repo_abs, ["log", "-1", "--format=%H|%an|%aI|%P%n%B", commit_sha])
            lines = log_res.stdout.splitlines() if log_res.returncode == 0 and log_res.stdout else []
            header_line = lines[0] if lines else ""
            commit_msg = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

            parts = header_line.split('|')
            author = parts[1] if len(parts) > 1 else "Unknown"
            timestamp = parts[2] if len(parts) > 2 else ""
            parents = parts[3].split() if len(parts) > 3 and parts[3] else []
            is_merge = len(parents) > 1

            if is_merge:
                record = PRCommitPrediction(
                    commit_hash=commit_sha[:7],
                    full_hash=commit_sha,
                    author=author,
                    timestamp=timestamp,
                    commit_message=commit_msg,
                    is_merge_commit=True,
                    prediction=None,
                    error=f"Commit {commit_sha[:7]} is a merge commit with {len(parents)} parents. ML risk prediction skipped."
                )
            else:
                try:
                    pred = self.predictor.predict(
                        repo_path=repo_abs,
                        commit_hash=commit_sha,
                        threshold=threshold
                    )
                    record = PRCommitPrediction(
                        commit_hash=commit_sha[:7],
                        full_hash=commit_sha,
                        author=author,
                        timestamp=timestamp,
                        commit_message=commit_msg,
                        is_merge_commit=False,
                        prediction=pred,
                        error=None
                    )
                except Exception as ex:
                    record = PRCommitPrediction(
                        commit_hash=commit_sha[:7],
                        full_hash=commit_sha,
                        author=author,
                        timestamp=timestamp,
                        commit_message=commit_msg,
                        is_merge_commit=False,
                        prediction=None,
                        error=str(ex)
                    )

            commit_records.append(record)

        # 5. Extract cumulative diff (BASE -> HEAD via merge-base) and run LLM analysis
        diff_extractor = GitDiffExtractor(repo_abs)
        cumulative_diff = diff_extractor.extract_cumulative_diff(base_ref, head_ref)

        if cumulative_diff.is_empty_commit:
            cumulative_analysis = None
        else:
            cumulative_analysis = self.llm_analyzer.analyze_commit(
                commit_diff=cumulative_diff,
                prediction_result=None
            )

        return PRAnalysisResult(
            repo_path=repo_abs,
            base_ref=base_ref,
            head_ref=head_ref,
            resolved_base_sha=resolved_base_sha,
            resolved_head_sha=resolved_head_sha,
            resolved_merge_base_sha=resolved_merge_base_sha,
            is_empty=len(commit_records) == 0 or (cumulative_diff.is_empty_commit if cumulative_diff else False),
            commit_count=len(commit_records),
            commits=commit_records,
            cumulative_analysis=cumulative_analysis
        )
