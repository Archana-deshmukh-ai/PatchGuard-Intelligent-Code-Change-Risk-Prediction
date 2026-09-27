"""
Authenticated GitHub repository acquisition and temporary workspace management for PatchGuard Phase 7.2.2.
Clones/fetches GitHub repositories using GitHub App installation access tokens into isolated temporary directories
and validates that required base and head commit SHAs exist before handing off to PRAnalyzer.
Strict security enforcement: zero repository code execution, zero token leakage in CLI arguments/logs/exceptions.
"""

import os
import re
import shutil
import tempfile
import subprocess
from contextlib import contextmanager
from typing import Optional, List, Generator, Any

from src.github.exceptions import (
    RepositoryAcquisitionError,
    RepositoryCloneError,
    RepositoryFetchError,
    RepositoryValidationError,
    CommitNotAvailableError,
    RepositoryIdentityMismatchError,
    RepositoryCleanupError
)

HEX_SHA1_REGEX = re.compile(r'^[0-9a-fA-F]{40}$')


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


class RepositoryAcquisitionManager:
    """
    Manages authenticated repository acquisition, Git operations, SHA validation,
    and temporary workspace lifecycle.
    """
    def __init__(
        self,
        git_executable: Optional[str] = None,
        git_runner: Optional[Any] = None
    ):
        self.git_executable = git_executable if git_executable is not None else _find_git_executable()
        self.git_runner = git_runner

    def _run_git(
        self,
        args: List[str],
        cwd: str,
        env: Optional[dict] = None
    ) -> subprocess.CompletedProcess:
        """Executes Git commands safely without exposing sensitive tokens in subprocess logs."""
        if self.git_runner is not None:
            return self.git_runner(args, cwd, env)

        cmd = [self.git_executable] + args
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        return subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=full_env
        )

    def validate_commit_exists(self, repo_path: str, commit_sha: str, ref_label: str = "commit"):
        """Validates that a 40-character hexadecimal commit SHA exists in the local repository."""
        if not commit_sha or not HEX_SHA1_REGEX.match(commit_sha):
            raise RepositoryValidationError(f"Invalid {ref_label} commit SHA format: '{commit_sha}'")

        res = self._run_git(["cat-file", "-e", f"{commit_sha}^{{commit}}"], cwd=repo_path)
        if res.returncode != 0:
            raise CommitNotAvailableError(
                f"Requested {ref_label} commit SHA '{commit_sha[:12]}' is not present in repository '{repo_path}'."
            )

    def validate_repository_identity(self, repo_path: str, owner: str, repo: str):
        """
        Validates that a local repository's configured origin remote corresponds to owner/repo.
        """
        res = self._run_git(["config", "--get", "remote.origin.url"], cwd=repo_path)
        if res.returncode != 0 or not res.stdout.strip():
            return

        remote_url = res.stdout.strip()
        expected = f"{owner.lower()}/{repo.lower()}"
        clean_url = remote_url.rstrip('/').lower()
        if clean_url.endswith('.git'):
            clean_url = clean_url[:-4]

        if not (clean_url.endswith(f"/{expected}") or clean_url.endswith(f":{expected}")):
            raise RepositoryIdentityMismatchError(
                f"Local repository remote URL '{remote_url}' does not match expected GitHub repository '{owner}/{repo}'."
            )

    @contextmanager
    def acquire(
        self,
        owner: str,
        repo: str,
        base_sha: str,
        head_sha: str,
        token: Optional[str] = None,
        local_repo_override: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Context manager for acquiring and validating a Git repository.
        If local_repo_override is provided, validates and yields the local directory without deleting it.
        Otherwise, creates a temporary workspace, clones/fetches the remote repository, validates commits,
        yields the path, and guarantees cleanup upon exit.
        """
        if local_repo_override:
            abs_local = os.path.abspath(local_repo_override)
            if not os.path.exists(abs_local):
                raise RepositoryValidationError(f"Local repository override path does not exist: '{abs_local}'")

            rev_check = self._run_git(["rev-parse", "--is-inside-work-tree"], cwd=abs_local)
            if rev_check.returncode != 0:
                raise RepositoryValidationError(f"Path is not a valid Git repository: '{abs_local}'")

            self.validate_repository_identity(abs_local, owner, repo)
            self.validate_commit_exists(abs_local, base_sha, ref_label="base")
            self.validate_commit_exists(abs_local, head_sha, ref_label="head")

            yield abs_local
            return

        # Remote Acquisition Path (Temporary Workspace)
        temp_dir = tempfile.mkdtemp(prefix="patchguard_repo_")
        git_config_path = os.path.join(temp_dir, ".patchguard_gitconfig")

        try:
            # Secure token handling via include.path git config file
            if token:
                with open(git_config_path, "w", encoding="utf-8") as f:
                    f.write("[http]\n")
                    f.write(f"    extraheader = Authorization: Bearer {token}\n")

            clone_url = f"https://github.com/{owner}/{repo}.git"

            # Prepare git command arguments with secure config file inclusion
            git_config_args = ["-c", f"include.path={git_config_path}"] if token else []

            # 1. Initialize git repo in temporary directory
            init_res = self._run_git(git_config_args + ["init", "."], cwd=temp_dir)
            if init_res.returncode != 0:
                raise RepositoryCloneError(f"Failed to initialize temporary Git repository for '{owner}/{repo}'.")

            # 2. Add remote origin
            remote_res = self._run_git(git_config_args + ["remote", "add", "origin", clone_url], cwd=temp_dir)
            if remote_res.returncode != 0:
                raise RepositoryCloneError(f"Failed to configure remote origin for '{owner}/{repo}'.")

            # 3. Fetch base_sha and head_sha
            fetch_res = self._run_git(git_config_args + ["fetch", "--depth=100", "origin", base_sha, head_sha], cwd=temp_dir)
            if fetch_res.returncode != 0:
                # Fallback to general fetch if commit SHA fetch fails
                fetch_res_2 = self._run_git(git_config_args + ["fetch", "--depth=100", "origin"], cwd=temp_dir)
                if fetch_res_2.returncode != 0:
                    raise RepositoryFetchError(
                        f"Failed to fetch commit objects for repository '{owner}/{repo}' from GitHub remote."
                    )

            # 4. Validate base and head commit SHAs exist
            self.validate_commit_exists(temp_dir, base_sha, ref_label="base")
            self.validate_commit_exists(temp_dir, head_sha, ref_label="head")

            yield temp_dir

        finally:
            # Guaranteed workspace cleanup
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=False)
            except Exception as e:
                raise RepositoryCleanupError(f"Failed to clean up temporary repository directory '{temp_dir}': {e}")
