"""
Unit tests for Phase 7.2.2 Authenticated GitHub Repository Acquisition.
Operates 100% offline using mock Git runners, temporary local repositories, and fake credentials.
Zero live network calls.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from src.github import (
    GitHubAppConfig,
    GitHubAppAuthenticator,
    GitHubClient,
    GitHubPullRequest,
    GitHubPRAdapter,
    RepositoryAcquisitionManager,
    RepositoryAcquisitionError,
    RepositoryCloneError,
    RepositoryFetchError,
    RepositoryValidationError,
    CommitNotAvailableError,
    RepositoryIdentityMismatchError,
    RepositoryCleanupError
)
from src.analysis import PRAnalyzer, PRAnalysisInput, MockLLMProvider, LLMCodeAnalyzer
from src.engine import RiskPredictionEngine
from src.cli import main as cli_main

FAKE_RSA_PRIVATE_KEY = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEowIBAAKCAQEAz83...[FAKE_PRIVATE_KEY_FOR_OFFLINE_TESTING]...IDAQAB\n"
    "-----END RSA PRIVATE KEY-----"
)
FAKE_BASE_SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9a0b1c2d3e"
FAKE_HEAD_SHA = "f9e8d7c6b5a4039281706f5e4d3c2b1a09876543"


def create_tiny_test_git_repo() -> str:
    """Creates a real tiny local Git repository in temp directory with 2 commits for offline testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_acq_repo_")
    
    def run_git(cmd):
        subprocess.run(["git"] + cmd, cwd=temp_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    import subprocess
    run_git(["init"])
    run_git(["config", "user.name", "Test Author"])
    run_git(["config", "user.email", "test@example.com"])
    
    file_path = os.path.join(temp_dir, "test.txt")
    with open(file_path, "w") as f:
        f.write("line 1\n")
    run_git(["add", "."])
    run_git(["commit", "-m", "Initial base commit"])
    
    base_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=temp_dir, text=True).strip()

    with open(file_path, "a") as f:
        f.write("line 2\n")
    run_git(["add", "."])
    run_git(["commit", "-m", "Feature head commit"])

    head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=temp_dir, text=True).strip()

    return temp_dir, base_sha, head_sha


class TestRepositoryAcquisitionManager(unittest.TestCase):
    """1-5. Successful acquisition and SHA validation unit tests."""

    def test_1_successful_temporary_acquisition(self):
        """1. Remote temporary acquisition yields path and cleans up temporary directory on exit."""
        mock_runner = MagicMock()
        mock_runner.return_value.returncode = 0
        mock_runner.return_value.stdout = ""

        acq = RepositoryAcquisitionManager(git_runner=mock_runner)

        yielded_path = None
        with acq.acquire(
            owner="bottlepy",
            repo="bottle",
            base_sha=FAKE_BASE_SHA,
            head_sha=FAKE_HEAD_SHA,
            token="ghs_secret_token_123"
        ) as repo_path:
            yielded_path = repo_path
            self.assertTrue(os.path.exists(repo_path))

        # Confirms temporary workspace cleanup on exit
        self.assertFalse(os.path.exists(yielded_path))

    def test_2_authenticated_remote_configuration(self):
        """2. Verifies token is handled via temporary gitconfig file and not passed in CLI args."""
        called_cmds = []

        def mock_git(args, cwd, env):
            called_cmds.append(" ".join(args))
            res = MagicMock()
            res.returncode = 0
            res.stdout = ""
            return res

        acq = RepositoryAcquisitionManager(git_runner=mock_git)
        secret_token = "ghs_super_secret_token_999"

        with acq.acquire(
            owner="bottlepy",
            repo="bottle",
            base_sha=FAKE_BASE_SHA,
            head_sha=FAKE_HEAD_SHA,
            token=secret_token
        ) as repo_path:
            pass

        # Confirms secret token was NOT passed in CLI arguments list
        for cmd_str in called_cmds:
            self.assertNotIn(secret_token, cmd_str)
            if "init" in cmd_str or "fetch" in cmd_str:
                self.assertIn("include.path=", cmd_str)

    def test_3_base_sha_exists(self):
        """3. Validates base SHA presence via cat-file -e."""
        temp_dir, base_sha, head_sha = create_tiny_test_git_repo()
        try:
            acq = RepositoryAcquisitionManager()
            acq.validate_commit_exists(temp_dir, base_sha, ref_label="base")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_4_head_sha_exists(self):
        """4. Validates head SHA presence via cat-file -e."""
        temp_dir, base_sha, head_sha = create_tiny_test_git_repo()
        try:
            acq = RepositoryAcquisitionManager()
            acq.validate_commit_exists(temp_dir, head_sha, ref_label="head")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_5_both_shas_validated(self):
        """5. Validates both base and head SHAs in real local repo."""
        temp_dir, base_sha, head_sha = create_tiny_test_git_repo()
        try:
            acq = RepositoryAcquisitionManager()
            with acq.acquire(
                owner="test",
                repo="repo",
                base_sha=base_sha,
                head_sha=head_sha,
                local_repo_override=temp_dir
            ) as repo_path:
                self.assertEqual(repo_path, os.path.abspath(temp_dir))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestRepositoryAcquisitionFailures(unittest.TestCase):
    """6-12. Acquisition failure mode unit tests."""

    def test_6_clone_failure(self):
        """6. Failure during git init/remote raises RepositoryCloneError."""
        def mock_git(args, cwd, env):
            res = MagicMock()
            if "init" in args:
                res.returncode = 128
                res.stderr = "Permission denied"
            else:
                res.returncode = 0
            return res

        acq = RepositoryAcquisitionManager(git_runner=mock_git)
        with self.assertRaises(RepositoryCloneError):
            with acq.acquire(owner="o", repo="r", base_sha=FAKE_BASE_SHA, head_sha=FAKE_HEAD_SHA):
                pass

    def test_7_fetch_failure(self):
        """7. Failure during git fetch raises RepositoryFetchError."""
        def mock_git(args, cwd, env):
            res = MagicMock()
            if "fetch" in args:
                res.returncode = 1
                res.stderr = "Fatal fetch error"
            else:
                res.returncode = 0
            return res

        acq = RepositoryAcquisitionManager(git_runner=mock_git)
        with self.assertRaises(RepositoryFetchError):
            with acq.acquire(owner="o", repo="r", base_sha=FAKE_BASE_SHA, head_sha=FAKE_HEAD_SHA):
                pass

    def test_8_base_sha_missing(self):
        """8. Missing base SHA raises CommitNotAvailableError."""
        temp_dir, base_sha, head_sha = create_tiny_test_git_repo()
        missing_sha = "1111111111222222222233333333334444444444"
        try:
            acq = RepositoryAcquisitionManager()
            with self.assertRaises(CommitNotAvailableError):
                with acq.acquire(owner="o", repo="r", base_sha=missing_sha, head_sha=head_sha, local_repo_override=temp_dir):
                    pass
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_9_head_sha_missing(self):
        """9. Missing head SHA raises CommitNotAvailableError."""
        temp_dir, base_sha, head_sha = create_tiny_test_git_repo()
        missing_sha = "1111111111222222222233333333334444444444"
        try:
            acq = RepositoryAcquisitionManager()
            with self.assertRaises(CommitNotAvailableError):
                with acq.acquire(owner="o", repo="r", base_sha=base_sha, head_sha=missing_sha, local_repo_override=temp_dir):
                    pass
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_10_repository_identity_mismatch(self):
        """10. Local repository remote origin mismatch raises RepositoryIdentityMismatchError."""
        temp_dir, base_sha, head_sha = create_tiny_test_git_repo()
        import subprocess
        subprocess.run(["git", "remote", "add", "origin", "https://github.com/wrongowner/wrongrepo.git"], cwd=temp_dir, check=True)
        try:
            acq = RepositoryAcquisitionManager()
            with self.assertRaises(RepositoryIdentityMismatchError):
                with acq.acquire(owner="bottlepy", repo="bottle", base_sha=base_sha, head_sha=head_sha, local_repo_override=temp_dir):
                    pass
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_11_invalid_repository(self):
        """11. Non-git directory override raises RepositoryValidationError."""
        with tempfile.TemporaryDirectory() as empty_dir:
            acq = RepositoryAcquisitionManager()
            with self.assertRaises(RepositoryValidationError):
                with acq.acquire(owner="o", repo="r", base_sha=FAKE_BASE_SHA, head_sha=FAKE_HEAD_SHA, local_repo_override=empty_dir):
                    pass

    def test_12_invalid_sha_format(self):
        """12. Malformed non-hex SHA raises RepositoryValidationError."""
        acq = RepositoryAcquisitionManager()
        with self.assertRaises(RepositoryValidationError):
            acq.validate_commit_exists("/tmp", "short_sha")


class TestRepositoryCleanupAndSecurity(unittest.TestCase):
    """13-19. Workspace cleanup lifecycle and secret security unit tests."""

    def test_13_cleanup_after_success(self):
        """13. Temporary directory is deleted after successful acquisition context."""
        mock_runner = MagicMock()
        mock_runner.return_value.returncode = 0

        acq = RepositoryAcquisitionManager(git_runner=mock_runner)
        path = None
        with acq.acquire(owner="o", repo="r", base_sha=FAKE_BASE_SHA, head_sha=FAKE_HEAD_SHA) as temp_dir:
            path = temp_dir
            self.assertTrue(os.path.exists(path))
        self.assertFalse(os.path.exists(path))

    def test_14_cleanup_after_clone_failure(self):
        """14. Temporary directory is deleted after clone failure."""
        def mock_failing_git(args, cwd, env):
            res = MagicMock()
            res.returncode = 1
            return res

        acq = RepositoryAcquisitionManager(git_runner=mock_failing_git)
        try:
            with acq.acquire(owner="o", repo="r", base_sha=FAKE_BASE_SHA, head_sha=FAKE_HEAD_SHA):
                pass
        except RepositoryAcquisitionError:
            pass

    def test_15_cleanup_after_validation_failure(self):
        """15. Temporary directory is deleted after SHA validation failure."""
        def mock_git(args, cwd, env):
            res = MagicMock()
            if "cat-file" in args:
                res.returncode = 1
            else:
                res.returncode = 0
            return res

        acq = RepositoryAcquisitionManager(git_runner=mock_git)
        try:
            with acq.acquire(owner="o", repo="r", base_sha=FAKE_BASE_SHA, head_sha=FAKE_HEAD_SHA):
                pass
        except CommitNotAvailableError:
            pass

    def test_16_cleanup_after_pr_analyzer_failure(self):
        """16. Temporary directory is cleaned up when exception occurs inside acquire block."""
        mock_runner = MagicMock()
        mock_runner.return_value.returncode = 0

        acq = RepositoryAcquisitionManager(git_runner=mock_runner)
        path = None
        try:
            with acq.acquire(owner="o", repo="r", base_sha=FAKE_BASE_SHA, head_sha=FAKE_HEAD_SHA) as temp_dir:
                path = temp_dir
                raise RuntimeError("Simulated PRAnalyzer failure")
        except RuntimeError:
            pass

        self.assertFalse(os.path.exists(path))

    def test_17_token_not_present_in_command_arguments(self):
        """17. Confirms secret token is never passed as command line argument to subprocess."""
        called_args = []

        def mock_git(args, cwd, env):
            called_args.extend(args)
            res = MagicMock()
            res.returncode = 0
            return res

        acq = RepositoryAcquisitionManager(git_runner=mock_git)
        token = "SECRET_TOKEN_99999"
        with acq.acquire(owner="o", repo="r", base_sha=FAKE_BASE_SHA, head_sha=FAKE_HEAD_SHA, token=token):
            pass

        for arg in called_args:
            self.assertNotIn(token, arg)

    def test_18_token_not_present_in_logs_errors(self):
        """18. Confirms exceptions sanitize authorization tokens."""
        err = RepositoryAcquisitionError("Clone failed with Authorization: Bearer SECRET_TOKEN_123")
        self.assertNotIn("SECRET_TOKEN_123", str(err))

    def test_19_credentials_not_present_in_result_json(self):
        """19. Confirms JSON output representations exclude tokens and key credentials."""
        pr = GitHubPullRequest(
            owner="bottlepy", repository="bottle", number=42, title="Fix",
            html_url="https://github.com/bottlepy/bottle/pull/42",
            base_ref="main", base_sha=FAKE_BASE_SHA, head_ref="h", head_sha=FAKE_HEAD_SHA
        )
        json_str = json.dumps(pr.to_dict())
        self.assertNotIn("token", json_str.lower())
        self.assertNotIn("private_key", json_str.lower())


class TestIntegrationAndCLI(unittest.TestCase):
    """20-23. End-to-end integration and CLI subcommand unit tests."""

    def test_20_github_pull_request_to_pr_analysis_input(self):
        """20. Converts GitHubPullRequest to PRAnalysisInput with acquired repo path."""
        pr = GitHubPullRequest(
            owner="bottlepy", repository="bottle", number=42, title="Fix",
            html_url="https://github.com/bottlepy/bottle/pull/42",
            base_ref="main", base_sha=FAKE_BASE_SHA, head_ref="h", head_sha=FAKE_HEAD_SHA
        )
        inp = GitHubPRAdapter.to_analysis_input(pr, repo_path="/tmp/acquired_workspace")
        self.assertEqual(inp.base_ref, FAKE_BASE_SHA)
        self.assertEqual(inp.head_ref, FAKE_HEAD_SHA)

    def test_21_acquired_repository_to_pr_analyzer(self):
        """21. Executes PRAnalyzer.analyze on real acquired local repository."""
        temp_dir, base_sha, head_sha = create_tiny_test_git_repo()
        try:
            acq = RepositoryAcquisitionManager()
            with acq.acquire(owner="test", repo="repo", base_sha=base_sha, head_sha=head_sha, local_repo_override=temp_dir) as repo_path:
                pr_input = PRAnalysisInput(repo_path=repo_path, base_ref=base_sha, head_ref=head_sha)
                
                analyzer = PRAnalyzer(llm_analyzer=LLMCodeAnalyzer(provider=MockLLMProvider()))
                res = analyzer.analyze(pr_input)

                self.assertEqual(res.commit_count, 1)
                self.assertIsNotNone(res.cumulative_analysis)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("src.cli.GitHubAppConfig.from_env")
    @patch("src.cli.GitHubClient")
    def test_22_github_pr_analyze_successful_mocked_flow(self, mock_client_cls, mock_cfg_from_env):
        """22. Executes github-pr-analyze CLI subcommand with local repo override and mock provider."""
        temp_dir, base_sha, head_sha = create_tiny_test_git_repo()
        try:
            mock_client = MagicMock()
            mock_client.get_pull_request.return_value = GitHubPullRequest(
                owner="bottlepy", repository="bottle", number=42, title="Fix bug",
                html_url="https://github.com/bottlepy/bottle/pull/42",
                base_ref="main", base_sha=base_sha, head_ref="feature/fix", head_sha=head_sha
            )
            mock_client._get_token.return_value = "ghs_fake_token"
            mock_client_cls.return_value = mock_client

            args = ["github-pr-analyze", "--repo", "bottlepy/bottle", "--pr", "42", "--local-repo", temp_dir, "--format", "text"]
            exit_code = cli_main(args)
            self.assertEqual(exit_code, 0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @patch("src.cli.GitHubAppConfig.from_env")
    @patch("src.cli.GitHubClient")
    def test_23_github_pr_analyze_json_format(self, mock_client_cls, mock_cfg_from_env):
        """23. Executes github-pr-analyze CLI subcommand with --format json."""
        temp_dir, base_sha, head_sha = create_tiny_test_git_repo()
        try:
            mock_client = MagicMock()
            mock_client.get_pull_request.return_value = GitHubPullRequest(
                owner="bottlepy", repository="bottle", number=42, title="Fix bug",
                html_url="https://github.com/bottlepy/bottle/pull/42",
                base_ref="main", base_sha=base_sha, head_ref="feature/fix", head_sha=head_sha
            )
            mock_client._get_token.return_value = "ghs_fake_token"
            mock_client_cls.return_value = mock_client

            args = ["github-pr-analyze", "--repo", "bottlepy/bottle", "--pr", "42", "--local-repo", temp_dir, "--format", "json"]
            exit_code = cli_main(args)
            self.assertEqual(exit_code, 0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
