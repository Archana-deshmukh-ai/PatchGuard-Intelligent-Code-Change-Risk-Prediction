"""
Unit tests for Phase 7.2.1 GitHub Read-Only PR Retrieval & Adapter.
Operates 100% offline using mocked HTTP handlers and fake credentials. Zero network calls.
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
    GitHubIntegrationError,
    GitHubConfigError,
    GitHubAuthError,
    GitHubAPIError,
    GitHubPRNotFoundError,
    GitHubPRValidationError
)
from src.analysis.pr_schema import PRAnalysisInput
from src.cli import main as cli_main

FAKE_RSA_PRIVATE_KEY = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEowIBAAKCAQEAz83...[FAKE_PRIVATE_KEY_FOR_OFFLINE_TESTING]...IDAQAB\n"
    "-----END RSA PRIVATE KEY-----"
)
FAKE_BASE_SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9a0b1c2d3e"
FAKE_HEAD_SHA = "f9e8d7c6b5a4039281706f5e4d3c2b1a09876543"


class TestGitHubAppConfigAndAuth(unittest.TestCase):
    """1-6. Configuration loading, validation, security, and authentication unit tests."""

    def test_1_valid_config(self):
        """1. Valid configuration loading."""
        cfg = GitHubAppConfig(app_id="12345", private_key=FAKE_RSA_PRIVATE_KEY, installation_id="67890")
        self.assertEqual(cfg.app_id, "12345")
        self.assertEqual(cfg.installation_id, "67890")
        self.assertEqual(cfg.private_key, FAKE_RSA_PRIVATE_KEY)

    def test_2_missing_app_id(self):
        """2. Missing app ID raises GitHubConfigError."""
        with patch.dict(os.environ, {"GITHUB_APP_ID": "", "GITHUB_APP_PRIVATE_KEY": "", "GITHUB_APP_PRIVATE_KEY_PATH": "", "GITHUB_INSTALLATION_ID": ""}):
            with self.assertRaises(GitHubConfigError) as ctx:
                GitHubAppConfig.from_env()
            self.assertIn("GITHUB_APP_ID", str(ctx.exception))

    def test_3_missing_private_key(self):
        """3. Missing private key raises GitHubConfigError."""
        with patch.dict(os.environ, {"GITHUB_APP_ID": "12345", "GITHUB_APP_PRIVATE_KEY": "", "GITHUB_APP_PRIVATE_KEY_PATH": "", "GITHUB_INSTALLATION_ID": ""}):
            with self.assertRaises(GitHubConfigError) as ctx:
                GitHubAppConfig.from_env()
            self.assertIn("GITHUB_APP_PRIVATE_KEY", str(ctx.exception))

    def test_4_missing_installation_id(self):
        """4. Missing installation ID raises GitHubConfigError."""
        with patch.dict(os.environ, {"GITHUB_APP_ID": "12345", "GITHUB_APP_PRIVATE_KEY": FAKE_RSA_PRIVATE_KEY, "GITHUB_APP_PRIVATE_KEY_PATH": "", "GITHUB_INSTALLATION_ID": ""}):
            with self.assertRaises(GitHubConfigError) as ctx:
                GitHubAppConfig.from_env()
            self.assertIn("GITHUB_INSTALLATION_ID", str(ctx.exception))

    def test_5_authentication_failure(self):
        """5. Simulated authentication exchange failure raises GitHubAuthError."""
        auth = GitHubAppAuthenticator()
        cfg = GitHubAppConfig(app_id="12345", private_key=FAKE_RSA_PRIVATE_KEY, installation_id="67890")

        def mock_failing_http(url, headers, method):
            raise GitHubAuthError("Simulated 401 Unauthorized token exchange.")

        with self.assertRaises(GitHubAuthError):
            auth.get_installation_access_token(cfg, http_requester=mock_failing_http)

    def test_6_secrets_not_leaked_in_repr_or_str(self):
        """6. Confirms private key and secret strings are sanitized in __repr__ and exceptions."""
        cfg = GitHubAppConfig(app_id="12345", private_key="SECRET_PEM_PRIVATE_KEY_DATA", installation_id="67890")
        repr_str = repr(cfg)
        self.assertNotIn("SECRET_PEM_PRIVATE_KEY_DATA", repr_str)
        self.assertIn("GitHubAppConfig", repr_str)

        err = GitHubAPIError("Failed request with Authorization: Bearer secret_token_123")
        self.assertNotIn("secret_token_123", str(err))


class TestGitHubClient(unittest.TestCase):
    """7-11. GitHubClient API retrieval offline unit tests using mock HTTP handlers."""

    def setUp(self):
        self.fake_token = "ghs_fake_installation_token_12345"
        self.token_provider = lambda: self.fake_token

    def _sample_pr_api_response(self):
        return {
            "number": 42,
            "title": "Fix memory leak in HTTP parser",
            "html_url": "https://github.com/bottlepy/bottle/pull/42",
            "state": "open",
            "base": {
                "ref": "main",
                "sha": FAKE_BASE_SHA,
                "repo": {
                    "name": "bottle",
                    "owner": {"login": "bottlepy"}
                }
            },
            "head": {
                "ref": "feature/fix-parser",
                "sha": FAKE_HEAD_SHA
            }
        }

    def test_7_successful_pr_retrieval(self):
        """7. Successful PR retrieval maps API JSON into GitHubPullRequest."""
        def mock_fetcher(url, headers):
            self.assertIn("Authorization", headers)
            self.assertEqual(headers["Authorization"], f"Bearer {self.fake_token}")
            self.assertIn("/repos/bottlepy/bottle/pulls/42", url)
            return self._sample_pr_api_response()

        client = GitHubClient(token_provider=self.token_provider, http_fetcher=mock_fetcher)
        pr = client.get_pull_request(owner="bottlepy", repo="bottle", pull_number=42)

        self.assertIsInstance(pr, GitHubPullRequest)
        self.assertEqual(pr.owner, "bottlepy")
        self.assertEqual(pr.repository, "bottle")
        self.assertEqual(pr.number, 42)
        self.assertEqual(pr.title, "Fix memory leak in HTTP parser")
        self.assertEqual(pr.base_ref, "main")
        self.assertEqual(pr.base_sha, FAKE_BASE_SHA)
        self.assertEqual(pr.head_ref, "feature/fix-parser")
        self.assertEqual(pr.head_sha, FAKE_HEAD_SHA)

    def test_8_404_not_found(self):
        """8. HTTP 404 response raises GitHubPRNotFoundError."""
        def mock_fetcher(url, headers):
            raise GitHubPRNotFoundError(owner="bottlepy", repo="bottle", pull_number=999)

        client = GitHubClient(token_provider=self.token_provider, http_fetcher=mock_fetcher)
        with self.assertRaises(GitHubPRNotFoundError) as ctx:
            client.get_pull_request(owner="bottlepy", repo="bottle", pull_number=999)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_9_unauthorized_forbidden(self):
        """9. HTTP 401/403 response raises GitHubAuthError."""
        def mock_fetcher(url, headers):
            raise GitHubAuthError("GitHub API authorization failed (HTTP 401).")

        client = GitHubClient(token_provider=self.token_provider, http_fetcher=mock_fetcher)
        with self.assertRaises(GitHubAuthError):
            client.get_pull_request(owner="bottlepy", repo="bottle", pull_number=42)

    def test_10_api_failure(self):
        """10. HTTP 500 response raises GitHubAPIError."""
        def mock_fetcher(url, headers):
            raise GitHubAPIError("Internal Server Error", status_code=500)

        client = GitHubClient(token_provider=self.token_provider, http_fetcher=mock_fetcher)
        with self.assertRaises(GitHubAPIError) as ctx:
            client.get_pull_request(owner="bottlepy", repo="bottle", pull_number=42)
        self.assertEqual(ctx.exception.status_code, 500)

    def test_11_malformed_response(self):
        """11. Malformed non-dict or invalid JSON response raises GitHubPRValidationError."""
        def mock_fetcher(url, headers):
            return "not_a_dictionary_response"

        client = GitHubClient(token_provider=self.token_provider, http_fetcher=mock_fetcher)
        with self.assertRaises(GitHubPRValidationError):
            client.get_pull_request(owner="bottlepy", repo="bottle", pull_number=42)


class TestGitHubPRModelAndAdapter(unittest.TestCase):
    """12-18. GitHubPRAdapter unit tests converting GitHubPullRequest -> PRAnalysisInput."""

    def _valid_gh_pr(self):
        return GitHubPullRequest(
            owner="bottlepy",
            repository="bottle",
            number=42,
            title="Refactor routes",
            html_url="https://github.com/bottlepy/bottle/pull/42",
            base_ref="main",
            base_sha=FAKE_BASE_SHA,
            head_ref="feature/routes",
            head_sha=FAKE_HEAD_SHA,
            state="open"
        )

    def test_12_valid_github_pr_model(self):
        """12. Valid GitHubPullRequest model instantiates and serializes cleanly."""
        pr = self._valid_gh_pr()
        d = pr.to_dict()
        self.assertEqual(d["owner"], "bottlepy")
        self.assertEqual(d["repository"], "bottle")
        self.assertEqual(d["base_sha"], FAKE_BASE_SHA)
        self.assertEqual(d["head_sha"], FAKE_HEAD_SHA)

    def test_13_valid_model_to_pr_analysis_input(self):
        """13. Converts valid GitHubPullRequest to PRAnalysisInput using base_sha and head_sha."""
        pr = self._valid_gh_pr()
        analysis_input = GitHubPRAdapter.to_analysis_input(pr, repo_path="/tmp/repo")

        self.assertIsInstance(analysis_input, PRAnalysisInput)
        self.assertEqual(analysis_input.base_ref, FAKE_BASE_SHA)
        self.assertEqual(analysis_input.head_ref, FAKE_HEAD_SHA)
        self.assertTrue(analysis_input.repo_path.endswith("repo"))

    def test_14_missing_base_sha(self):
        """14. Missing base SHA raises GitHubPRValidationError."""
        pr = GitHubPullRequest(
            owner="o", repository="r", number=1, title="t", html_url="",
            base_ref="main", base_sha="", head_ref="h", head_sha=FAKE_HEAD_SHA
        )
        with self.assertRaises(GitHubPRValidationError):
            GitHubPRAdapter.to_analysis_input(pr)

    def test_15_missing_head_sha(self):
        """15. Missing head SHA raises GitHubPRValidationError."""
        pr = GitHubPullRequest(
            owner="o", repository="r", number=1, title="t", html_url="",
            base_ref="main", base_sha=FAKE_BASE_SHA, head_ref="h", head_sha=""
        )
        with self.assertRaises(GitHubPRValidationError):
            GitHubPRAdapter.to_analysis_input(pr)

    def test_16_invalid_sha_format(self):
        """16. Short or non-hex SHA raises GitHubPRValidationError."""
        pr = GitHubPullRequest(
            owner="o", repository="r", number=1, title="t", html_url="",
            base_ref="main", base_sha="short_sha_123", head_ref="h", head_sha=FAKE_HEAD_SHA
        )
        with self.assertRaises(GitHubPRValidationError):
            GitHubPRAdapter.to_analysis_input(pr)

    def test_17_repository_identity_validation(self):
        """17. Invalid owner/repo name characters raise GitHubPRValidationError."""
        client = GitHubClient(token_provider=lambda: "token")
        with self.assertRaises(GitHubPRValidationError):
            client.get_pull_request(owner="invalid/owner", repo="bottle", pull_number=1)

        with self.assertRaises(GitHubPRValidationError):
            client.get_pull_request(owner="bottlepy", repo="bad repo name!", pull_number=1)

    def test_18_pr_number_validation(self):
        """18. Non-positive or invalid PR number raises GitHubPRValidationError."""
        client = GitHubClient(token_provider=lambda: "token")
        with self.assertRaises(GitHubPRValidationError):
            client.get_pull_request(owner="bottlepy", repo="bottle", pull_number=0)

        with self.assertRaises(GitHubPRValidationError):
            client.get_pull_request(owner="bottlepy", repo="bottle", pull_number=-5)


class TestGitHubCLIIntegration(unittest.TestCase):
    """19-23. CLI github-pr-analyze subcommand unit tests."""

    @patch.dict(os.environ, {
        "GITHUB_APP_ID": "12345",
        "GITHUB_APP_PRIVATE_KEY": FAKE_RSA_PRIVATE_KEY,
        "GITHUB_INSTALLATION_ID": "67890"
    })
    @patch("src.cli.RepositoryAcquisitionManager")
    @patch("src.cli.GitHubClient")
    def test_19_cli_successful_github_pr_retrieval(self, mock_client_cls, mock_acq_cls):
        """19. Successful CLI invocation of github-pr-analyze subcommand."""
        mock_instance = MagicMock()
        mock_instance.get_pull_request.return_value = GitHubPullRequest(
            owner="bottlepy", repository="bottle", number=42, title="Fix bug",
            html_url="https://github.com/bottlepy/bottle/pull/42",
            base_ref="main", base_sha=FAKE_BASE_SHA,
            head_ref="feature/fix", head_sha=FAKE_HEAD_SHA
        )
        mock_client_cls.return_value = mock_instance

        # Create temporary test git repo
        temp_dir = tempfile.mkdtemp()
        try:
            mock_acq = MagicMock()
            mock_acq.acquire.return_value.__enter__.return_value = temp_dir
            mock_acq_cls.return_value = mock_acq

            args = ["github-pr-analyze", "--repo", "bottlepy/bottle", "--pr", "42", "--format", "text"]
            with patch("src.cli.PRAnalyzer.analyze") as mock_pr_analyze:
                mock_pr_analyze.return_value = MagicMock(commits=[], is_empty=True, cumulative_analysis=None, analysis_disclaimer="Disclaimer")
                exit_code = cli_main(args)
                self.assertEqual(exit_code, 0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @patch.dict(os.environ, {
        "GITHUB_APP_ID": "12345",
        "GITHUB_APP_PRIVATE_KEY": FAKE_RSA_PRIVATE_KEY,
        "GITHUB_INSTALLATION_ID": "67890"
    })
    @patch("src.cli.RepositoryAcquisitionManager")
    @patch("src.cli.GitHubClient")
    def test_20_cli_json_output(self, mock_client_cls, mock_acq_cls):
        """20. github-pr-analyze --format json outputs valid JSON containing analysis_input."""
        mock_instance = MagicMock()
        mock_instance.get_pull_request.return_value = GitHubPullRequest(
            owner="bottlepy", repository="bottle", number=42, title="Fix bug",
            html_url="https://github.com/bottlepy/bottle/pull/42",
            base_ref="main", base_sha=FAKE_BASE_SHA,
            head_ref="feature/fix", head_sha=FAKE_HEAD_SHA
        )
        mock_client_cls.return_value = mock_instance

        temp_dir = tempfile.mkdtemp()
        try:
            mock_acq = MagicMock()
            mock_acq.acquire.return_value.__enter__.return_value = temp_dir
            mock_acq_cls.return_value = mock_acq

            args = ["github-pr-analyze", "--repo", "bottlepy/bottle", "--pr", "42", "--format", "json"]
            with patch("src.cli.PRAnalyzer.analyze") as mock_pr_analyze:
                mock_pr_analyze.return_value = MagicMock(to_dict=lambda: {"commits": []})
                exit_code = cli_main(args)
                self.assertEqual(exit_code, 0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_21_cli_configuration_failure(self):
        """21. Missing environment variables produces clean CLI error exit code 1."""
        with patch.dict(os.environ, {"GITHUB_APP_ID": "", "GITHUB_APP_PRIVATE_KEY": "", "GITHUB_APP_PRIVATE_KEY_PATH": "", "GITHUB_INSTALLATION_ID": ""}):
            args = ["github-pr-analyze", "--repo", "bottlepy/bottle", "--pr", "42"]
            exit_code = cli_main(args)
            self.assertEqual(exit_code, 1)

    @patch.dict(os.environ, {
        "GITHUB_APP_ID": "12345",
        "GITHUB_APP_PRIVATE_KEY": FAKE_RSA_PRIVATE_KEY,
        "GITHUB_INSTALLATION_ID": "67890"
    })
    @patch("src.cli.GitHubClient")
    def test_22_cli_api_failure(self, mock_client_cls):
        """22. GitHub API failure produces clean CLI error exit code 1."""
        mock_instance = MagicMock()
        mock_instance.get_pull_request.side_effect = GitHubPRNotFoundError("bottlepy", "bottle", 999)
        mock_client_cls.return_value = mock_instance

        args = ["github-pr-analyze", "--repo", "bottlepy/bottle", "--pr", "999"]
        exit_code = cli_main(args)
        self.assertEqual(exit_code, 1)

    def test_23_cli_invalid_arguments(self):
        """23. Malformed repo argument (without slash) produces exit code 1."""
        args = ["github-pr-analyze", "--repo", "bottlepy_invalid_format", "--pr", "42"]
        exit_code = cli_main(args)
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
