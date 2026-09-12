"""
Unit and Integration Test Suite for PatchGuard CLI (src/cli.py).
Tests 'predict' and 'analyze' subcommands 100% offline.
"""

import io
import json
import os
import shutil
import tempfile
import unittest
import subprocess
from unittest.mock import patch, MagicMock

from src.cli import main

class TestPatchGuardCLI(unittest.TestCase):

    def setUp(self):
        self.repo_dir = tempfile.mkdtemp()
        subprocess.run(["git", "init"], cwd=self.repo_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "config", "user.name", "TestUser"], cwd=self.repo_dir, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.repo_dir, check=True)
        
        test_file = os.path.join(self.repo_dir, "auth.py")
        with open(test_file, "w") as f:
            f.write("def login(user):\n    return True\n")
        
        subprocess.run(["git", "add", "."], cwd=self.repo_dir, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=self.repo_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def tearDown(self):
        def remove_readonly(func, path, exc_info):
            try:
                os.chmod(path, 0o777)
                func(path)
            except Exception:
                pass
        shutil.rmtree(self.repo_dir, onerror=remove_readonly)

    def test_1_no_args_returns_non_zero(self):
        with patch('sys.stdout', new=io.StringIO()), patch('sys.stderr', new=io.StringIO()):
            exit_code = main([])
            self.assertNotEqual(exit_code, 0)

    def test_2_predict_subcommand_text_output(self):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            exit_code = main(["predict", "--repo", self.repo_dir, "--commit", "HEAD", "--format", "text"])
            self.assertEqual(exit_code, 0)
            output = fake_out.getvalue()
            self.assertIn("PATCHGUARD CODE CHANGE RISK PREDICTION", output)
            self.assertIn("Estimated Risk Score", output)

    def test_3_predict_subcommand_json_output(self):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            exit_code = main(["predict", "--repo", self.repo_dir, "--commit", "HEAD", "--format", "json"])
            self.assertEqual(exit_code, 0)
            data = json.loads(fake_out.getvalue())
            self.assertIn("raw_probability", data)
            self.assertIn("is_above_threshold", data)

    def test_4_analyze_subcommand_default_mock_text(self):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            exit_code = main(["analyze", "--repo", self.repo_dir, "--commit", "HEAD", "--provider", "mock", "--format", "text"])
            self.assertEqual(exit_code, 0)
            output = fake_out.getvalue()
            self.assertIn("PATCHGUARD INTELLIGENT CODE CHANGE ANALYSIS", output)
            self.assertIn("ML DEFECT RISK PREDICTION", output)
            self.assertIn("QUALITATIVE CODE CHANGE ANALYSIS", output)
            self.assertIn("mock (deterministic-mock)", output)

    def test_5_analyze_subcommand_default_mock_json(self):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            exit_code = main(["analyze", "--repo", self.repo_dir, "--commit", "HEAD", "--provider", "mock", "--format", "json"])
            self.assertEqual(exit_code, 0)
            data = json.loads(fake_out.getvalue())
            self.assertIn("commit", data)
            self.assertIn("ml_prediction", data)
            self.assertIn("code_change_analysis", data)
            self.assertEqual(data["code_change_analysis"]["model_provider"], "mock")

    def test_6_analyze_invalid_repository_returns_error(self):
        with patch('sys.stderr', new=io.StringIO()) as fake_err:
            exit_code = main(["analyze", "--repo", "/non_existent_directory_xyz", "--commit", "HEAD"])
            self.assertNotEqual(exit_code, 0)
            err_msg = fake_err.getvalue()
            self.assertIn("[PatchGuard Error]", err_msg)

    def test_7_analyze_invalid_commit_returns_error(self):
        with patch('sys.stderr', new=io.StringIO()) as fake_err:
            exit_code = main(["analyze", "--repo", self.repo_dir, "--commit", "invalid_sha_123456789"])
            self.assertNotEqual(exit_code, 0)
            err_msg = fake_err.getvalue()
            self.assertIn("[PatchGuard Error]", err_msg)

    def test_8_analyze_openai_missing_api_key_returns_error(self):
        env_without_key = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            with patch('sys.stderr', new=io.StringIO()) as fake_err:
                exit_code = main(["analyze", "--repo", self.repo_dir, "--commit", "HEAD", "--provider", "openai"])
                self.assertNotEqual(exit_code, 0)
                err_msg = fake_err.getvalue()
                self.assertIn("[PatchGuard Error]", err_msg)
                self.assertIn("MissingAPIKeyError", err_msg)

    def test_9_analyze_openai_with_mocked_client(self):
        with patch("src.cli.OpenAIProvider") as mock_openai_cls:
            mock_provider_instance = MagicMock()
            mock_provider_instance.provider_name = "openai"
            mock_provider_instance.model_name = "gpt-4o"
            mock_provider_instance.generate_analysis.return_value = json.dumps({
                "commit_hash": "HEAD",
                "summary": "Mocked OpenAI summary",
                "key_changes": ["Added logic"],
                "potential_risk_factors": [],
                "affected_areas": ["auth"],
                "testing_observations": [],
                "recommended_review_actions": [],
                "confidence_notes": "Clean"
            })
            mock_openai_cls.return_value = mock_provider_instance

            with patch('sys.stdout', new=io.StringIO()) as fake_out:
                exit_code = main(["analyze", "--repo", self.repo_dir, "--commit", "HEAD", "--provider", "openai", "--model", "gpt-4o"])
                self.assertEqual(exit_code, 0)
                output = fake_out.getvalue()
                self.assertIn("Mocked OpenAI summary", output)


if __name__ == "__main__":
    unittest.main()
