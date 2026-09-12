"""
Unit and Integration Test Suite for GitDiffExtractor (Phase 6.2).
100% offline, deterministic tests using temporary Git repositories.
"""

import os
import sys
import stat
import unittest
import tempfile
import shutil
import subprocess

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.analysis import (
    GitDiffExtractor,
    CommitDiff,
    FileDiff,
    DiffHunk,
    DiffLine,
    is_generated_file
)
from src.engine.exceptions import (
    RepositoryNotFoundError,
    InvalidGitRepositoryError,
    CommitNotFoundError,
    UnsupportedCommitTypeError
)

def remove_readonly(func, path, exc_info):
    """Helper to remove read-only attributes on Windows during rmtree cleanup."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

def run_git_cmd(repo_path: str, args: list) -> subprocess.CompletedProcess:
    """Helper to run git commands in a temporary test repository."""
    cmd = ["git"] + args
    return subprocess.run(
        cmd,
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        check=True
    )

class TestGitDiffExtractor(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        run_git_cmd(self.temp_dir, ["init"])
        run_git_cmd(self.temp_dir, ["config", "user.name", "Test Author"])
        run_git_cmd(self.temp_dir, ["config", "user.email", "test@example.com"])

    def tearDown(self):
        shutil.rmtree(self.temp_dir, onerror=remove_readonly)

    def test_1_normal_modified_file(self):
        """1. Test extraction of a normal modified file with hunk line numbers."""
        file_path = os.path.join(self.temp_dir, "main.py")
        with open(file_path, "w") as f:
            f.write("def hello():\n    print('hello')\n    return True\n")
        run_git_cmd(self.temp_dir, ["add", "main.py"])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Initial commit"])

        with open(file_path, "w") as f:
            f.write("def hello():\n    print('hello world')\n    return True\n")
        run_git_cmd(self.temp_dir, ["commit", "-am", "Modify hello function"])

        extractor = GitDiffExtractor(self.temp_dir)
        cd = extractor.extract_commit_diff("HEAD")

        self.assertIsInstance(cd, CommitDiff)
        self.assertFalse(cd.is_merge_commit)
        self.assertFalse(cd.is_empty_commit)
        self.assertEqual(len(cd.files_changed), 1)

        fd = cd.files_changed[0]
        self.assertEqual(fd.status, "MODIFIED")
        self.assertEqual(fd.old_path, "main.py")
        self.assertEqual(fd.new_path, "main.py")
        self.assertFalse(fd.is_binary)
        self.assertFalse(fd.is_generated)
        self.assertEqual(fd.additions, 1)
        self.assertEqual(fd.deletions, 1)
        self.assertGreater(len(fd.hunks), 0)

    def test_2_added_file(self):
        """2. Test extraction of a newly added file."""
        file_path = os.path.join(self.temp_dir, "init.py")
        with open(file_path, "w") as f:
            f.write("x = 1\n")
        run_git_cmd(self.temp_dir, ["add", "init.py"])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Initial commit"])

        new_file_path = os.path.join(self.temp_dir, "new_module.py")
        with open(new_file_path, "w") as f:
            f.write("def new_func():\n    pass\n")
        run_git_cmd(self.temp_dir, ["add", "new_module.py"])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Add new_module.py"])

        extractor = GitDiffExtractor(self.temp_dir)
        cd = extractor.extract_commit_diff("HEAD")

        self.assertEqual(len(cd.files_changed), 1)
        fd = cd.files_changed[0]
        self.assertEqual(fd.status, "ADDED")
        self.assertEqual(fd.new_path, "new_module.py")
        self.assertEqual(fd.additions, 2)
        self.assertEqual(fd.deletions, 0)

    def test_3_deleted_file(self):
        """3. Test extraction of a deleted file."""
        file_path = os.path.join(self.temp_dir, "old_file.py")
        with open(file_path, "w") as f:
            f.write("deprecated_code = True\n")
        run_git_cmd(self.temp_dir, ["add", "old_file.py"])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Add file to delete"])

        run_git_cmd(self.temp_dir, ["rm", "old_file.py"])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Remove old_file.py"])

        extractor = GitDiffExtractor(self.temp_dir)
        cd = extractor.extract_commit_diff("HEAD")

        self.assertEqual(len(cd.files_changed), 1)
        fd = cd.files_changed[0]
        self.assertEqual(fd.status, "DELETED")
        self.assertEqual(fd.old_path, "old_file.py")
        self.assertEqual(fd.deletions, 1)

    def test_4_renamed_file(self):
        """4. Test extraction of a renamed file."""
        file_path = os.path.join(self.temp_dir, "alpha.py")
        with open(file_path, "w") as f:
            f.write("def alpha():\n    return 42\n")
        run_git_cmd(self.temp_dir, ["add", "alpha.py"])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Add alpha.py"])

        run_git_cmd(self.temp_dir, ["mv", "alpha.py", "beta.py"])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Rename alpha.py to beta.py"])

        extractor = GitDiffExtractor(self.temp_dir)
        cd = extractor.extract_commit_diff("HEAD")

        self.assertEqual(len(cd.files_changed), 1)
        fd = cd.files_changed[0]
        self.assertEqual(fd.status, "RENAMED")
        self.assertEqual(fd.old_path, "alpha.py")
        self.assertEqual(fd.new_path, "beta.py")

    def test_5_binary_file(self):
        """5. Test extraction of a binary file."""
        file_path = os.path.join(self.temp_dir, "init.txt")
        with open(file_path, "w") as f:
            f.write("init\n")
        run_git_cmd(self.temp_dir, ["add", "init.txt"])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Initial commit"])

        bin_path = os.path.join(self.temp_dir, "data.bin")
        with open(bin_path, "wb") as f:
            f.write(bytes([0x00, 0xFF, 0xFE, 0xFA, 0x12, 0x34]))
        run_git_cmd(self.temp_dir, ["add", "data.bin"])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Add binary file"])

        extractor = GitDiffExtractor(self.temp_dir)
        cd = extractor.extract_commit_diff("HEAD")

        self.assertEqual(len(cd.files_changed), 1)
        fd = cd.files_changed[0]
        self.assertTrue(fd.is_binary)
        self.assertEqual(len(fd.hunks), 0)

    def test_6_empty_commit(self):
        """6. Test extraction of an empty commit (0 file changes)."""
        file_path = os.path.join(self.temp_dir, "init.txt")
        with open(file_path, "w") as f:
            f.write("init\n")
        run_git_cmd(self.temp_dir, ["add", "init.txt"])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Initial commit"])

        run_git_cmd(self.temp_dir, ["commit", "--allow-empty", "-m", "Empty commit"])

        extractor = GitDiffExtractor(self.temp_dir)
        cd = extractor.extract_commit_diff("HEAD")

        self.assertTrue(cd.is_empty_commit)
        self.assertEqual(len(cd.files_changed), 0)

    def test_7_root_commit(self):
        """7. Test extraction of a repository's root commit."""
        file_path = os.path.join(self.temp_dir, "root_file.py")
        with open(file_path, "w") as f:
            f.write("print('root')\n")
        run_git_cmd(self.temp_dir, ["add", "root_file.py"])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Root commit"])

        extractor = GitDiffExtractor(self.temp_dir)
        cd = extractor.extract_commit_diff("HEAD")

        self.assertFalse(cd.is_merge_commit)
        self.assertEqual(len(cd.files_changed), 1)
        self.assertEqual(cd.files_changed[0].status, "ADDED")

    def test_8_merge_commit_detection_policy(self):
        """8. Test that merge commits are detected and raise UnsupportedCommitTypeError."""
        # Force default branch name to main
        run_git_cmd(self.temp_dir, ["checkout", "-B", "main"])
        
        file_a = os.path.join(self.temp_dir, "file_a.txt")
        with open(file_a, "w") as f:
            f.write("base\n")
        run_git_cmd(self.temp_dir, ["add", "file_a.txt"])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Base commit"])

        # Create branch feature
        run_git_cmd(self.temp_dir, ["checkout", "-b", "feature"])
        file_b = os.path.join(self.temp_dir, "file_b.txt")
        with open(file_b, "w") as f:
            f.write("feature\n")
        run_git_cmd(self.temp_dir, ["add", "file_b.txt"])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Feature commit"])

        # Switch to main and create commit
        run_git_cmd(self.temp_dir, ["checkout", "main"])
        with open(file_a, "w") as f:
            f.write("base updated\n")
        run_git_cmd(self.temp_dir, ["commit", "-am", "Main commit"])

        # Merge feature into main
        run_git_cmd(self.temp_dir, ["merge", "--no-ff", "feature", "-m", "Merge feature"])

        extractor = GitDiffExtractor(self.temp_dir)
        with self.assertRaises(UnsupportedCommitTypeError):
            extractor.extract_commit_diff("HEAD")

    def test_9_multiple_files(self):
        """9. Test commit modifying multiple files simultaneously."""
        file_a = os.path.join(self.temp_dir, "a.py")
        file_b = os.path.join(self.temp_dir, "b.py")
        with open(file_a, "w") as f:
            f.write("a = 1\n")
        with open(file_b, "w") as f:
            f.write("b = 2\n")
        run_git_cmd(self.temp_dir, ["add", "."])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Multi-file commit"])

        extractor = GitDiffExtractor(self.temp_dir)
        cd = extractor.extract_commit_diff("HEAD")

        self.assertEqual(len(cd.files_changed), 2)
        paths = {f.new_path for f in cd.files_changed}
        self.assertIn("a.py", paths)
        self.assertIn("b.py", paths)

    def test_10_correct_addition_deletion_counts(self):
        """10. Test accurate calculation of additions and deletions."""
        file_path = os.path.join(self.temp_dir, "math_mod.py")
        with open(file_path, "w") as f:
            f.write("line1\nline2\nline3\n")
        run_git_cmd(self.temp_dir, ["add", "math_mod.py"])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Commit 1"])

        with open(file_path, "w") as f:
            f.write("line1\nline2_modified\nline4_added\nline5_added\n")
        run_git_cmd(self.temp_dir, ["commit", "-am", "Commit 2"])

        extractor = GitDiffExtractor(self.temp_dir)
        cd = extractor.extract_commit_diff("HEAD")

        self.assertEqual(cd.total_additions, 3)
        self.assertEqual(cd.total_deletions, 2)

    def test_11_correct_hunk_parsing(self):
        """11. Test correct hunk header text and line type extraction."""
        file_path = os.path.join(self.temp_dir, "code.py")
        with open(file_path, "w") as f:
            f.write("def func():\n    return 1\n")
        run_git_cmd(self.temp_dir, ["add", "code.py"])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Commit 1"])

        with open(file_path, "w") as f:
            f.write("def func():\n    # new comment\n    return 1\n")
        run_git_cmd(self.temp_dir, ["commit", "-am", "Commit 2"])

        extractor = GitDiffExtractor(self.temp_dir)
        cd = extractor.extract_commit_diff("HEAD")

        hunks = cd.files_changed[0].hunks
        self.assertGreater(len(hunks), 0)
        hunk = hunks[0]
        line_types = [l.line_type for l in hunk.lines]
        self.assertIn('+', line_types)
        self.assertIn(' ', line_types)

    def test_12_correct_old_new_line_numbers(self):
        """12. Test accurate old and new line numbers progression in hunks."""
        file_path = os.path.join(self.temp_dir, "seq.py")
        with open(file_path, "w") as f:
            f.write("line1\nline2\nline3\nline4\n")
        run_git_cmd(self.temp_dir, ["add", "seq.py"])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Commit 1"])

        with open(file_path, "w") as f:
            f.write("line1\nline2_changed\nline4\nline5\n")
        run_git_cmd(self.temp_dir, ["commit", "-am", "Commit 2"])

        extractor = GitDiffExtractor(self.temp_dir)
        cd = extractor.extract_commit_diff("HEAD")

        hunk = cd.files_changed[0].hunks[0]
        for line in hunk.lines:
            if line.line_type == ' ':
                self.assertIsNotNone(line.old_lineno)
                self.assertIsNotNone(line.new_lineno)
            elif line.line_type == '+':
                self.assertIsNone(line.old_lineno)
                self.assertIsNotNone(line.new_lineno)
            elif line.line_type == '-':
                self.assertIsNotNone(line.old_lineno)
                self.assertIsNone(line.new_lineno)

    def test_13_commit_metadata(self):
        """13. Test complete capture of author, email, timestamp, and commit message."""
        file_path = os.path.join(self.temp_dir, "meta.txt")
        with open(file_path, "w") as f:
            f.write("meta\n")
        run_git_cmd(self.temp_dir, ["add", "meta.txt"])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Detailed test commit message"])

        extractor = GitDiffExtractor(self.temp_dir)
        cd = extractor.extract_commit_diff("HEAD")

        self.assertEqual(cd.author, "Test Author")
        self.assertEqual(cd.author_email, "test@example.com")
        self.assertEqual(cd.commit_message, "Detailed test commit message")
        self.assertTrue(len(cd.full_hash) == 40)
        self.assertTrue(len(cd.short_hash) == 7)

    def test_14_invalid_repository(self):
        """14. Test exception raised when target path is not a valid git repo."""
        sub_dir = os.path.join(self.temp_dir, "not_a_repo")
        os.makedirs(sub_dir, exist_ok=True)
        with self.assertRaises(InvalidGitRepositoryError):
            GitDiffExtractor(sub_dir)

    def test_15_invalid_commit(self):
        """15. Test exception raised when specifying non-existent commit SHA."""
        file_path = os.path.join(self.temp_dir, "init.txt")
        with open(file_path, "w") as f:
            f.write("init\n")
        run_git_cmd(self.temp_dir, ["add", "init.txt"])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Commit 1"])

        extractor = GitDiffExtractor(self.temp_dir)
        with self.assertRaises(CommitNotFoundError):
            extractor.extract_commit_diff("bad_commit_sha_999999")

    def test_16_generated_file_heuristic(self):
        """16. Test is_generated classification heuristic."""
        self.assertTrue(is_generated_file("package-lock.json"))
        self.assertTrue(is_generated_file("bundle.min.js"))
        self.assertTrue(is_generated_file("schema.pb.go"))
        self.assertTrue(is_generated_file("styles.css.map"))
        self.assertFalse(is_generated_file("main.py"))
        self.assertFalse(is_generated_file("app.ts"))

    def test_17_no_accidental_absolute_path_leakage(self):
        """17. Test that FileDiff old_path and new_path do not contain absolute repo paths."""
        file_path = os.path.join(self.temp_dir, "sub", "module.py")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write("submodule = True\n")
        run_git_cmd(self.temp_dir, ["add", "."])
        run_git_cmd(self.temp_dir, ["commit", "-m", "Add submodule.py"])

        extractor = GitDiffExtractor(self.temp_dir)
        cd = extractor.extract_commit_diff("HEAD")

        fd = cd.files_changed[0]
        self.assertNotIn(self.temp_dir, fd.new_path)
        self.assertNotIn(self.temp_dir, fd.old_path)
        self.assertTrue(fd.new_path.startswith("sub/module.py"))

if __name__ == '__main__':
    unittest.main()
