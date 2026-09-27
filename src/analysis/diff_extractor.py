"""
Git diff extraction module for PatchGuard Phase 6 qualitative code analysis.
Extracts structured commit diffs (CommitDiff, FileDiff, DiffHunk, DiffLine) with
accurate line numbering, rename tracking, status classification, and metadata.
"""

import os
import re
import subprocess
import shutil
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

from src.engine.exceptions import (
    RepositoryNotFoundError,
    InvalidGitRepositoryError,
    CommitNotFoundError,
    UnsupportedCommitTypeError
)

GENERATED_FILE_PATTERNS = [
    r'package-lock\.json$', r'yarn\.lock$', r'pnpm-lock\.yaml$', r'cargo\.lock$', r'Gemfile\.lock$',
    r'\.min\.js$', r'\.min\.css$', r'\.map$', r'\.generated\.', r'\.pb\.go$', r'swagger\.json$'
]

def is_generated_file(filepath: str) -> bool:
    """Deterministic heuristic for identifying generated files and lockfiles."""
    clean_path = filepath.strip().replace('\\', '/')
    for pattern in GENERATED_FILE_PATTERNS:
        if re.search(pattern, clean_path, re.IGNORECASE):
            return True
    return False

@dataclass(frozen=True)
class DiffLine:
    line_type: str  # '+', '-', ' '
    old_lineno: Optional[int]
    new_lineno: Optional[int]
    content: str

@dataclass(frozen=True)
class DiffHunk:
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    header: str
    lines: List[DiffLine]

@dataclass(frozen=True)
class FileDiff:
    old_path: str
    new_path: str
    status: str  # "ADDED", "DELETED", "MODIFIED", "RENAMED", "COPIED"
    is_binary: bool
    is_generated: bool
    additions: int
    deletions: int
    hunks: List[DiffHunk]

@dataclass(frozen=True)
class CommitDiff:
    full_hash: str
    short_hash: str
    author: str
    author_email: str
    timestamp: str
    commit_message: str
    files_changed: List[FileDiff]
    total_additions: int
    total_deletions: int
    is_merge_commit: bool
    is_empty_commit: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serializes CommitDiff to standard dictionary representation."""
        return asdict(self)

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


class GitDiffExtractor:
    """
    Extracts complete, structured commit diff representations from a local Git repository.
    Strictly inference-only, repository-agnostic, deterministic, and safe.
    """
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        if not os.path.exists(self.repo_path):
            raise RepositoryNotFoundError(f"Repository directory does not exist: {self.repo_path}")
            
        res = self._run_git(["rev-parse", "--is-inside-work-tree"])
        if res.returncode != 0:
            raise InvalidGitRepositoryError(f"Directory is not a valid Git repository: {self.repo_path}")

        top_res = self._run_git(["rev-parse", "--show-toplevel"])
        if top_res.returncode != 0:
            raise InvalidGitRepositoryError(f"Directory is not a valid Git repository: {self.repo_path}")
            
        top_level = os.path.abspath(top_res.stdout.strip())
        if top_level != self.repo_path and not os.path.exists(os.path.join(self.repo_path, ".git")):
            raise InvalidGitRepositoryError(f"Directory is not the root of a Git repository: {self.repo_path}")

    def _run_git(self, args: List[str]) -> subprocess.CompletedProcess:
        """Executes git command safely using subprocess."""
        git_executable = _find_git_executable()
        cmd = [git_executable] + args
        return subprocess.run(
            cmd,
            cwd=self.repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

    def _parse_filepath(self, file_path_str: str) -> str:
        """Parses filepath from git diff/numstat output, handling rename braces."""
        if '=>' in file_path_str:
            match = re.search(r'(.*?)(?:\{.*?=>\s*(.*?)\}|(\S+)\s*=>\s*(\S+))(.*)', file_path_str)
            if match:
                prefix = match.group(1) or ""
                new_part = match.group(2) or match.group(4) or ""
                suffix = match.group(5) or ""
                return (prefix + new_part + suffix).replace('//', '/').strip()
        return file_path_str.strip()

    def extract_commit_diff(self, commit_hash: str = "HEAD") -> CommitDiff:
        """
        Extracts a structured CommitDiff object for a specific commit hash, ref, or branch.
        """
        # Resolve full 40-character SHA
        res = self._run_git(["rev-parse", "--verify", f"{commit_hash}^{{commit}}"])
        if res.returncode != 0:
            raise CommitNotFoundError(f"Commit not found or invalid: {commit_hash}")
        full_hash = res.stdout.strip()
        short_hash = full_hash[:7]

        # Extract commit metadata
        log_res = self._run_git(["log", "-1", "--format=%H|%an|%ae|%aI|%P%n%B", full_hash])
        if log_res.returncode != 0 or not log_res.stdout:
            raise CommitNotFoundError(f"Failed to retrieve log details for commit: {full_hash}")

        lines = log_res.stdout.splitlines()
        header_line = lines[0] if lines else ""
        commit_message = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

        header_parts = header_line.split('|')
        author = header_parts[1] if len(header_parts) > 1 else "Unknown"
        author_email = header_parts[2] if len(header_parts) > 2 else ""
        timestamp = header_parts[3] if len(header_parts) > 3 else ""
        parents = header_parts[4].split() if len(header_parts) > 4 and header_parts[4] else []

        is_merge = len(parents) > 1
        if is_merge:
            raise UnsupportedCommitTypeError(
                f"Commit {short_hash} is a merge commit with {len(parents)} parents. "
                "Merge commits are unsupported for risk analysis."
            )

        parent = parents[0] if parents else None

        # Numstat mapping for exact additions, deletions, and binary flags
        numstat_args = ["diff-tree", "--numstat", "-M"]
        if parent:
            numstat_args.extend([parent, full_hash])
        else:
            numstat_args.extend(["--root", full_hash])

        numstat_res = self._run_git(numstat_args)
        numstat_map = {}
        for num_line in numstat_res.stdout.splitlines():
            num_line = num_line.strip()
            if not num_line:
                continue
            num_parts = num_line.split('\t')
            if len(num_parts) >= 3:
                add_s, del_s, path_s = num_parts[0], num_parts[1], num_parts[2]
                is_bin = (add_s == '-' or del_s == '-')
                adds = int(add_s) if not is_bin else 0
                dels = int(del_s) if not is_bin else 0
                numstat_map[path_s] = (adds, dels, is_bin)

        # Extraction of unified patch text
        diff_args = ["diff-tree", "-p", "-M", "-U3"]
        if parent:
            diff_args.extend([parent, full_hash])
        else:
            diff_args.extend(["--root", full_hash])

        diff_res = self._run_git(diff_args)
        raw_diff_output = diff_res.stdout

        # Parse file diffs from raw diff output
        file_diffs, total_adds, total_dels = self._parse_unified_diff(raw_diff_output, numstat_map)

        is_empty = len(file_diffs) == 0

        return CommitDiff(
            full_hash=full_hash,
            short_hash=short_hash,
            author=author,
            author_email=author_email,
            timestamp=timestamp,
            commit_message=commit_message,
            files_changed=file_diffs,
            total_additions=total_adds,
            total_deletions=total_dels,
            is_merge_commit=is_merge,
            is_empty_commit=is_empty
        )

    def extract_cumulative_diff(self, base_ref: str, head_ref: str) -> CommitDiff:
        """
        Extracts a structured CommitDiff object representing the cumulative code change
        between base_ref and head_ref relative to their merge-base.
        """
        # Resolve base reference to full 40-character SHA
        res_base = self._run_git(["rev-parse", "--verify", f"{base_ref}^{{commit}}"])
        if res_base.returncode != 0:
            raise CommitNotFoundError(f"Base reference not found or invalid: {base_ref}")
        base_sha = res_base.stdout.strip()

        # Resolve head reference to full 40-character SHA
        res_head = self._run_git(["rev-parse", "--verify", f"{head_ref}^{{commit}}"])
        if res_head.returncode != 0:
            raise CommitNotFoundError(f"Head reference not found or invalid: {head_ref}")
        head_sha = res_head.stdout.strip()

        # Determine merge base
        mb_res = self._run_git(["merge-base", base_sha, head_sha])
        if mb_res.returncode != 0 or not mb_res.stdout.strip():
            raise CommitNotFoundError(f"No common merge-base found between base '{base_ref}' and head '{head_ref}'")
        merge_base_sha = mb_res.stdout.strip()

        # Retrieve head commit metadata for authorship/timestamp
        log_res = self._run_git(["log", "-1", "--format=%H|%an|%ae|%aI%n%B", head_sha])
        lines = log_res.stdout.splitlines() if log_res.returncode == 0 and log_res.stdout else []
        header_line = lines[0] if lines else ""
        head_commit_msg = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

        header_parts = header_line.split('|')
        author = header_parts[1] if len(header_parts) > 1 else "Unknown"
        author_email = header_parts[2] if len(header_parts) > 2 else ""
        timestamp = header_parts[3] if len(header_parts) > 3 else ""

        pr_commit_msg = f"Cumulative PR Diff ({base_ref}..{head_ref})\n\nHead Commit ({head_sha[:7]}): {head_commit_msg}"

        if merge_base_sha == head_sha or base_sha == head_sha:
            return CommitDiff(
                full_hash=head_sha,
                short_hash=head_sha[:7],
                author=author,
                author_email=author_email,
                timestamp=timestamp,
                commit_message=pr_commit_msg,
                files_changed=[],
                total_additions=0,
                total_deletions=0,
                is_merge_commit=False,
                is_empty_commit=True
            )

        # Run numstat for cumulative diff from merge_base_sha to head_sha
        numstat_res = self._run_git(["diff", "--numstat", "-M", merge_base_sha, head_sha])
        numstat_map = {}
        for num_line in numstat_res.stdout.splitlines():
            num_line = num_line.strip()
            if not num_line:
                continue
            num_parts = num_line.split('\t')
            if len(num_parts) >= 3:
                add_s, del_s, path_s = num_parts[0], num_parts[1], num_parts[2]
                is_bin = (add_s == '-' or del_s == '-')
                adds = int(add_s) if not is_bin else 0
                dels = int(del_s) if not is_bin else 0
                numstat_map[path_s] = (adds, dels, is_bin)

        # Run diff patch for cumulative diff from merge_base_sha to head_sha
        diff_res = self._run_git(["diff", "-p", "-M", "-U3", merge_base_sha, head_sha])
        raw_diff_output = diff_res.stdout

        file_diffs, total_adds, total_dels = self._parse_unified_diff(raw_diff_output, numstat_map)
        is_empty = len(file_diffs) == 0

        return CommitDiff(
            full_hash=head_sha,
            short_hash=head_sha[:7],
            author=author,
            author_email=author_email,
            timestamp=timestamp,
            commit_message=pr_commit_msg,
            files_changed=file_diffs,
            total_additions=total_adds,
            total_deletions=total_dels,
            is_merge_commit=False,
            is_empty_commit=is_empty
        )


    def _parse_unified_diff(
        self,
        diff_text: str,
        numstat_map: Dict[str, Tuple[int, int, bool]]
    ) -> Tuple[List[FileDiff], int, int]:
        """Parses unified diff output into structured FileDiff objects."""
        if not diff_text.strip():
            return [], 0, 0

        # Split diff output into per-file chunks
        raw_blocks = re.split(r'(^diff --git )', diff_text, flags=re.MULTILINE)
        
        file_diffs = []
        total_additions = 0
        total_deletions = 0

        i = 1
        while i < len(raw_blocks):
            if raw_blocks[i].startswith('diff --git '):
                block_content = raw_blocks[i] + (raw_blocks[i+1] if i+1 < len(raw_blocks) else "")
                i += 2
            else:
                block_content = raw_blocks[i]
                i += 1

            lines = block_content.splitlines()
            if not lines:
                continue

            git_header = lines[0]
            header_match = re.match(r'diff --git a/(.*?) b/(.*)', git_header)
            if header_match:
                raw_old_path = header_match.group(1)
                raw_new_path = header_match.group(2)
            else:
                raw_old_path = ""
                raw_new_path = ""

            old_path = raw_old_path
            new_path = raw_new_path
            status = "MODIFIED"
            is_binary = False

            for line in lines[1:10]:
                if line.startswith("new file mode"):
                    status = "ADDED"
                elif line.startswith("deleted file mode"):
                    status = "DELETED"
                elif line.startswith("rename from"):
                    status = "RENAMED"
                    old_path = line[len("rename from "):].strip()
                elif line.startswith("rename to"):
                    status = "RENAMED"
                    new_path = line[len("rename to "):].strip()
                elif line.startswith("copy from"):
                    status = "COPIED"
                    old_path = line[len("copy from "):].strip()
                elif line.startswith("copy to"):
                    status = "COPIED"
                    new_path = line[len("copy to "):].strip()
                elif "Binary files" in line or line.startswith("GIT binary patch"):
                    is_binary = True

            additions = 0
            deletions = 0

            # Match with numstat data if available
            matched_numstat = None
            for num_path, num_val in numstat_map.items():
                parsed_num_path = self._parse_filepath(num_path)
                if parsed_num_path in (new_path, old_path) or num_path in (new_path, old_path):
                    matched_numstat = num_val
                    break

            if matched_numstat:
                additions, deletions, numstat_bin = matched_numstat
                if numstat_bin:
                    is_binary = True

            hunks = []
            if not is_binary:
                hunks = self._parse_hunks(lines)
                if not matched_numstat:
                    for h in hunks:
                        for l in h.lines:
                            if l.line_type == '+':
                                additions += 1
                            elif l.line_type == '-':
                                deletions += 1

            is_gen = is_generated_file(new_path if new_path else old_path)

            file_diff = FileDiff(
                old_path=old_path,
                new_path=new_path,
                status=status,
                is_binary=is_binary,
                is_generated=is_gen,
                additions=additions,
                deletions=deletions,
                hunks=hunks
            )
            file_diffs.append(file_diff)
            total_additions += additions
            total_deletions += deletions

        return file_diffs, total_additions, total_deletions

    def _parse_hunks(self, lines: List[str]) -> List[DiffHunk]:
        """Parses unified diff hunk blocks and tracks old/new line numbers."""
        hunks = []
        current_old_start = 0
        current_old_lines = 0
        current_new_start = 0
        current_new_lines = 0
        current_header_text = ""

        current_old_lineno = 0
        current_new_lineno = 0
        current_hunk_lines = []

        hunk_regex = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)')

        for line in lines:
            match = hunk_regex.match(line)
            if match:
                if current_hunk_lines or current_old_start > 0 or current_new_start > 0:
                    hunks.append(DiffHunk(
                        old_start=current_old_start,
                        old_lines=current_old_lines,
                        new_start=current_new_start,
                        new_lines=current_new_lines,
                        header=current_header_text,
                        lines=current_hunk_lines
                    ))

                current_old_start = int(match.group(1))
                current_old_lines = int(match.group(2)) if match.group(2) is not None else 1
                current_new_start = int(match.group(3))
                current_new_lines = int(match.group(4)) if match.group(4) is not None else 1
                current_header_text = match.group(5).strip()

                current_old_lineno = current_old_start
                current_new_lineno = current_new_start
                current_hunk_lines = []
                continue

            if current_old_start > 0 or current_new_start > 0:
                if line.startswith('+') and not line.startswith('+++'):
                    current_hunk_lines.append(DiffLine(
                        line_type='+',
                        old_lineno=None,
                        new_lineno=current_new_lineno,
                        content=line[1:]
                    ))
                    current_new_lineno += 1
                elif line.startswith('-') and not line.startswith('---'):
                    current_hunk_lines.append(DiffLine(
                        line_type='-',
                        old_lineno=current_old_lineno,
                        new_lineno=None,
                        content=line[1:]
                    ))
                    current_old_lineno += 1
                elif line.startswith(' ') or (line == '' and not line.startswith('index ') and not line.startswith('diff ')):
                    content = line[1:] if line.startswith(' ') else line
                    current_hunk_lines.append(DiffLine(
                        line_type=' ',
                        old_lineno=current_old_lineno,
                        new_lineno=current_new_lineno,
                        content=content
                    ))
                    current_old_lineno += 1
                    current_new_lineno += 1

        if current_hunk_lines or current_old_start > 0 or current_new_start > 0:
            hunks.append(DiffHunk(
                old_start=current_old_start,
                old_lines=current_old_lines,
                new_start=current_new_start,
                new_lines=current_new_lines,
                header=current_header_text,
                lines=current_hunk_lines
            ))

        return hunks
