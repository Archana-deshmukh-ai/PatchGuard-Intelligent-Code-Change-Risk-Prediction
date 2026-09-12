import os
import re
import subprocess
from typing import List, Dict, Any, Optional
import pandas as pd

# Standard source code file extensions
SOURCE_EXTENSIONS = {
    '.py', '.java', '.cpp', '.c', '.h', '.hpp', '.js', '.ts', '.jsx', '.tsx',
    '.go', '.rs', '.cs', '.rb', '.php', '.swift', '.kt', '.sh', '.ps1',
    '.html', '.css', '.scss', '.sql', '.scala', '.m', '.mm'
}

# Patterns to identify test files
TEST_PATTERNS = [
    r'test_', r'_test\.', r'/tests?/', r'/spec/', r'Test\.java$', r'Spec\.'
]

class GitRepositoryExtractor:
    """
    Extracts commit-level software engineering features from a local Git repository.
    Strictly uses commit-time information (no future-leakage).
    """
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        if not os.path.exists(self.repo_path):
            raise FileNotFoundError(f"Repository path does not exist: {self.repo_path}")
            
        # Validate that path is inside a git repository
        res = self._run_git(["rev-parse", "--is-inside-work-tree"])
        if res.returncode != 0:
            raise ValueError(f"Directory is not a valid Git repository: {self.repo_path}")

    def _run_git(self, args: List[str]) -> subprocess.CompletedProcess:
        """Executes a git command via subprocess inside the target repository."""
        cmd = ["git"] + args
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
        """
        Parses filepath from git numstat output, handling rename arrows.
        Example renames:
        - 'old_dir/{old_name => new_name}.py' -> 'old_dir/new_name.py'
        - 'old_name.py => new_name.py' -> 'new_name.py'
        """
        if '=>' in file_path_str:
            # Handle format: pre{old => new}post or old => new
            match = re.search(r'(.*?)(?:\{.*?=>\s*(.*?)\}|(\S+)\s*=>\s*(\S+))(.*)', file_path_str)
            if match:
                prefix = match.group(1) or ""
                new_part = match.group(2) or match.group(4) or ""
                suffix = match.group(5) or ""
                clean_path = (prefix + new_part + suffix).replace('//', '/')
                return clean_path.strip()
        return file_path_str.strip()

    def _estimate_functions_changed(self, parent: Optional[str], commit_hash: str) -> int:
        """
        Heuristically estimates the number of functions/methods modified in a commit
        by inspecting diff hunk headers (@@ ... @@) containing function signatures.
        """
        diff_args = ["diff-tree", "-U0", "-M"]
        if parent:
            diff_args.extend([parent, commit_hash])
        else:
            diff_args.extend(["--root", commit_hash])
            
        res = self._run_git(diff_args)
        if res.returncode != 0:
            return 0
            
        # Count hunk headers that have trailing text (e.g. @@ ... @@ def func_name():)
        hunk_headers = 0
        for line in res.stdout.splitlines():
            if line.startswith('@@'):
                # Extract text after closing @@
                parts = line.split('@@')
                if len(parts) >= 3 and parts[2].strip():
                    hunk_headers += 1
                else:
                    hunk_headers += 1  # count hunk even without signature context
        return hunk_headers

    def extract_commit_history(
        self,
        branch: str = "HEAD",
        max_commits: Optional[int] = None,
        include_merges: bool = False,
        first_parent_only: bool = False
    ) -> pd.DataFrame:
        """
        Extracts commit-level features in chronological order (oldest to newest).
        
        Args:
            branch: Git branch/ref to log (default: "HEAD").
            max_commits: Maximum commits to extract (default: None for all).
            include_merges: If False, skips merge commits (parents > 1).
            first_parent_only: If True, follows only the first parent commit.
            
        Returns:
            pd.DataFrame: DataFrame containing metadata and 10 features.
        """
        log_args = ["log", "--reverse", "--format=%H|%an|%aI|%P", branch]
        if first_parent_only:
            log_args.append("--first-parent")
        if max_commits:
            log_args.extend(["-n", str(max_commits)])
            
        res = self._run_git(log_args)
        if res.returncode != 0:
            raise RuntimeError(f"Error running git log: {res.stderr}")
            
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        
        records = []
        
        for line in lines:
            parts = line.split('|')
            if len(parts) < 3:
                continue
                
            commit_hash = parts[0]
            author = parts[1]
            commit_timestamp = parts[2]
            parents = parts[3].split() if len(parts) > 3 and parts[3] else []
            
            # Filter merge commits if requested
            if len(parents) > 1 and not include_merges:
                continue
                
            parent = parents[0] if parents else None
            
            # Run numstat with rename detection (-M)
            numstat_args = ["diff-tree", "--numstat", "-M"]
            if parent:
                numstat_args.extend([parent, commit_hash])
            else:
                numstat_args.extend(["--root", commit_hash])
                
            numstat_res = self._run_git(numstat_args)
            numstat_lines = [l.strip() for l in numstat_res.stdout.splitlines() if l.strip()]
            
            lines_added = 0
            lines_deleted = 0
            files_changed = 0
            directories_set = set()
            test_file_modified = 0
            max_single_file_churn = 0
            source_files_count = 0
            
            for num_line in numstat_lines:
                num_parts = num_line.split('\t')
                if len(num_parts) < 3:
                    continue
                    
                add_str, del_str, raw_path = num_parts[0], num_parts[1], num_parts[2]
                clean_path = self._parse_filepath(raw_path)
                files_changed += 1
                
                # Check line counts (handling binary files marked with '-')
                if add_str != '-' and del_str != '-':
                    added = int(add_str)
                    deleted = int(del_str)
                else:
                    added = 0
                    deleted = 0
                    
                file_churn = added + deleted
                lines_added += added
                lines_deleted += deleted
                
                if file_churn > max_single_file_churn:
                    max_single_file_churn = file_churn
                    
                # Directory tracking
                dirname = os.path.dirname(clean_path)
                if dirname:
                    directories_set.add(dirname)
                else:
                    directories_set.add('.')
                    
                # Test file detection
                for pattern in TEST_PATTERNS:
                    if re.search(pattern, clean_path, re.IGNORECASE):
                        test_file_modified = 1
                        break
                        
                # Source file detection
                ext = os.path.splitext(clean_path)[1].lower()
                if ext in SOURCE_EXTENSIONS:
                    source_files_count += 1
                    
            code_churn = lines_added + lines_deleted
            num_directories_touched = len(directories_set) if files_changed > 0 else 0
            avg_lines_per_file = round(code_churn / max(1, files_changed), 2)
            functions_changed = self._estimate_functions_changed(parent, commit_hash)
            
            record = {
                'commit_hash': commit_hash[:7],
                'full_hash': commit_hash,
                'commit_timestamp': commit_timestamp,
                'author': author,
                'lines_added': lines_added,
                'lines_deleted': lines_deleted,
                'code_churn': code_churn,
                'files_changed': files_changed,
                'functions_changed': functions_changed,
                'num_directories_touched': num_directories_touched,
                'is_test_file_modified': test_file_modified,
                'avg_lines_changed_per_file': avg_lines_per_file,
                'max_lines_changed_in_single_file': max_single_file_churn,
                'num_source_files_changed': source_files_count
            }
            records.append(record)
            
        df = pd.DataFrame(records)
        print(f"[GitExtractor] Extracted {len(df)} commits from: {self.repo_path}")
        return df

    def extract_single_commit(self, commit_hash: str = "HEAD") -> Dict[str, Any]:
        """
        Extracts commit-level features for a single specific commit.
        
        Args:
            commit_hash: Commit SHA, ref, or branch (default: "HEAD").
            
        Returns:
            Dict[str, Any]: Dictionary containing commit metadata and the 10 ML features.
        """
        res = self._run_git(["rev-parse", "--verify", f"{commit_hash}^{{commit}}"])
        if res.returncode != 0:
            raise ValueError(f"Commit not found or invalid: {commit_hash}")
        full_hash = res.stdout.strip()
        
        log_res = self._run_git(["log", "-1", "--format=%H|%an|%aI|%P", full_hash])
        if log_res.returncode != 0 or not log_res.stdout.strip():
            raise ValueError(f"Failed to retrieve log details for commit: {full_hash}")
            
        parts = log_res.stdout.strip().split('|')
        author = parts[1] if len(parts) > 1 else "Unknown"
        commit_timestamp = parts[2] if len(parts) > 2 else ""
        parents = parts[3].split() if len(parts) > 3 and parts[3] else []
        
        parent = parents[0] if parents else None
        
        numstat_args = ["diff-tree", "--numstat", "-M"]
        if parent:
            numstat_args.extend([parent, full_hash])
        else:
            numstat_args.extend(["--root", full_hash])
            
        numstat_res = self._run_git(numstat_args)
        numstat_lines = [l.strip() for l in numstat_res.stdout.splitlines() if l.strip()]
        
        lines_added = 0
        lines_deleted = 0
        files_changed = 0
        directories_set = set()
        test_file_modified = 0
        max_single_file_churn = 0
        source_files_count = 0
        
        for num_line in numstat_lines:
            num_parts = num_line.split('\t')
            if len(num_parts) < 3:
                continue
                
            add_str, del_str, raw_path = num_parts[0], num_parts[1], num_parts[2]
            clean_path = self._parse_filepath(raw_path)
            files_changed += 1
            
            if add_str != '-' and del_str != '-':
                added = int(add_str)
                deleted = int(del_str)
            else:
                added = 0
                deleted = 0
                
            file_churn = added + deleted
            lines_added += added
            lines_deleted += deleted
            
            if file_churn > max_single_file_churn:
                max_single_file_churn = file_churn
                
            dirname = os.path.dirname(clean_path)
            if dirname:
                directories_set.add(dirname)
            else:
                directories_set.add('.')
                
            for pattern in TEST_PATTERNS:
                if re.search(pattern, clean_path, re.IGNORECASE):
                    test_file_modified = 1
                    break
                    
            ext = os.path.splitext(clean_path)[1].lower()
            if ext in SOURCE_EXTENSIONS:
                source_files_count += 1
                
        code_churn = lines_added + lines_deleted
        num_directories_touched = len(directories_set) if files_changed > 0 else 0
        avg_lines_per_file = round(code_churn / max(1, files_changed), 2)
        functions_changed = self._estimate_functions_changed(parent, full_hash)
        
        return {
            'commit_hash': full_hash[:7],
            'full_hash': full_hash,
            'commit_timestamp': commit_timestamp,
            'author': author,
            'parents': parents,
            'lines_added': lines_added,
            'lines_deleted': lines_deleted,
            'code_churn': code_churn,
            'files_changed': files_changed,
            'functions_changed': functions_changed,
            'num_directories_touched': num_directories_touched,
            'is_test_file_modified': test_file_modified,
            'avg_lines_changed_per_file': avg_lines_per_file,
            'max_lines_changed_in_single_file': max_single_file_churn,
            'num_source_files_changed': source_files_count
        }


if __name__ == '__main__':
    # Test on the local PatchGuard repository itself
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    extractor = GitRepositoryExtractor(repo_root)
    df_extracted = extractor.extract_commit_history(include_merges=False)
    
    print("\n--- EXTRACTED GIT COMMITS SAMPLE (HEAD 5) ---")
    print(df_extracted[['commit_hash', 'author', 'lines_added', 'lines_deleted', 'code_churn', 'files_changed', 'num_directories_touched', 'is_test_file_modified']].head())
