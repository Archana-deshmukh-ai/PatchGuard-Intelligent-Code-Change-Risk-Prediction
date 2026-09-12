import os
import re
import subprocess
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd

# Patterns for candidate detection
ISSUE_ID_REGEX = re.compile(r'\b(GH-\d+|JIRA-\d+|#\d+|[Ii]ssue\s*#?\d+)\b', re.IGNORECASE)
FIX_KEYWORD_REGEX = re.compile(r'\b(fix|fixes|fixed|bug|bugs|issue|issues|resolve|resolves|resolved|defect|patch|error|crash|panic)\b', re.IGNORECASE)
REVERT_KEYWORD_REGEX = re.compile(r'^(Revert\s+|^\b(revert|reverted|reverting)\b)', re.IGNORECASE)
COSMETIC_KEYWORD_REGEX = re.compile(r'\b(typo|formatting|indentation|docs|style|readme|lint)\b', re.IGNORECASE)

class SZZDefectLabeler:
    """
    Implements the SZZ Algorithm to retrospectivesly identify bug-introducing commits
    from Git commit history with multi-signal confidence scoring and audit provenance.
    """
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        if not os.path.exists(self.repo_path):
            raise FileNotFoundError(f"Repository path does not exist: {self.repo_path}")
            
        res = self._run_git(["rev-parse", "--is-inside-work-tree"])
        if res.returncode != 0:
            raise ValueError(f"Directory is not a valid Git repository: {self.repo_path}")

    def _run_git(self, args: List[str]) -> subprocess.CompletedProcess:
        """Executes native git command via subprocess."""
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

    def detect_candidate_fix_commits(self, branch: str = "HEAD") -> List[Dict[str, Any]]:
        """
        Scans commit log for evidence of bug-fixing activity (keywords, issue IDs, reverts).
        Does NOT hard-code test-file exclusions or file count limits.
        """
        log_args = ["log", "--reverse", "--format=%H|%an|%aI|%s|%P", branch]
        res = self._run_git(log_args)
        if res.returncode != 0:
            raise RuntimeError(f"Error running git log: {res.stderr}")

        candidates = []
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]

        for line in lines:
            parts = line.split('|')
            if len(parts) < 4:
                continue

            commit_hash = parts[0]
            author = parts[1]
            commit_timestamp = parts[2]
            subject = parts[3]
            parents = parts[4].split() if len(parts) > 4 and parts[4] else []

            # 1. Evidence Signal Extractions
            has_fix_kw = bool(FIX_KEYWORD_REGEX.search(subject))
            has_issue_id = bool(ISSUE_ID_REGEX.search(subject))
            is_revert = bool(REVERT_KEYWORD_REGEX.search(subject))
            is_cosmetic = bool(COSMETIC_KEYWORD_REGEX.search(subject))

            issue_ids = ISSUE_ID_REGEX.findall(subject)
            issue_id_str = ",".join(list(set(issue_ids))) if issue_ids else ""

            # Check diff files
            parent = parents[0] if parents else None
            diff_args = ["diff-tree", "--numstat", "-M"]
            if parent:
                diff_args.extend([parent, commit_hash])
            else:
                diff_args.extend(["--root", commit_hash])

            numstat_res = self._run_git(diff_args)
            files_count = len([l for l in numstat_res.stdout.splitlines() if l.strip()])

            # Is candidate fix if explicit fix keyword or issue ID is present.
            # Pure reverts without defect keywords are tracked separately as revert evidence, NOT automatically as bug fixes.
            is_candidate = (has_fix_kw or has_issue_id) and not is_cosmetic

            candidates.append({
                "commit_hash": commit_hash[:7],
                "full_hash": commit_hash,
                "commit_timestamp": commit_timestamp,
                "author": author,
                "subject": subject,
                "parents": parents,
                "parent": parent,
                "has_fix_keyword": has_fix_kw,
                "has_issue_id": has_issue_id,
                "issue_id": issue_id_str,
                "is_revert": is_revert,
                "is_cosmetic": is_cosmetic,
                "files_changed_count": files_count,
                "is_candidate_fix": is_candidate
            })

        return candidates

    def trace_line_blame(self, parent_commit: str, file_path: str, start_line: int, end_line: int) -> List[str]:
        """
        Runs git blame -L start_line,end_line on parent state to find origin commit SHAs.
        """
        if start_line > end_line or start_line < 1:
            return []

        blame_args = ["blame", "-L", f"{start_line},{end_line}", "-M", "-C", "--porcelain", parent_commit, "--", file_path]
        res = self._run_git(blame_args)
        if res.returncode != 0:
            return []

        origin_hashes = []
        for line in res.stdout.splitlines():
            # In porcelain format, header line starts with 40-char SHA
            match = re.match(r'^([0-9a-f]{40})\s', line)
            if match:
                sha = match.group(1)
                if sha not in origin_hashes:
                    origin_hashes.append(sha)
        return origin_hashes

    def parse_deleted_line_hunks(self, parent_commit: str, commit_hash: str) -> List[Tuple[str, int, int]]:
        """
        Parses unified diff (git diff -U0) between parent and fixing commit to extract deleted line ranges.
        Returns list of tuples: (file_path, old_start_line, old_line_count)
        """
        diff_args = ["diff-tree", "-U0", "-M", parent_commit, commit_hash]
        res = self._run_git(diff_args)
        if res.returncode != 0:
            return []

        hunks = []
        current_file = None

        for line in res.stdout.splitlines():
            if line.startswith("--- a/"):
                current_file = line[6:].strip()
            elif line.startswith("--- /dev/null"):
                current_file = None
            elif line.startswith("@@ ") and current_file:
                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                match = re.search(r'@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@', line)
                if match:
                    old_start = int(match.group(1))
                    old_count = int(match.group(2)) if match.group(2) is not None else 1
                    if old_count > 0:
                        hunks.append((current_file, old_start, old_count))

        return hunks

    def generate_labeled_dataset(
        self,
        extracted_commits_df: Optional[pd.DataFrame] = None,
        branch: str = "HEAD"
    ) -> pd.DataFrame:
        """
        Runs full SZZ algorithm pipeline, links fixing commits to introducing commits,
        scores confidence using multi-signal evidence, and exports labeled dataset.
        """
        # 1. Detect fixing candidates
        candidates = self.detect_candidate_fix_commits(branch=branch)

        # Tracing evidence map: introducing_hash -> list of fix records
        blame_evidence_map: Dict[str, List[Dict[str, Any]]] = {}

        # 2. Perform SZZ line-tracing for each candidate fix
        for fix in candidates:
            if not fix["is_candidate_fix"] or not fix["parent"]:
                continue

            fix_sha = fix["full_hash"]
            parent_sha = fix["parent"]

            # Parse deleted line hunks
            hunks = self.parse_deleted_line_hunks(parent_sha, fix_sha)

            for file_path, old_start, old_count in hunks:
                end_line = old_start + old_count - 1
                origin_shas = self.trace_line_blame(parent_sha, file_path, old_start, end_line)

                for intro_sha in origin_shas:
                    # Exclude self-blame
                    if intro_sha == fix_sha:
                        continue

                    # Determine confidence level
                    if fix["has_issue_id"] and fix["has_fix_keyword"]:
                        confidence = "HIGH"
                    elif fix["has_issue_id"] or fix["has_fix_keyword"]:
                        confidence = "MEDIUM"
                    else:
                        confidence = "LOW"

                    label_source = "revert_trace" if fix["is_revert"] else "szz_blame_traced"

                    evidence_record = {
                        "fixing_commit_hash": fix["commit_hash"],
                        "fixing_full_hash": fix_sha,
                        "fixing_issue_id": fix["issue_id"],
                        "confidence": confidence,
                        "label_source": label_source,
                        "traced_file": file_path,
                        "traced_lines_count": old_count
                    }

                    if intro_sha not in blame_evidence_map:
                        blame_evidence_map[intro_sha] = []
                    blame_evidence_map[intro_sha].append(evidence_record)

        # 3. Construct labeled dataset
        # Log all commits in history to establish baseline
        log_res = self._run_git(["log", "--reverse", "--format=%H|%h|%aI|%an", branch])
        all_commits = []
        for l in log_res.stdout.splitlines():
            if not l.strip(): continue
            parts = l.strip().split('|')
            all_commits.append({
                "full_hash": parts[0],
                "commit_hash": parts[1],
                "commit_timestamp": parts[2],
                "author": parts[3]
            })

        labeled_records = []

        for c in all_commits:
            full_sha = c["full_hash"]

            if full_sha in blame_evidence_map:
                evidences = blame_evidence_map[full_sha]
                
                # Aggregate multiple fixing commits cleanly
                fixing_shas = list(set([e["fixing_commit_hash"] for e in evidences]))
                issue_ids = list(set([e["fixing_issue_id"] for e in evidences if e["fixing_issue_id"]]))
                confidences = [e["confidence"] for e in evidences]
                sources = [e["label_source"] for e in evidences]
                total_lines = sum([e["traced_lines_count"] for e in evidences])

                # Highest confidence assignment
                if "HIGH" in confidences:
                    agg_confidence = "HIGH"
                elif "MEDIUM" in confidences:
                    agg_confidence = "MEDIUM"
                else:
                    agg_confidence = "LOW"

                agg_source = "revert_trace" if "revert_trace" in sources else "szz_blame_traced"

                record = {
                    "commit_hash": c["commit_hash"],
                    "full_hash": full_sha,
                    "commit_timestamp": c["commit_timestamp"],
                    "author": c["author"],
                    "bug_introduced": 1,  # SZZ-identified bug-introducing commit
                    "label_semantics": "SZZ-identified bug-introducing commit",
                    "label_source": agg_source,
                    "label_confidence": agg_confidence,
                    "fixing_commit_hash": ",".join(fixing_shas),
                    "fixing_issue_id": ",".join(issue_ids) if issue_ids else "",
                    "traced_lines_count": total_lines
                }
            else:
                record = {
                    "commit_hash": c["commit_hash"],
                    "full_hash": full_sha,
                    "commit_timestamp": c["commit_timestamp"],
                    "author": c["author"],
                    "bug_introduced": 0,  # No SZZ evidence that the commit introduced a later fixed defect
                    "label_semantics": "No SZZ evidence that the commit introduced a later fixed defect",
                    "label_source": "unlabeled_no_evidence",
                    "label_confidence": "NONE",
                    "fixing_commit_hash": "",
                    "fixing_issue_id": "",
                    "traced_lines_count": 0
                }

            labeled_records.append(record)

        df_labeled = pd.DataFrame(labeled_records)
        print(f"[SZZLabeler] Generated labels for {len(df_labeled)} commits. (SZZ-identified bug-introducing: {(df_labeled['bug_introduced']==1).sum()})")
        return df_labeled

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    labeler = SZZDefectLabeler(repo_root)
    df_labels = labeler.generate_labeled_dataset()
    print("\n--- SZZ LABELED DATASET SAMPLE (HEAD 5) ---")
    print(df_labels[['commit_hash', 'bug_introduced', 'label_semantics', 'label_source', 'label_confidence', 'fixing_commit_hash']].head())
