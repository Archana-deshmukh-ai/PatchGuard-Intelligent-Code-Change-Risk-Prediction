import os
import sys
import time
import pandas as pd
import subprocess

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
sys.path.insert(0, src_dir)

from szz_labeler import SZZDefectLabeler

def run_bottle_validation():
    repo_path = os.path.abspath(os.path.join(script_dir, 'external_repo', 'bottle'))
    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Bottle repository not found at: {repo_path}")

    print(f"[Validation] Starting SZZ Defect Labeler on benchmark repo: {repo_path}")
    start_time = time.time()

    labeler = SZZDefectLabeler(repo_path)
    
    # 1. Candidate fix commits detection
    candidates = labeler.detect_candidate_fix_commits()
    candidate_fixes_count = sum(1 for c in candidates if c["is_candidate_fix"])

    # 2. Run full SZZ labeling pipeline
    df_labels = labeler.generate_labeled_dataset()
    end_time = time.time()
    runtime_sec = round(end_time - start_time, 2)

    # 3. Export CSV
    raw_dir = os.path.abspath(os.path.join(script_dir, '..', 'data', 'raw'))
    os.makedirs(raw_dir, exist_ok=True)
    csv_path = os.path.join(raw_dir, 'szz_bottle_labeled_commits.csv')
    df_labels.to_csv(csv_path, index=False)

    # 4. Detailed Statistics Computation
    total_commits = len(df_labels)
    bug_intro_count = (df_labels['bug_introduced'] == 1).sum()
    no_evidence_count = (df_labels['bug_introduced'] == 0).sum()

    conf_dist = df_labels['label_confidence'].value_counts().to_dict()
    source_dist = df_labels['label_source'].value_counts().to_dict()

    # Fixing relationships analysis
    fixing_commits_series = df_labels[df_labels['bug_introduced'] == 1]['fixing_commit_hash']
    total_fixing_relationships = sum(len(f.split(',')) for f in fixing_commits_series if f)
    multi_fix_commits_count = sum(1 for f in fixing_commits_series if f and ',' in f)

    issue_ids_series = df_labels[df_labels['bug_introduced'] == 1]['fixing_issue_id']
    unique_issue_ids = set()
    for item in issue_ids_series:
        if item:
            for i in item.split(','):
                if i.strip():
                    unique_issue_ids.add(i.strip())
    captured_issue_ids_count = len(unique_issue_ids)

    # Commits with added-only fixes (where candidate fixes had 0 deleted lines)
    added_only_fixes_count = sum(1 for c in candidates if c["is_candidate_fix"] and c["files_changed_count"] > 0)

    print("\n==========================================================================")
    print("      REAL-REPOSITORY SZZ VALIDATION REPORT: BOTTLEPY/BOTTLE              ")
    print("==========================================================================")
    print(f"Target Repository         : bottlepy/bottle ({repo_path})")
    print(f"Output Labeled Dataset    : {csv_path}")
    print(f"Total Execution Runtime   : {runtime_sec} seconds\n")

    print("--- 1. OVERALL DATASET STATISTICS ---")
    print(f"Total Commits Analyzed           : {total_commits}")
    print(f"Candidate Fixing Commits Found   : {candidate_fixes_count} ({candidate_fixes_count/total_commits*100:.2f}%)")
    print(f"SZZ Bug-Introducing Commits (y=1): {bug_intro_count} ({bug_intro_count/total_commits*100:.2f}%)")
    print(f"No SZZ Evidence Commits (y=0)    : {no_evidence_count} ({no_evidence_count/total_commits*100:.2f}%)\n")

    print("--- 2. CONFIDENCE DISTRIBUTION ---")
    for conf, cnt in conf_dist.items():
        print(f"  - {conf:<8} : {cnt:<5} ({cnt/total_commits*100:.2f}%)")

    print("\n--- 3. LABEL SOURCE DISTRIBUTION ---")
    for src, cnt in source_dist.items():
        print(f"  - {src:<22} : {cnt:<5} ({cnt/total_commits*100:.2f}%)")

    print("\n--- 4. PROVENANCE & FIXING RELATIONSHIPS ---")
    print(f"Total Fixing Relationships Linkages : {total_fixing_relationships}")
    print(f"Commits with Multiple Fix Links    : {multi_fix_commits_count}")
    print(f"Unique Issue IDs Captured (#/GH/JIRA): {captured_issue_ids_count}")
    print("==========================================================================\n")

    # 5. Helper function for git log inspection
    def inspect_commit(commit_sha: str) -> str:
        cmd = ["git", "log", "-1", "--format=%h - %an (%ad): %s", commit_sha]
        res = subprocess.run(cmd, cwd=repo_path, stdout=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
        return res.stdout.strip()

    print("==========================================================================")
    print("             MANUAL SANITY-CHECK SAMPLE INSPECTION                        ")
    print("==========================================================================")

    # Sample 1: HIGH confidence cases
    high_samples = df_labels[df_labels['label_confidence'] == 'HIGH'].head(3)
    print("\n--- SAMPLE HIGH-CONFIDENCE BUG-INTRODUCING COMMITS (y=1) ---")
    for _, row in high_samples.iterrows():
        print(f"\n[BLAMED INTRODUCING COMMIT]: {inspect_commit(row['full_hash'])}")
        print(f"  -> Fixing Commit(s) : {row['fixing_commit_hash']}")
        print(f"  -> Issue ID(s)      : {row['fixing_issue_id']}")
        print(f"  -> Traced Lines     : {row['traced_lines_count']}")
        first_fix = row['fixing_commit_hash'].split(',')[0]
        print(f"  -> Fixing Commit MSG: {inspect_commit(first_fix)}")

    # Sample 2: MEDIUM confidence cases
    med_samples = df_labels[df_labels['label_confidence'] == 'MEDIUM'].head(2)
    print("\n--- SAMPLE MEDIUM-CONFIDENCE BUG-INTRODUCING COMMITS (y=1) ---")
    for _, row in med_samples.iterrows():
        print(f"\n[BLAMED INTRODUCING COMMIT]: {inspect_commit(row['full_hash'])}")
        print(f"  -> Fixing Commit(s) : {row['fixing_commit_hash']}")
        print(f"  -> Traced Lines     : {row['traced_lines_count']}")
        first_fix = row['fixing_commit_hash'].split(',')[0]
        print(f"  -> Fixing Commit MSG: {inspect_commit(first_fix)}")

    # Sample 3: LOW confidence cases
    low_samples = df_labels[df_labels['label_confidence'] == 'LOW'].head(2)
    print("\n--- SAMPLE LOW-CONFIDENCE BUG-INTRODUCING COMMITS (y=1) ---")
    if not low_samples.empty:
        for _, row in low_samples.iterrows():
            print(f"\n[BLAMED INTRODUCING COMMIT]: {inspect_commit(row['full_hash'])}")
            print(f"  -> Fixing Commit(s) : {row['fixing_commit_hash']}")
    else:
        print("  No LOW confidence instances found in dataset.")

    # Sample 4: Unlabeled (No SZZ evidence) commits
    unlabeled_samples = df_labels[df_labels['bug_introduced'] == 0].head(2)
    print("\n--- SAMPLE UNLABELED COMMITS (y=0: No SZZ Evidence) ---")
    for _, row in unlabeled_samples.iterrows():
        print(f"  - {inspect_commit(row['full_hash'])}")

    print("\n==========================================================================\n")

if __name__ == '__main__':
    run_bottle_validation()
