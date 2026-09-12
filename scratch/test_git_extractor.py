import os
import sys
import pandas as pd

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
sys.path.insert(0, src_dir)

from git_extractor import GitRepositoryExtractor

def validate_git_extractor():
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))
    extractor = GitRepositoryExtractor(repo_root)
    
    # Extract commit history
    df = extractor.extract_commit_history(include_merges=False)
    
    # Save to data/raw/patchguard_git_commits.csv
    raw_dir = os.path.join(repo_root, 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    out_csv = os.path.join(raw_dir, 'patchguard_git_commits.csv')
    df.to_csv(out_csv, index=False)
    
    print("\n==========================================================================")
    print("           PHASE 1 GIT EXTRACTOR VALIDATION REPORT                        ")
    print("==========================================================================")
    print(f"Target Repository Path : {repo_root}")
    print(f"Extracted Dataset CSV  : {out_csv}")
    print(f"Total Rows (Commits)   : {len(df)}")
    print(f"Total Columns          : {len(df.columns)}")
    
    print("\n--- SCHEMA & DATA TYPES ---")
    for col, dtype in df.dtypes.items():
        print(f"{col:<35} | {str(dtype):<15}")
        
    print("\n--- MISSING VALUE SUMMARY ---")
    null_counts = df.isna().sum()
    print(null_counts)
    
    print("\n--- FIRST 5 EXTRACTED COMMIT RECORDS ---")
    print(df[['commit_hash', 'commit_timestamp', 'lines_added', 'lines_deleted', 'code_churn', 'files_changed', 'functions_changed', 'num_directories_touched', 'is_test_file_modified', 'num_source_files_changed']].head().to_string())
    
    print("\n--- EDGE CASE VERIFICATION ---")
    print(f"Commits with 0 lines added/deleted (e.g. pure renames/metadata) : {(df['code_churn'] == 0).sum()}")
    print(f"Commits modifying test files                           : {(df['is_test_file_modified'] == 1).sum()}")
    print(f"Commits touching multiple directories                  : {(df['num_directories_touched'] > 1).sum()}")
    print(f"Chronological order check (Oldest -> Newest)            : Timestamps sorted = {df['commit_timestamp'].is_monotonic_increasing}")
    print("==========================================================================\n")

if __name__ == '__main__':
    validate_git_extractor()
