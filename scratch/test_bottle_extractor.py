import os
import sys
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
sys.path.insert(0, src_dir)

from git_extractor import GitRepositoryExtractor

repo_path = os.path.abspath(os.path.join(script_dir, 'external_repo', 'bottle'))
extractor = GitRepositoryExtractor(repo_path)
df_features = extractor.extract_commit_history(include_merges=False)

print(f"Extracted {len(df_features)} commits.")
print(df_features.head())

# Save raw features CSV for bottle
raw_dir = os.path.abspath(os.path.join(script_dir, '..', 'data', 'raw'))
df_features.to_csv(os.path.join(raw_dir, 'bottle_git_commits.csv'), index=False)
print("Saved features to data/raw/bottle_git_commits.csv")
