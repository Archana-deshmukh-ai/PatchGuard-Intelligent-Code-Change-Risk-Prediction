import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

FEATURE_COLUMNS: List[str] = [
    'lines_added',
    'lines_deleted',
    'code_churn',
    'files_changed',
    'functions_changed',
    'num_directories_touched',
    'is_test_file_modified',
    'avg_lines_changed_per_file',
    'max_lines_changed_in_single_file',
    'num_source_files_changed'
]

TARGET_COLUMN: str = 'bug_introduced'
IDENTIFIER_COLUMNS: List[str] = ['commit_hash', 'commit_timestamp']
OUTPUT_COLUMNS: List[str] = IDENTIFIER_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]

# Columns that MUST NOT be present in ML features (provenance / future SZZ leakage)
PROVENANCE_COLUMNS: List[str] = [
    'fixing_commit_hash',
    'fixing_issue_id',
    'label_confidence',
    'label_source',
    'label_semantics',
    'traced_lines_count'
]

class RealDatasetBuilder:
    """
    Constructs and validates real-data ML datasets by joining commit-time Git features
    with retrospective SZZ defect labels using commit_hash as the primary join key.
    Enforces strict temporal ordering, leakage protection, and data integrity checks.
    """
    def __init__(self, features_path: str, szz_labels_path: str):
        self.features_path = os.path.abspath(features_path)
        self.szz_labels_path = os.path.abspath(szz_labels_path)
        
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Git features file not found: {self.features_path}")
        if not os.path.exists(self.szz_labels_path):
            raise FileNotFoundError(f"SZZ labels file not found: {self.szz_labels_path}")

    def build_and_validate(self, output_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads, joins, validates, and exports the real-data ML dataset.
        
        Args:
            output_path: Target filepath for processed ML CSV.
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Final dataset and validation metrics dict.
        """
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 1. Load Datasets
        df_features = pd.read_csv(self.features_path)
        df_szz = pd.read_csv(self.szz_labels_path)

        features_initial_count = len(df_features)
        szz_initial_count = len(df_szz)

        print(f"[DatasetBuilder] Loaded {features_initial_count} commits from Git features ({os.path.basename(self.features_path)})")
        print(f"[DatasetBuilder] Loaded {szz_initial_count} commits from SZZ labels ({os.path.basename(self.szz_labels_path)})")

        # 2. Validate Column Schemas
        missing_features_cols = [c for c in IDENTIFIER_COLUMNS + FEATURE_COLUMNS if c not in df_features.columns]
        if missing_features_cols:
            raise ValueError(f"Features file is missing required columns: {missing_features_cols}")

        missing_szz_cols = [c for c in ['commit_hash', TARGET_COLUMN] if c not in df_szz.columns]
        if missing_szz_cols:
            raise ValueError(f"SZZ file is missing required columns: {missing_szz_cols}")

        # Check for duplicates in join key
        features_dups = df_features['commit_hash'].duplicated().sum()
        szz_dups = df_szz['commit_hash'].duplicated().sum()
        if features_dups > 0:
            raise ValueError(f"Git features contain {features_dups} duplicate commit_hash records.")
        if szz_dups > 0:
            raise ValueError(f"SZZ labels contain {szz_dups} duplicate commit_hash records.")

        # 3. Perform LEFT JOIN on commit_hash
        # Git commit features dataset is primary to maintain complete commit-time history
        df_joined = pd.merge(
            df_features,
            df_szz[['commit_hash', TARGET_COLUMN]],
            on='commit_hash',
            how='left'
        )

        # 4. Check Join Coverage & Unmatched Commits
        unmatched_commits_count = df_joined[TARGET_COLUMN].isna().sum()
        features_only_count = unmatched_commits_count
        
        # Check commits in SZZ but not in Git features (e.g., merge commits filtered from features)
        szz_set = set(df_szz['commit_hash'])
        features_set = set(df_features['commit_hash'])
        szz_only_count = len(szz_set - features_set)

        if unmatched_commits_count > 0:
            print(f"[DatasetBuilder] WARNING: {unmatched_commits_count} Git feature commits had no SZZ label match.")

        # 5. Handle Missing Target Labels safely
        # Fill missing SZZ labels with 0 ('no identified defect evidence') and cast to int
        df_joined[TARGET_COLUMN] = df_joined[TARGET_COLUMN].fillna(0).astype(int)

        # 6. Data Integrity & Invariants Validation
        # Invariant 1: code_churn == lines_added + lines_deleted
        churn_mismatches = (df_joined['code_churn'] != (df_joined['lines_added'] + df_joined['lines_deleted'])).sum()
        if churn_mismatches > 0:
            raise ValueError(f"Invariance Violation: {churn_mismatches} rows have code_churn != lines_added + lines_deleted")

        # Invariant 2: bug_introduced contains only 0 or 1
        invalid_labels = (~df_joined[TARGET_COLUMN].isin([0, 1])).sum()
        if invalid_labels > 0:
            raise ValueError(f"Invalid Target Values: {invalid_labels} rows have target values outside {{0, 1}}")

        # Invariant 3: No missing values in 10 ML features
        missing_feature_vals = df_joined[FEATURE_COLUMNS].isna().sum().sum()
        if missing_feature_vals > 0:
            raise ValueError(f"Missing Values: {missing_feature_vals} missing values found in ML feature columns.")

        # Invariant 4: All ML features numeric
        non_numeric_features = []
        for col in FEATURE_COLUMNS:
            if not np.issubdtype(df_joined[col].dtype, np.number):
                non_numeric_features.append(col)
        if non_numeric_features:
            raise TypeError(f"Non-numeric ML feature columns found: {non_numeric_features}")

        # Invariant 5: No SZZ provenance/future leakage columns in final output
        leakage_cols_found = [c for c in PROVENANCE_COLUMNS if c in OUTPUT_COLUMNS]
        if leakage_cols_found:
            raise ValueError(f"DATA LEAKAGE ERROR: Provenance columns found in output schema: {leakage_cols_found}")

        # 7. Temporal Validation
        # Convert timestamp to datetime for check
        ts_datetime = pd.to_datetime(df_joined['commit_timestamp'], utc=True)
        min_timestamp = ts_datetime.min().isoformat()
        max_timestamp = ts_datetime.max().isoformat()
        is_monotonically_increasing = ts_datetime.is_monotonic_increasing

        if not is_monotonically_increasing:
            print("[DatasetBuilder] Note: Commit timestamps are not strictly monotonic. Sorting chronologically.")
            df_joined['datetime_temp'] = ts_datetime
            df_joined = df_joined.sort_values('datetime_temp').drop(columns=['datetime_temp']).reset_index(drop=True)
            ts_datetime = pd.to_datetime(df_joined['commit_timestamp'], utc=True)
            is_monotonically_increasing = ts_datetime.is_monotonic_increasing

        # 8. Select Clean Output Columns
        df_final = df_joined[OUTPUT_COLUMNS].copy()

        # 9. Compute Summary Validation Statistics
        total_rows = len(df_final)
        total_cols = len(df_final.columns)
        bug_intro_count = (df_final[TARGET_COLUMN] == 1).sum()
        no_evidence_count = (df_final[TARGET_COLUMN] == 0).sum()
        pct_label_1 = round(bug_intro_count / total_rows * 100, 2)
        pct_label_0 = round(no_evidence_count / total_rows * 100, 2)

        # 10. Save Output CSV
        df_final.to_csv(output_path, index=False)
        print(f"[DatasetBuilder] Successfully exported real ML dataset to: {output_path}")

        metrics = {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "duplicate_commit_hash_count": 0,
            "missing_values_by_column": df_final.isna().sum().to_dict(),
            "commits_in_features_missing_from_szz": features_only_count,
            "commits_in_szz_missing_from_features": szz_only_count,
            "label_1_count": bug_intro_count,
            "label_0_count": no_evidence_count,
            "percentage_label_1": pct_label_1,
            "percentage_label_0": pct_label_0,
            "min_timestamp": min_timestamp,
            "max_timestamp": max_timestamp,
            "is_monotonically_increasing": is_monotonically_increasing,
            "all_features_numeric": len(non_numeric_features) == 0,
            "future_provenance_leakage_detected": len(leakage_cols_found) > 0,
            "output_filepath": output_path
        }

        return df_final, metrics

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.abspath(os.path.join(script_dir, '..', 'data', 'raw'))
    processed_dir = os.path.abspath(os.path.join(script_dir, '..', 'data', 'processed'))

    features_file = os.path.join(raw_dir, 'bottle_git_commits.csv')
    szz_file = os.path.join(raw_dir, 'szz_bottle_labeled_commits.csv')
    output_file = os.path.join(processed_dir, 'bottle_real_dataset.csv')

    builder = RealDatasetBuilder(features_file, szz_file)
    df_result, metrics_dict = builder.build_and_validate(output_file)

    print("\n==========================================================================")
    print("           REAL ML DATASET BUILD & VALIDATION METRICS                     ")
    print("==========================================================================")
    for k, v in metrics_dict.items():
        print(f"  {k:<42} : {v}")
    print("==========================================================================\n")
