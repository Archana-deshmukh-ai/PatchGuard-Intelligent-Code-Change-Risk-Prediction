import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

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
EXPECTED_COLUMNS: List[str] = IDENTIFIER_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]

class TemporalSplitter:
    """
    Performs chronological (temporal) train / validation / test splits on a real-data ML dataset.
    Ensures zero temporal overlap, verifies feature consistency, and audits class distributions.
    """
    def __init__(self, dataset_path: str):
        self.dataset_path = os.path.abspath(dataset_path)
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

    def split_and_validate(
        self,
        output_dir: str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """
        Splits dataset chronologically into train, validation, and test CSV artifacts.
        
        Args:
            output_dir: Destination directory for train/val/test CSV files.
            train_ratio: Target proportion for train set (default: 0.70).
            val_ratio: Target proportion for validation set (default: 0.15).
            test_ratio: Target proportion for test set (default: 0.15).
            
        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]: Split DataFrames and metrics dict.
        """
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        df = pd.read_csv(self.dataset_path)

        # 1. Verify Expected Column Schema
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

        # 2. Check & Enforce Chronological Order
        df['dt_temp'] = pd.to_datetime(df['commit_timestamp'], utc=True)
        if not df['dt_temp'].is_monotonic_increasing:
            print("[TemporalSplitter] Dataset was not strictly monotonic. Sorting chronologically.")
            df = df.sort_values('dt_temp').reset_index(drop=True)

        df = df.drop(columns=['dt_temp'])

        # 3. Compute Programmatic Row Boundaries (70% / 15% / 15%)
        total_rows = len(df)
        train_count = int(round(total_rows * train_ratio))
        val_count = int(round(total_rows * val_ratio))
        test_count = total_rows - train_count - val_count

        df_train = df.iloc[:train_count].copy().reset_index(drop=True)
        df_val = df.iloc[train_count:train_count + val_count].copy().reset_index(drop=True)
        df_test = df.iloc[train_count + val_count:].copy().reset_index(drop=True)

        # 4. Save Processed Split CSV Artifacts
        train_file = os.path.join(output_dir, 'bottle_train.csv')
        val_file = os.path.join(output_dir, 'bottle_validation.csv')
        test_file = os.path.join(output_dir, 'bottle_test.csv')

        df_train.to_csv(train_file, index=False)
        df_val.to_csv(val_file, index=False)
        df_test.to_csv(test_file, index=False)

        # 5. Temporal Boundary Validation
        train_ts_min, train_ts_max = df_train['commit_timestamp'].min(), df_train['commit_timestamp'].max()
        val_ts_min, val_ts_max = df_val['commit_timestamp'].min(), df_val['commit_timestamp'].max()
        test_ts_min, test_ts_max = df_test['commit_timestamp'].min(), df_test['commit_timestamp'].max()

        train_val_strict = pd.to_datetime(train_ts_max, utc=True) <= pd.to_datetime(val_ts_min, utc=True)
        val_test_strict = pd.to_datetime(val_ts_max, utc=True) <= pd.to_datetime(test_ts_min, utc=True)

        # 6. Duplicate & Overlap Audits
        train_hashes = set(df_train['commit_hash'])
        val_hashes = set(df_val['commit_hash'])
        test_hashes = set(df_test['commit_hash'])

        train_val_overlap = len(train_hashes.intersection(val_hashes))
        val_test_overlap = len(val_hashes.intersection(test_hashes))
        train_test_overlap = len(train_hashes.intersection(test_hashes))
        total_overlaps = train_val_overlap + val_test_overlap + train_test_overlap

        # 7. Class Distribution Computations
        def calc_class_dist(df_sub: pd.DataFrame) -> Dict[str, Any]:
            pos = int((df_sub[TARGET_COLUMN] == 1).sum())
            neg = int((df_sub[TARGET_COLUMN] == 0).sum())
            tot = len(df_sub)
            return {
                "total_rows": tot,
                "positive_label_1_count": pos,
                "negative_label_0_count": neg,
                "positive_label_1_pct": round(pos / tot * 100, 2),
                "negative_label_0_pct": round(neg / tot * 100, 2)
            }

        train_dist = calc_class_dist(df_train)
        val_dist = calc_class_dist(df_val)
        test_dist = calc_class_dist(df_test)

        # 8. Feature Invariance & Consistency Audits
        def audit_split_invariants(df_sub: pd.DataFrame, split_name: str):
            churn_errs = (df_sub['code_churn'] != (df_sub['lines_added'] + df_sub['lines_deleted'])).sum()
            if churn_errs > 0:
                raise ValueError(f"Invariance Error ({split_name}): {churn_errs} rows mismatch code_churn")
            null_errs = df_sub[FEATURE_COLUMNS].isna().sum().sum()
            if null_errs > 0:
                raise ValueError(f"Null Error ({split_name}): {null_errs} missing feature values")
            for col in FEATURE_COLUMNS:
                if not np.issubdtype(df_sub[col].dtype, np.number):
                    raise TypeError(f"Type Error ({split_name}): Column {col} is not numeric")

        audit_split_invariants(df_train, "TRAIN")
        audit_split_invariants(df_val, "VALIDATION")
        audit_split_invariants(df_test, "TEST")

        print(f"[TemporalSplitter] TRAIN split: {len(df_train)} rows ({train_ts_min} -> {train_ts_max})")
        print(f"[TemporalSplitter] VAL split  : {len(df_val)} rows ({val_ts_min} -> {val_ts_max})")
        print(f"[TemporalSplitter] TEST split : {len(df_test)} rows ({test_ts_min} -> {test_ts_max})")

        metrics = {
            "total_dataset_rows": total_rows,
            "train_row_count": train_count,
            "val_row_count": val_count,
            "test_row_count": test_count,
            "row_count_checksum_valid": (train_count + val_count + test_count) == total_rows,
            "train_timestamp_range": [train_ts_min, train_ts_max],
            "val_timestamp_range": [val_ts_min, val_ts_max],
            "test_timestamp_range": [test_ts_min, test_ts_max],
            "train_to_val_strictly_chronological": train_val_strict,
            "val_to_test_strictly_chronological": val_test_strict,
            "commit_hash_overlap_count": total_overlaps,
            "train_class_distribution": train_dist,
            "val_class_distribution": val_dist,
            "test_class_distribution": test_dist,
            "test_set_contains_both_classes": (test_dist["positive_label_1_count"] > 0) and (test_dist["negative_label_0_count"] > 0),
            "output_files": {
                "train": train_file,
                "validation": val_file,
                "test": test_file
            }
        }

        return {"train": df_train, "validation": df_val, "test": df_test}, metrics

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.abspath(os.path.join(script_dir, '..', 'data', 'processed'))
    dataset_file = os.path.join(processed_dir, 'bottle_real_dataset.csv')

    splitter = TemporalSplitter(dataset_file)
    splits_dict, metrics_dict = splitter.split_and_validate(processed_dir)

    print("\n==========================================================================")
    print("         REAL DATASET TEMPORAL SPLIT & VALIDATION METRICS                 ")
    print("==========================================================================")
    for k, v in metrics_dict.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sub_k, sub_v in v.items():
                print(f"    - {sub_k:<30} : {sub_v}")
        else:
            print(f"  {k:<38} : {v}")
    print("==========================================================================\n")
