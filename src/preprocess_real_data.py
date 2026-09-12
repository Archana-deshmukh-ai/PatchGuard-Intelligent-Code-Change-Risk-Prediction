import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler

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

class RealDataPreprocessor:
    """
    Model-aware preprocessing pipeline for real-world software defect prediction.
    Enforces strict temporal scaling isolation (fitting StandardScaler ONLY on X_train).
    Preserves raw data artifacts for tree-based models (Random Forest, XGBoost).
    """
    def __init__(self, data_dir: str):
        self.data_dir = os.path.abspath(data_dir)
        self.train_path = os.path.join(self.data_dir, 'bottle_train.csv')
        self.val_path = os.path.join(self.data_dir, 'bottle_validation.csv')
        self.test_path = os.path.join(self.data_dir, 'bottle_test.csv')

        for p in [self.train_path, self.val_path, self.test_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Temporal dataset file not found: {p}")

        # Load raw temporal split DataFrames
        self.df_train_raw = pd.read_csv(self.train_path)
        self.df_val_raw = pd.read_csv(self.val_path)
        self.df_test_raw = pd.read_csv(self.test_path)

        self.scaler: StandardScaler = None

    def validate_missing_values(self) -> Dict[str, int]:
        """Verifies that zero missing values exist across Train, Validation, and Test splits."""
        missing = {
            "train": int(self.df_train_raw[FEATURE_COLUMNS].isna().sum().sum()),
            "val": int(self.df_val_raw[FEATURE_COLUMNS].isna().sum().sum()),
            "test": int(self.df_test_raw[FEATURE_COLUMNS].isna().sum().sum())
        }
        if sum(missing.values()) > 0:
            raise ValueError(f"Missing values detected in feature columns: {missing}")
        return missing

    def compute_descriptive_stats(self) -> Dict[str, pd.DataFrame]:
        """Calculates raw feature descriptive statistics (min, max, mean, median, std) for each split."""
        def get_stats(df_sub: pd.DataFrame) -> pd.DataFrame:
            stats = pd.DataFrame({
                "min": df_sub[FEATURE_COLUMNS].min(),
                "max": df_sub[FEATURE_COLUMNS].max(),
                "mean": df_sub[FEATURE_COLUMNS].mean().round(4),
                "median": df_sub[FEATURE_COLUMNS].median().round(4),
                "std": df_sub[FEATURE_COLUMNS].std().round(4)
            })
            return stats

        return {
            "train": get_stats(self.df_train_raw),
            "val": get_stats(self.df_val_raw),
            "test": get_stats(self.df_test_raw)
        }

    def compute_train_correlation(self, threshold: float = 0.70) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Calculates Pearson correlation matrix on X_train only and reports strongly correlated pairs.
        Does NOT drop correlated features automatically.
        """
        corr_matrix = self.df_train_raw[FEATURE_COLUMNS].corr()
        strong_pairs = []

        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_matrix.iloc[i, j]
                if abs(val) >= threshold:
                    strong_pairs.append({
                        "feature_1": cols[i],
                        "feature_2": cols[j],
                        "correlation": round(val, 4)
                    })

        return corr_matrix, strong_pairs

    def process_and_fit_scaler(self, model_dir: str) -> Dict[str, Any]:
        """
        Executes standard scaling with STRICT Train-Only fitting.
        Saves fitted scaler artifact to models/real_data_scaler.pkl.
        """
        model_dir = os.path.abspath(model_dir)
        os.makedirs(model_dir, exist_ok=True)
        scaler_file = os.path.join(model_dir, 'real_data_scaler.pkl')

        # 1. Extract X and y DataFrames/Series
        X_train = self.df_train_raw[FEATURE_COLUMNS].copy()
        y_train = self.df_train_raw[TARGET_COLUMN].copy()

        X_val = self.df_val_raw[FEATURE_COLUMNS].copy()
        y_val = self.df_val_raw[TARGET_COLUMN].copy()

        X_test = self.df_test_raw[FEATURE_COLUMNS].copy()
        y_test = self.df_test_raw[TARGET_COLUMN].copy()

        # 2. Fit StandardScaler ONLY on X_train (CRITICAL LEAKAGE RULE)
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)

        # 3. Transform X_train, X_val, X_test using the SAME fitted scaler
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # 4. Save fitted scaler artifact
        joblib.dump(self.scaler, scaler_file)
        print(f"[Preprocessor] Saved fitted StandardScaler artifact to: {scaler_file}")

        # 5. Compute post-scaling statistics
        def get_scaled_stats(scaled_arr: np.ndarray) -> pd.DataFrame:
            return pd.DataFrame({
                "mean": np.mean(scaled_arr, axis=0).round(4),
                "std": np.std(scaled_arr, axis=0).round(4),
                "min": np.min(scaled_arr, axis=0).round(4),
                "max": np.max(scaled_arr, axis=0).round(4)
            }, index=FEATURE_COLUMNS)

        scaled_stats = {
            "train": get_scaled_stats(X_train_scaled),
            "val": get_scaled_stats(X_val_scaled),
            "test": get_scaled_stats(X_test_scaled)
        }

        # 6. Verify Target and Row Counts
        label_integrity = {
            "y_train_unique": sorted(list(y_train.unique())),
            "y_val_unique": sorted(list(y_val.unique())),
            "y_test_unique": sorted(list(y_test.unique())),
            "row_counts": {
                "train": len(X_train_scaled),
                "val": len(X_val_scaled),
                "test": len(X_test_scaled)
            }
        }

        metrics = {
            "scaler_artifact_path": scaler_file,
            "fitted_on_train_only": True,
            "raw_files_unaltered": True,
            "scaled_stats": scaled_stats,
            "label_integrity": label_integrity
        }

        return metrics

    def get_data_for_model(self, model_type: str) -> Tuple[Any, Any, Any, pd.Series, pd.Series, pd.Series]:
        """
        Returns model-aware data splits:
        - Linear/Logistic models -> Scaled feature arrays
        - Tree-based models (Random Forest, XGBoost) -> Raw feature DataFrames
        """
        X_train = self.df_train_raw[FEATURE_COLUMNS]
        y_train = self.df_train_raw[TARGET_COLUMN]

        X_val = self.df_val_raw[FEATURE_COLUMNS]
        y_val = self.df_val_raw[TARGET_COLUMN]

        X_test = self.df_test_raw[FEATURE_COLUMNS]
        y_test = self.df_test_raw[TARGET_COLUMN]

        if model_type.lower() in ['logistic_regression', 'logistic', 'linear']:
            if self.scaler is None:
                self.scaler = StandardScaler()
                self.scaler.fit(X_train)
            return (
                self.scaler.transform(X_train),
                self.scaler.transform(X_val),
                self.scaler.transform(X_test),
                y_train,
                y_val,
                y_test
            )
        elif model_type.lower() in ['random_forest', 'xgboost', 'tree', 'rf', 'xgb']:
            # Tree models use raw unscaled features
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.abspath(os.path.join(script_dir, '..', 'data', 'processed'))
    models_dir = os.path.abspath(os.path.join(script_dir, '..', 'models'))

    preprocessor = RealDataPreprocessor(processed_dir)

    # 1. Missing Value Check
    missing_dict = preprocessor.validate_missing_values()

    # 2. Raw Feature Stats
    raw_stats = preprocessor.compute_descriptive_stats()

    # 3. TRAIN Feature Correlation
    corr_matrix, strong_corrs = preprocessor.compute_train_correlation(threshold=0.70)

    # 4. Standard Scaling with Train-Only Fitting
    prep_metrics = preprocessor.process_and_fit_scaler(models_dir)

    print("\n==========================================================================")
    print("         REAL DATASET PREPROCESSING & SCALING METRICS                     ")
    print("==========================================================================")
    print("--- 1. MISSING VALUES ---")
    print(f"  {missing_dict}")

    print("\n--- 2. STRONGLY CORRELATED FEATURE PAIRS (|r| >= 0.70 ON TRAIN) ---")
    for pair in strong_corrs:
        print(f"  - {pair['feature_1']:<32} <-> {pair['feature_2']:<32} : r = {pair['correlation']}")

    print("\n--- 3. TRAIN SCALED STATISTICS (EXPECT MEAN=0, STD=1) ---")
    print(prep_metrics['scaled_stats']['train'])

    print("\n--- 4. VALIDATION SCALED STATISTICS (NATURAL SHIFT EXPECTED) ---")
    print(prep_metrics['scaled_stats']['val'])

    print("\n--- 5. TEST SCALED STATISTICS (NATURAL SHIFT EXPECTED) ---")
    print(prep_metrics['scaled_stats']['test'])

    print("\n--- 6. LABEL INTEGRITY & LEAKAGE CHECK ---")
    print(f"  Fitted Scaler Path       : {prep_metrics['scaler_artifact_path']}")
    print(f"  Fitted on TRAIN Only     : {prep_metrics['fitted_on_train_only']}")
    print(f"  Raw CSV Files Unaltered  : {prep_metrics['raw_files_unaltered']}")
    print(f"  Label Unique Values      : {prep_metrics['label_integrity']['y_train_unique']}")
    print(f"  Row Counts (Train/Val/Test): {prep_metrics['label_integrity']['row_counts']}")
    print("==========================================================================\n")
