import os
from typing import Tuple, List
import pandas as pd

EXPECTED_FEATURES: List[str] = [
    'lines_added',
    'lines_deleted',
    'files_changed',
    'functions_changed',
    'code_churn'
]
TARGET_COLUMN: str = 'bug_introduced'

def load_and_validate_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads dataset from CSV, validates column schema and data integrity,
    and splits into features (X) and target (y).
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features X and Target y.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or if null values are detected.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
        
    df = pd.read_csv(filepath)
    
    # 1. Validate expected columns
    expected_columns = EXPECTED_FEATURES + [TARGET_COLUMN]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")
        
    # 2. Validate missing values
    null_counts = df[expected_columns].isna().sum()
    if null_counts.sum() > 0:
        raise ValueError(f"Dataset contains missing values:\n{null_counts[null_counts > 0]}")
        
    # 3. Separate features X and target y
    X = df[EXPECTED_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    
    print(f"[DataLoader] Successfully loaded {len(df)} rows from {os.path.basename(filepath)}")
    print(f"[DataLoader] Features shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y

if __name__ == '__main__':
    # Test script execution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'raw', 'synthetic_commits.csv'))
    
    X, y = load_and_validate_data(raw_data_path)
    print("\nSample Features X (head):")
    print(X.head())
    print("\nSample Target y (head):")
    print(y.head())
