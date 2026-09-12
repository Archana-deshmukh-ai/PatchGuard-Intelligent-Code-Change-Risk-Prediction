import os
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_and_preprocess_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler]:
    """
    Splits data into train/test sets and applies StandardScaler.
    
    CRITICAL: StandardScaler is fit ONLY on X_train to prevent data leakage.
    X_test is transformed using the X_train scaler parameters.
    
    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Proportion of dataset to reserve for test split (default: 0.2).
        random_state: Seed for random number generator.
        
    Returns:
        Tuple containing:
        - X_train_scaled (np.ndarray)
        - X_test_scaled (np.ndarray)
        - y_train (pd.Series)
        - y_test (pd.Series)
        - scaler (StandardScaler)
    """
    # 1. Train / Test Split with Stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    # 2. Feature Preprocessing (Standardization)
    scaler = StandardScaler()
    
    # Fit scaler ONLY on training features
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform test features using training statistics
    X_test_scaled = scaler.transform(X_test)
    
    print(f"[Preprocessing] Train split: {X_train_scaled.shape[0]} samples")
    print(f"[Preprocessing] Test split: {X_test_scaled.shape[0]} samples")
    print(f"[Preprocessing] Feature scaling applied using StandardScaler (Fit on Train only).")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_processed_data(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list,
    output_dir: str
) -> None:
    """
    Saves processed feature arrays and target splits to CSV files in data/processed/.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    train_df['bug_introduced'] = y_train.values
    
    test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    test_df['bug_introduced'] = y_test.values
    
    train_df.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)
    
    print(f"[Preprocessing] Processed splits saved to: {output_dir}")

if __name__ == '__main__':
    from data_loader import load_and_validate_data
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'raw', 'synthetic_commits.csv'))
    processed_dir = os.path.abspath(os.path.join(script_dir, '..', 'data', 'processed'))
    
    X, y = load_and_validate_data(raw_data_path)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_and_preprocess_data(X, y)
    save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, list(X.columns), processed_dir)
