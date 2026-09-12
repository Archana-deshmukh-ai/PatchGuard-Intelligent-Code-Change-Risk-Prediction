import os
import numpy as np
import pandas as pd

def generate_synthetic_dataset(num_samples: int = 300, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset of 10 commit metadata features and binary bug labels for pipeline testing.
    
    The target `bug_introduced` is generated using a log-odds probabilistic model 
    influenced by multiple features plus Gaussian noise to ensure realistic non-deterministic behavior.
    """
    np.random.seed(random_state)
    
    # 1. Core Churn Features
    lines_added = np.random.exponential(scale=40, size=num_samples).astype(int) + 2
    lines_deleted = np.random.exponential(scale=20, size=num_samples).astype(int)
    code_churn = lines_added + lines_deleted
    
    # 2. File and Function Scope Features
    files_changed = np.maximum(1, np.random.poisson(lam=1.0 + code_churn / 40.0))
    functions_changed = np.maximum(1, np.random.poisson(lam=files_changed * 1.1 + code_churn / 50.0))
    
    # 3. New Advanced Features
    # Architectural Sprawl: Directories touched (cannot exceed files_changed)
    num_directories_touched = np.maximum(
        1, 
        np.minimum(files_changed, np.random.poisson(lam=0.8 + files_changed / 3.0))
    )
    
    # Testing Safeguard: Larger commits have higher probability of touching test files
    prob_test_modified = 1.0 - np.exp(-files_changed * 0.18)
    is_test_file_modified = (np.random.uniform(0.0, 1.0, size=num_samples) < prob_test_modified).astype(int)
    
    # Concentration Metrics
    avg_lines_changed_per_file = np.round(code_churn / files_changed, 2)
    
    # Max lines changed in a single file (between avg_lines and total code_churn)
    concentration_ratio = np.random.uniform(0.45, 0.95, size=num_samples)
    max_lines_changed_in_single_file = np.maximum(
        avg_lines_changed_per_file.astype(int),
        np.round(code_churn * concentration_ratio).astype(int)
    )
    
    # Executable Scope: Source files changed vs non-source (docs/configs)
    non_source_count = np.random.binomial(n=files_changed, p=0.15)
    num_source_files_changed = np.maximum(1, files_changed - non_source_count)
    
    # 4. Probabilistic Label Generation (Log-odds with multi-feature influence + Gaussian noise)
    log_odds = (
        -2.3
        + 0.006 * code_churn
        + 0.08 * files_changed
        + 0.06 * functions_changed
        + 0.15 * num_directories_touched
        - 0.45 * is_test_file_modified                # Modifying tests reduces bug risk
        + 0.004 * max_lines_changed_in_single_file    # High single-file concentration increases risk
        + 0.05 * num_source_files_changed
        + np.random.normal(loc=0.0, scale=0.75, size=num_samples)  # Random noise
    )
    
    # Convert log-odds to probabilities via Sigmoid
    probabilities = 1.0 / (1.0 + np.exp(-log_odds))
    
    # Assign binary labels based on probabilities
    bug_introduced = (np.random.uniform(0.0, 1.0, size=num_samples) < probabilities).astype(int)
    
    df = pd.DataFrame({
        'lines_added': lines_added,
        'lines_deleted': lines_deleted,
        'code_churn': code_churn,
        'files_changed': files_changed,
        'functions_changed': functions_changed,
        'num_directories_touched': num_directories_touched,
        'is_test_file_modified': is_test_file_modified,
        'avg_lines_changed_per_file': avg_lines_changed_per_file,
        'max_lines_changed_in_single_file': max_lines_changed_in_single_file,
        'num_source_files_changed': num_source_files_changed,
        'bug_introduced': bug_introduced
    })
    
    return df

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data', 'raw'))
    os.makedirs(raw_data_dir, exist_ok=True)
    
    df = generate_synthetic_dataset(num_samples=300, random_state=42)
    output_file = os.path.join(raw_data_dir, 'synthetic_commits.csv')
    df.to_csv(output_file, index=False)
    print(f"Successfully generated {len(df)} synthetic commit records (10 features) at: {output_file}")
