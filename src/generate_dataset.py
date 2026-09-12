import os
import numpy as np
import pandas as pd

def generate_synthetic_dataset(num_samples: int = 300, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset of commit metadata and bug labels for pipeline testing.
    
    The target `bug_introduced` is generated using a log-odds probabilistic model 
    influenced by multiple features plus Gaussian noise to ensure realistic non-deterministic behavior.
    """
    np.random.seed(random_state)
    
    # 1. Generate commit features
    lines_added = np.random.exponential(scale=40, size=num_samples).astype(int) + 2
    lines_deleted = np.random.exponential(scale=20, size=num_samples).astype(int)
    
    # Exactly maintain code_churn identity
    code_churn = lines_added + lines_deleted
    
    # files_changed & functions_changed scale with churn but include Poisson noise
    files_changed = np.maximum(1, np.random.poisson(lam=1.0 + code_churn / 40.0))
    functions_changed = np.maximum(1, np.random.poisson(lam=files_changed * 1.1 + code_churn / 50.0))
    
    # 2. Probabilistic label generation: log-odds with noise
    log_odds = (
        -2.2
        + 0.010 * code_churn
        + 0.12 * files_changed
        + 0.10 * functions_changed
        + np.random.normal(loc=0.0, scale=0.75, size=num_samples)  # Gaussian noise
    )
    
    # Convert log-odds to probabilities via Sigmoid
    probabilities = 1.0 / (1.0 + np.exp(-log_odds))
    
    # Assign binary labels based on probabilities
    bug_introduced = (np.random.uniform(0.0, 1.0, size=num_samples) < probabilities).astype(int)
    
    df = pd.DataFrame({
        'lines_added': lines_added,
        'lines_deleted': lines_deleted,
        'files_changed': files_changed,
        'functions_changed': functions_changed,
        'code_churn': code_churn,
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
    print(f"Successfully generated {len(df)} synthetic commit records at: {output_file}")
