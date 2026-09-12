import os
import joblib
from sklearn.linear_model import LogisticRegression

from data_loader import load_and_validate_data
from preprocessing import prepare_and_preprocess_data, save_processed_data

def train_baseline_model(raw_data_path: str, models_dir: str, processed_dir: str) -> None:
    """
    Loads raw data, performs train/test split and feature scaling, 
    trains Logistic Regression baseline, and persists model and scaler artifacts.
    """
    # 1. Load Data
    X, y = load_and_validate_data(raw_data_path)
    
    # 2. Preprocess (Split and Scale)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_and_preprocess_data(X, y)
    
    # Save processed datasets for transparency
    save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, list(X.columns), processed_dir)
    
    # 3. Train Logistic Regression Model
    model = LogisticRegression(random_state=42, solver='lbfgs')
    model.fit(X_train_scaled, y_train)
    
    print("[Train] Logistic Regression model trained successfully.")
    
    # 4. Save artifacts
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'logistic_regression.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"[Train] Saved model artifact to: {model_path}")
    print(f"[Train] Saved scaler artifact to: {scaler_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'raw', 'synthetic_commits.csv'))
    models_dir = os.path.abspath(os.path.join(script_dir, '..', 'models'))
    processed_dir = os.path.abspath(os.path.join(script_dir, '..', 'data', 'processed'))
    
    train_baseline_model(raw_data_path, models_dir, processed_dir)
