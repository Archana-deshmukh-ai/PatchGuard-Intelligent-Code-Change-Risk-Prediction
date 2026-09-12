import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data_loader import load_and_validate_data
from preprocessing import prepare_and_preprocess_data, save_processed_data

def train_all_models(raw_data_path: str, models_dir: str, processed_dir: str) -> None:
    """
    Loads raw data, performs train/test split, scales features for linear models,
    and trains Logistic Regression, Random Forest, and XGBoost models.
    
    Tree-based models (Random Forest, XGBoost) are trained on unscaled features,
    while Logistic Regression is trained on StandardScaler-transformed features.
    """
    # 1. Load Data
    X, y = load_and_validate_data(raw_data_path)
    
    # 2. Preprocess (Split and Scale)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_and_preprocess_data(X, y)
    save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, list(X.columns), processed_dir)
    
    # Extract unscaled train split for tree models (exact same sample split)
    X_train_raw, _, _, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # 3. Train Baseline & Tree Models
    # A. Logistic Regression (Scaled Features)
    lr_model = LogisticRegression(random_state=42, solver='lbfgs')
    lr_model.fit(X_train_scaled, y_train)
    
    # B. Random Forest (Unscaled Features)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_raw, y_train)
    
    # C. XGBoost (Unscaled Features)
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train_raw, y_train)
    
    print("[Train] Successfully trained Logistic Regression, Random Forest, and XGBoost models.")
    
    # 4. Save artifacts
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(lr_model, os.path.join(models_dir, 'logistic_regression.pkl'))
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest.pkl'))
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgboost.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    
    print(f"[Train] Saved model artifacts to: {models_dir}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'raw', 'synthetic_commits.csv'))
    models_dir = os.path.abspath(os.path.join(script_dir, '..', 'models'))
    processed_dir = os.path.abspath(os.path.join(script_dir, '..', 'data', 'processed'))
    
    train_all_models(raw_data_path, models_dir, processed_dir)
