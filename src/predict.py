import os
import sys
import joblib
import pandas as pd
import numpy as np

from data_loader import EXPECTED_FEATURES

def predict_commit_risk(
    lines_added: int,
    lines_deleted: int,
    files_changed: int,
    functions_changed: int,
    code_churn: int = None,
    num_directories_touched: int = 1,
    is_test_file_modified: int = 0,
    avg_lines_changed_per_file: float = None,
    max_lines_changed_in_single_file: int = None,
    num_source_files_changed: int = None,
    model_name: str = "random_forest",
    models_dir: str = None
) -> dict:
    """
    Predicts bug introduction risk for a single code change / commit.
    Supports model choice: 'random_forest' (default), 'xgboost', or 'logistic_regression'.
    """
    if code_churn is None:
        code_churn = lines_added + lines_deleted
        
    if avg_lines_changed_per_file is None:
        avg_lines_changed_per_file = round(code_churn / max(1, files_changed), 2)
        
    if max_lines_changed_in_single_file is None:
        max_lines_changed_in_single_file = int(max(avg_lines_changed_per_file, code_churn * 0.7))
        
    if num_source_files_changed is None:
        num_source_files_changed = max(1, files_changed)
        
    if models_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.abspath(os.path.join(script_dir, '..', 'models'))
        
    model_filename_map = {
        "logistic_regression": "logistic_regression.pkl",
        "random_forest": "random_forest.pkl",
        "xgboost": "xgboost.pkl"
    }
    
    if model_name not in model_filename_map:
        raise ValueError(f"Unknown model_name: {model_name}. Options: {list(model_filename_map.keys())}")
        
    model_path = os.path.join(models_dir, model_filename_map[model_name])
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact missing at {model_path}. Please run src/train.py first.")
        
    model = joblib.load(model_path)
    
    # Construct raw feature DataFrame
    raw_features = pd.DataFrame([{
        'lines_added': lines_added,
        'lines_deleted': lines_deleted,
        'code_churn': code_churn,
        'files_changed': files_changed,
        'functions_changed': functions_changed,
        'num_directories_touched': num_directories_touched,
        'is_test_file_modified': is_test_file_modified,
        'avg_lines_changed_per_file': avg_lines_changed_per_file,
        'max_lines_changed_in_single_file': max_lines_changed_in_single_file,
        'num_source_files_changed': num_source_files_changed
    }])[EXPECTED_FEATURES]
    
    # Scale features ONLY for Logistic Regression
    if model_name == "logistic_regression":
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler artifact missing at {scaler_path}.")
        scaler = joblib.load(scaler_path)
        features_for_pred = scaler.transform(raw_features)
    else:
        features_for_pred = raw_features
        
    # Inference
    prob = float(model.predict_proba(features_for_pred)[0, 1])
    pred_class = int(model.predict(features_for_pred)[0])
    
    if prob < 0.33:
        risk_level = "LOW"
    elif prob < 0.66:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
        
    prediction_label = "Bug-prone" if pred_class == 1 else "Clean"
    
    result = {
        "model_used": model_name,
        "prediction_class": pred_class,
        "prediction_label": prediction_label,
        "bug_probability": round(prob, 4),
        "risk_level": risk_level,
        "input_features": raw_features.to_dict(orient='records')[0]
    }
    
    return result

if __name__ == '__main__':
    sample_commit = {
        'lines_added': 120,
        'lines_deleted': 30,
        'files_changed': 6,
        'functions_changed': 10,
        'code_churn': 150,
        'num_directories_touched': 3,
        'is_test_file_modified': 1,
        'avg_lines_changed_per_file': 25.0,
        'max_lines_changed_in_single_file': 90,
        'num_source_files_changed': 5
    }
    
    for m_name in ["logistic_regression", "random_forest", "xgboost"]:
        res = predict_commit_risk(model_name=m_name, **sample_commit)
        print(f"[{m_name.upper():<20}] -> Class: {res['prediction_class']} ({res['prediction_label']:<10}), Prob: {res['bug_probability']:.4f}, Risk: {res['risk_level']}")
