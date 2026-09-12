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
    models_dir: str = None
) -> dict:
    """
    Predicts bug introduction risk for a single code change / commit.
    
    Args:
        lines_added: Number of added lines.
        lines_deleted: Number of deleted lines.
        files_changed: Number of modified files.
        functions_changed: Number of modified functions.
        code_churn: Optional explicit churn. If None, calculated as lines_added + lines_deleted.
        models_dir: Path to directory containing saved model and scaler artifacts.
        
    Returns:
        dict: Prediction summary containing prediction class, bug probability, and risk level.
    """
    if code_churn is None:
        code_churn = lines_added + lines_deleted
        
    if models_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.abspath(os.path.join(script_dir, '..', 'models'))
        
    model_path = os.path.join(models_dir, 'logistic_regression.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or Scaler artifact missing. Please run src/train.py first.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Construct feature dataframe matching training schema
    raw_features = pd.DataFrame([{
        'lines_added': lines_added,
        'lines_deleted': lines_deleted,
        'files_changed': files_changed,
        'functions_changed': functions_changed,
        'code_churn': code_churn
    }])[EXPECTED_FEATURES]
    
    # Scale input features using training scaler
    scaled_features = scaler.transform(raw_features)
    
    # Inference
    prob = float(model.predict_proba(scaled_features)[0, 1])
    pred_class = int(model.predict(scaled_features)[0])
    
    # Map risk level (Engineering heuristic boundaries for Step 1)
    if prob < 0.33:
        risk_level = "LOW"
    elif prob < 0.66:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
        
    prediction_label = "Bug-prone" if pred_class == 1 else "Clean"
    
    result = {
        "prediction_class": pred_class,
        "prediction_label": prediction_label,
        "bug_probability": round(prob, 4),
        "risk_level": risk_level,
        "input_features": {
            "lines_added": lines_added,
            "lines_deleted": lines_deleted,
            "files_changed": files_changed,
            "functions_changed": functions_changed,
            "code_churn": code_churn
        }
    }
    
    return result

if __name__ == '__main__':
    # Test example commit
    sample_commit = {
        'lines_added': 120,
        'lines_deleted': 30,
        'files_changed': 6,
        'functions_changed': 10,
        'code_churn': 150
    }
    
    res = predict_commit_risk(**sample_commit)
    
    print("\n==================================================")
    print("       SINGLE COMMIT RISK INFERENCE TEST          ")
    print("==================================================")
    print(f"Input Code Change  : {res['input_features']}")
    print(f"Prediction Label   : {res['prediction_label']} (Class {res['prediction_class']})")
    print(f"Bug Probability    : {res['bug_probability']}")
    print(f"Risk Level         : {res['risk_level']}")
    print("--------------------------------------------------")
    print("Note: Risk level boundaries (0-0.33 LOW, 0.33-0.66 MEDIUM, 0.66-1.0 HIGH)")
    print("are engineering choices for pipeline testing, not calibrated thresholds.")
    print("==================================================\n")
