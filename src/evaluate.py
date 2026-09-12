import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from data_loader import load_and_validate_data, EXPECTED_FEATURES
from preprocessing import prepare_and_preprocess_data

def evaluate_models(
    raw_data_path: str, 
    models_dir: str, 
    reports_dir: str
) -> dict:
    """
    Evaluates trained Logistic Regression model against a Dummy (Majority Class) baseline
    on the unseen test set. Generates comprehensive evaluation metrics, confusion matrix,
    feature coefficients, and saves report to reports/baseline_results.json.
    """
    # 1. Load Data & Preprocess to get exact unseen test split
    X, y = load_and_validate_data(raw_data_path)
    X_train_scaled, X_test_scaled, y_train, y_test, _ = prepare_and_preprocess_data(X, y)
    
    # 2. Load Trained Logistic Regression Model
    model_path = os.path.join(models_dir, 'logistic_regression.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Run train.py first.")
    
    lr_model = joblib.load(model_path)
    
    # 3. Logistic Regression Predictions
    lr_y_pred = lr_model.predict(X_test_scaled)
    lr_y_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # 4. Logistic Regression Metrics
    lr_acc = accuracy_score(y_test, lr_y_pred)
    lr_prec = precision_score(y_test, lr_y_pred, zero_division=0)
    lr_rec = recall_score(y_test, lr_y_pred, zero_division=0)
    lr_f1 = f1_score(y_test, lr_y_pred, zero_division=0)
    lr_auc = roc_auc_score(y_test, lr_y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, lr_y_pred).ravel()
    
    # 5. Dummy Baseline (Majority Class Classifier)
    dummy_model = DummyClassifier(strategy='most_frequent')
    dummy_model.fit(X_train_scaled, y_train)
    
    dummy_y_pred = dummy_model.predict(X_test_scaled)
    dummy_y_prob = dummy_model.predict_proba(X_test_scaled)[:, 1] if hasattr(dummy_model, "predict_proba") else np.zeros(len(y_test))
    
    dummy_acc = accuracy_score(y_test, dummy_y_pred)
    dummy_prec = precision_score(y_test, dummy_y_pred, zero_division=0)
    dummy_rec = recall_score(y_test, dummy_y_pred, zero_division=0)
    dummy_f1 = f1_score(y_test, dummy_y_pred, zero_division=0)
    dummy_auc = roc_auc_score(y_test, dummy_y_prob) if len(np.unique(dummy_y_prob)) > 1 else 0.5
    d_tn, d_fp, d_fn, d_tp = confusion_matrix(y_test, dummy_y_pred).ravel()
    
    # 6. Feature Coefficients
    coefficients = dict(zip(EXPECTED_FEATURES, lr_model.coef_[0].tolist()))
    intercept = float(lr_model.intercept_[0])
    
    # 7. Build Evaluation Results Dictionary
    results = {
        "dataset": {
            "total_samples": len(X),
            "train_samples": len(X_train_scaled),
            "test_samples": len(X_test_scaled),
            "feature_names": EXPECTED_FEATURES,
            "target_name": "bug_introduced"
        },
        "models": {
            "logistic_regression": {
                "accuracy": float(lr_acc),
                "precision": float(lr_prec),
                "recall": float(lr_rec),
                "f1_score": float(lr_f1),
                "roc_auc": float(lr_auc),
                "confusion_matrix": {
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp)
                },
                "intercept": intercept,
                "coefficients": coefficients
            },
            "dummy_baseline": {
                "strategy": "most_frequent",
                "accuracy": float(dummy_acc),
                "precision": float(dummy_prec),
                "recall": float(dummy_rec),
                "f1_score": float(dummy_f1),
                "roc_auc": float(dummy_auc),
                "confusion_matrix": {
                    "true_negatives": int(d_tn),
                    "false_positives": int(d_fp),
                    "false_negatives": int(d_fn),
                    "true_positives": int(d_tp)
                }
            }
        }
    }
    
    # 8. Save Report JSON
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, 'baseline_results.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"[Evaluate] Evaluation report saved to: {report_path}")
    
    # 9. Console Output Summary
    print("\n==================================================")
    print("           MODEL EVALUATION SUMMARY               ")
    print("==================================================")
    print(f"Test Set Size: {len(X_test_scaled)} samples (Stratified Split)\n")
    
    print("--- DUMMY BASELINE (Majority Class '0') ---")
    print(f"Accuracy : {dummy_acc:.4f}")
    print(f"Precision: {dummy_prec:.4f}")
    print(f"Recall   : {dummy_rec:.4f}")
    print(f"F1-Score : {dummy_f1:.4f}")
    print(f"ROC-AUC  : {dummy_auc:.4f}")
    print(f"Confusion Matrix: [TN={d_tn}, FP={d_fp}, FN={d_fn}, TP={d_tp}]\n")
    
    print("--- LOGISTIC REGRESSION BASELINE ---")
    print(f"Accuracy : {lr_acc:.4f}")
    print(f"Precision: {lr_prec:.4f}")
    print(f"Recall   : {lr_rec:.4f}")
    print(f"F1-Score : {lr_f1:.4f}")
    print(f"ROC-AUC  : {lr_auc:.4f}")
    print(f"Confusion Matrix: [TN={tn}, FP={fp}, FN={fn}, TP={tp}]\n")
    
    print("--- LOGISTIC REGRESSION COEFFICIENTS ---")
    print(f"{'Feature':<20} | {'Coefficient':<12}")
    print("-" * 35)
    for feat, coef in coefficients.items():
        print(f"{feat:<20} | {coef:>12.4f}")
    print(f"{'Intercept (bias)':<20} | {intercept:>12.4f}")
    print("==================================================\n")
    
    return results

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'raw', 'synthetic_commits.csv'))
    models_dir = os.path.abspath(os.path.join(script_dir, '..', 'models'))
    reports_dir = os.path.abspath(os.path.join(script_dir, '..', 'reports'))
    
    evaluate_models(raw_data_path, models_dir, reports_dir)
