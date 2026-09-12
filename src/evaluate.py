import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

def evaluate_all_models(raw_data_path: str, models_dir: str, reports_dir: str) -> dict:
    """
    Evaluates Dummy Classifier, Logistic Regression, Random Forest, and XGBoost models
    on the exact same unseen test split.
    """
    # 1. Load Data & Create Exact Unseen Test Split
    X, y = load_and_validate_data(raw_data_path)
    X_train_scaled, X_test_scaled, y_train, y_test, _ = prepare_and_preprocess_data(X, y)
    
    # Extract unscaled test features for tree models
    _, X_test_raw, _, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # 2. Load Trained Model Artifacts
    lr_model = joblib.load(os.path.join(models_dir, 'logistic_regression.pkl'))
    rf_model = joblib.load(os.path.join(models_dir, 'random_forest.pkl'))
    xgb_model = joblib.load(os.path.join(models_dir, 'xgboost.pkl'))
    
    # 3. Fit Dummy Reference Classifier on Train Split
    dummy_model = DummyClassifier(strategy='most_frequent')
    dummy_model.fit(X_train_scaled, y_train)
    
    # 4. Generate Predictions & Probabilities (Threshold 0.50)
    models_dict = {
        "Dummy Majority": (dummy_model, X_test_scaled),
        "Logistic Regression": (lr_model, X_test_scaled),
        "Random Forest": (rf_model, X_test_raw),
        "XGBoost": (xgb_model, X_test_raw)
    }
    
    results = {
        "dataset_info": {
            "total_samples": len(X),
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "test_class_0_clean": int((y_test == 0).sum()),
            "test_class_1_bug": int((y_test == 1).sum()),
            "features_used": EXPECTED_FEATURES
        },
        "models": {}
    }
    
    table_rows = []
    
    for name, (model, X_eval) in models_dict.items():
        y_pred = model.predict(X_eval)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_eval)[:, 1]
        else:
            y_prob = np.zeros(len(y_test))
            
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_prob)) > 1 else 0.5
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        pos_preds = int((y_pred == 1).sum())
        
        model_res = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "positive_predictions_count": pos_preds,
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
            }
        }
        
        # Feature importance / coefficients extraction
        if hasattr(model, "feature_importances_"):
            importances = dict(zip(EXPECTED_FEATURES, model.feature_importances_.tolist()))
            model_res["feature_importances"] = importances
        elif hasattr(model, "coef_"):
            coefs = dict(zip(EXPECTED_FEATURES, model.coef_[0].tolist()))
            model_res["coefficients"] = coefs
            
        results["models"][name] = model_res
        table_rows.append({
            "Model": name,
            "Accuracy": f"{acc:.4f}",
            "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}",
            "F1-Score": f"{f1:.4f}",
            "ROC-AUC": f"{auc:.4f}",
            "Pos Preds (>=0.5)": f"{pos_preds}/60",
            "Confusion Matrix [TN, FP, FN, TP]": f"[{tn}, {fp}, {fn}, {tp}]"
        })
        
    # 5. Print Comparison Summary Table
    df_summary = pd.DataFrame(table_rows)
    
    print("\n==========================================================================================")
    print("                      PATCHGUARD MODEL COMPARISON SUMMARY                                 ")
    print("==========================================================================================")
    print(f"Test Split: 60 samples (Clean: {(y_test==0).sum()}, Bug-prone: {(y_test==1).sum()})\n")
    print(df_summary.to_string(index=False))
    print("==========================================================================================\n")
    
    # 6. Print Feature Importances
    print("--- FEATURE IMPORTANCE ANALYSIS (Tree-Based Models) ---")
    print(f"{'Feature Name':<35} | {'Random Forest':<15} | {'XGBoost':<15}")
    print("-" * 70)
    rf_imp = results["models"]["Random Forest"]["feature_importances"]
    xgb_imp = results["models"]["XGBoost"]["feature_importances"]
    
    for feat in EXPECTED_FEATURES:
        print(f"{feat:<35} | {rf_imp[feat]:>15.4f} | {xgb_imp[feat]:>15.4f}")
    print("==========================================================================================\n")
    
    # 7. Save Reports
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, 'model_comparison.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    # Also save as baseline_results.json for backward compatibility
    with open(os.path.join(reports_dir, 'baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"[Evaluate] Comparison report saved to: {report_path}")
    return results

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'raw', 'synthetic_commits.csv'))
    models_dir = os.path.abspath(os.path.join(script_dir, '..', 'models'))
    reports_dir = os.path.abspath(os.path.join(script_dir, '..', 'reports'))
    
    evaluate_all_models(raw_data_path, models_dir, reports_dir)
