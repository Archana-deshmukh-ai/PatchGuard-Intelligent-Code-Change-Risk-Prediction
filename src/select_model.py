import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from preprocess_real_data import RealDataPreprocessor, FEATURE_COLUMNS, TARGET_COLUMN

THRESHOLDS = [
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
]

def analyze_thresholds_for_model(
    model_name: str,
    y_true: pd.Series,
    y_prob: np.ndarray,
    roc_auc: float
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluates classification performance metrics across a spectrum of decision thresholds.
    """
    threshold_results = []
    
    for t in THRESHOLDS:
        y_pred = (y_prob >= t).astype(int)
        
        acc = round(float(accuracy_score(y_true, y_pred)), 4)
        prec = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
        rec = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
        f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        pred_pos_count = int(np.sum(y_pred))
        
        res = {
            "threshold": round(t, 2),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": round(roc_auc, 4),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "predicted_positives": pred_pos_count
        }
        threshold_results.append(res)

    # Convert to DataFrame for operating point search
    df_res = pd.DataFrame(threshold_results)

    # 1. Operating Point: Maximum F1
    max_f1_row = df_res.loc[df_res['f1'].idxmax()].to_dict()

    # 2. Operating Point: High Recall (target recall >= 0.75 or maximum available recall with prec > 0.40)
    high_rec_candidates = df_res[df_res['recall'] >= 0.75]
    if not high_rec_candidates.empty:
        high_rec_row = high_rec_candidates.loc[high_rec_candidates['f1'].idxmax()].to_dict()
    else:
        high_rec_row = df_res.loc[df_res['recall'].idxmax()].to_dict()

    # 3. Operating Point: Balanced Precision / Recall (minimal abs(prec - rec))
    df_res['prec_rec_diff'] = (df_res['precision'] - df_res['recall']).abs()
    # exclude extreme 0 thresholds
    balanced_candidates = df_res[(df_res['precision'] > 0) & (df_res['recall'] > 0)]
    if not balanced_candidates.empty:
        balanced_row = balanced_candidates.loc[balanced_candidates['prec_rec_diff'].idxmin()].to_dict()
    else:
        balanced_row = max_f1_row

    # 4. Operating Point: Conservative / High Precision (precision >= 0.70 or max precision)
    high_prec_candidates = df_res[df_res['precision'] >= 0.70]
    if not high_prec_candidates.empty:
        high_prec_row = high_prec_candidates.loc[high_prec_candidates['f1'].idxmax()].to_dict()
    else:
        # highest precision with positive predicted count > 0
        non_zero_prec = df_res[df_res['predicted_positives'] > 0]
        high_prec_row = non_zero_prec.loc[non_zero_prec['precision'].idxmax()].to_dict()

    # Clean temporary column
    df_res.drop(columns=['prec_rec_diff'], inplace=True, errors='ignore')

    operating_points = {
        "max_f1": max_f1_row,
        "high_recall": high_rec_row,
        "balanced": balanced_row,
        "high_precision": high_prec_row
    }

    return threshold_results, operating_points

def run_model_selection():
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    data_dir = os.path.join(project_root, 'data', 'processed')
    models_dir = os.path.join(project_root, 'models')
    reports_dir = os.path.join(project_root, 'reports')

    os.makedirs(reports_dir, exist_ok=True)

    # 1. Initialize Preprocessor & Load Validation Data ONLY
    preprocessor = RealDataPreprocessor(data_dir)
    preprocessor.process_and_fit_scaler(models_dir)

    # Extract Validation features and target
    X_val_scaled = preprocessor.scaler.transform(preprocessor.df_val_raw[FEATURE_COLUMNS])
    X_val_raw = preprocessor.df_val_raw[FEATURE_COLUMNS]
    y_val = preprocessor.df_val_raw[TARGET_COLUMN]

    val_sample_count = len(y_val)
    val_pos_count = int(np.sum(y_val == 1))
    val_neg_count = int(np.sum(y_val == 0))

    print(f"[ModelSelection] Loaded VALIDATION set ONLY ({val_sample_count} rows: {val_pos_count} pos / {val_neg_count} neg).")
    print(f"[ModelSelection] TEST set bottle_test.csv is 100% UNTOUCHED.")

    # 2. Load Trained Model Artifacts
    logistic_model = joblib.load(os.path.join(models_dir, 'logistic_regression_real.pkl'))
    rf_model = joblib.load(os.path.join(models_dir, 'random_forest_real.pkl'))
    xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_real.pkl'))
    dummy_model = joblib.load(os.path.join(models_dir, 'dummy_classifier.pkl'))

    # 3. Generate Validation Probabilities
    logistic_prob = logistic_model.predict_proba(X_val_scaled)[:, 1]
    rf_prob = rf_model.predict_proba(X_val_raw)[:, 1]
    xgb_prob = xgb_model.predict_proba(X_val_raw)[:, 1]

    # Calculate ROC-AUC scores on Validation
    logistic_auc = float(roc_auc_score(y_val, logistic_prob))
    rf_auc = float(roc_auc_score(y_val, rf_prob))
    xgb_auc = float(roc_auc_score(y_val, xgb_prob))

    # 4. Run Threshold Analysis
    logistic_thresh_results, logistic_op = analyze_thresholds_for_model("Logistic Regression", y_val, logistic_prob, logistic_auc)
    rf_thresh_results, rf_op = analyze_thresholds_for_model("Random Forest", y_val, rf_prob, rf_auc)
    xgb_thresh_results, xgb_op = analyze_thresholds_for_model("XGBoost", y_val, xgb_prob, xgb_auc)

    # Flatten all threshold records to export CSV
    all_rows = []
    for m_name, res_list in [("Logistic Regression", logistic_thresh_results), ("Random Forest", rf_thresh_results), ("XGBoost", xgb_thresh_results)]:
        for r in res_list:
            r_copy = dict(r)
            r_copy["model"] = m_name
            all_rows.append(r_copy)

    df_thresholds = pd.DataFrame(all_rows)
    csv_path = os.path.join(reports_dir, 'threshold_analysis_validation.csv')
    df_thresholds.to_csv(csv_path, index=False)

    json_report = {
        "metadata": {
            "validation_sample_count": val_sample_count,
            "validation_positive_count": val_pos_count,
            "validation_negative_count": val_neg_count,
            "test_set_touched": False
        },
        "models": {
            "logistic_regression": {
                "roc_auc": round(logistic_auc, 4),
                "operating_points": logistic_op,
                "threshold_sweep": logistic_thresh_results
            },
            "random_forest": {
                "roc_auc": round(rf_auc, 4),
                "operating_points": rf_op,
                "threshold_sweep": rf_thresh_results
            },
            "xgboost": {
                "roc_auc": round(xgb_auc, 4),
                "operating_points": xgb_op,
                "threshold_sweep": xgb_thresh_results
            }
        }
    }

    json_path = os.path.join(reports_dir, 'threshold_analysis_validation.json')
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)

    print(f"[ModelSelection] Saved CSV analysis to: {csv_path}")
    print(f"[ModelSelection] Saved JSON report to  : {json_path}")

    # Print summary output
    print("\n==========================================================================")
    print("      PHASE 4.5 — VALIDATION THRESHOLD ANALYSIS & OPERATING POINTS        ")
    print("==========================================================================")
    
    for m_name, op in [("Logistic Regression", logistic_op), ("Random Forest", rf_op), ("XGBoost", xgb_op)]:
        print(f"\n--- {m_name.upper()} OPERATING POINTS ---")
        print(f"  Max F1 Operating Point     : Threshold={op['max_f1']['threshold']} | F1={op['max_f1']['f1']} | Prec={op['max_f1']['precision']} | Rec={op['max_f1']['recall']} | TP={op['max_f1']['tp']} | FP={op['max_f1']['fp']} | FN={op['max_f1']['fn']}")
        print(f"  High Recall Operating Point : Threshold={op['high_recall']['threshold']} | F1={op['high_recall']['f1']} | Prec={op['high_recall']['precision']} | Rec={op['high_recall']['recall']} | TP={op['high_recall']['tp']} | FP={op['high_recall']['fp']} | FN={op['high_recall']['fn']}")
        print(f"  Balanced Operating Point    : Threshold={op['balanced']['threshold']} | F1={op['balanced']['f1']} | Prec={op['balanced']['precision']} | Rec={op['balanced']['recall']} | TP={op['balanced']['tp']} | FP={op['balanced']['fp']} | FN={op['balanced']['fn']}")
        print(f"  High Precision Op Point     : Threshold={op['high_precision']['threshold']} | F1={op['high_precision']['f1']} | Prec={op['high_precision']['precision']} | Rec={op['high_precision']['recall']} | TP={op['high_precision']['tp']} | FP={op['high_precision']['fp']} | FN={op['high_precision']['fn']}")

    print("\n==========================================================================\n")

if __name__ == '__main__':
    run_model_selection()
