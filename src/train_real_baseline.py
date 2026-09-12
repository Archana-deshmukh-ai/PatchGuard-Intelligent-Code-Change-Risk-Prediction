import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Import preprocessing pipeline
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from preprocess_real_data import RealDataPreprocessor, FEATURE_COLUMNS, TARGET_COLUMN

def evaluate_model_performance(
    model: Any,
    X: Any,
    y: pd.Series,
    split_name: str
) -> Dict[str, Any]:
    """
    Evaluates model performance metrics on a given data split (Train or Validation) at default threshold 0.50.
    """
    y_pred = model.predict(X)
    
    # Predict probabilities if supported
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = np.zeros(len(y))

    acc = round(float(accuracy_score(y, y_pred)), 4)
    prec = round(float(precision_score(y, y_pred, zero_division=0)), 4)
    rec = round(float(recall_score(y, y_pred, zero_division=0)), 4)
    f1 = round(float(f1_score(y, y_pred, zero_division=0)), 4)
    
    try:
        auc = round(float(roc_auc_score(y, y_prob)), 4) if len(np.unique(y)) > 1 and np.sum(y_prob) > 0 else 0.5
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y, y_pred).tolist()
    pred_pos_count = int(np.sum(y_pred))

    prob_stats = {
        "min": round(float(np.min(y_prob)), 4),
        "max": round(float(np.max(y_prob)), 4),
        "mean": round(float(np.mean(y_prob)), 4),
        "median": round(float(np.median(y_prob)), 4)
    }

    return {
        "split_name": split_name,
        "sample_count": len(y),
        "positive_count": int(np.sum(y == 1)),
        "negative_count": int(np.sum(y == 0)),
        "predicted_positive_count": pred_pos_count,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "probability_stats": prob_stats
    }

def train_real_baseline_models():
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    data_dir = os.path.join(project_root, 'data', 'processed')
    models_dir = os.path.join(project_root, 'models')
    reports_dir = os.path.join(project_root, 'reports')

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Initialize Preprocessor & Scaling Artifact
    preprocessor = RealDataPreprocessor(data_dir)
    preprocessor.process_and_fit_scaler(models_dir)

    # Load splits
    X_train_scaled, X_val_scaled, _, y_train, y_val, _ = preprocessor.get_data_for_model('logistic_regression')
    X_train_raw, X_val_raw, _, _, _, _ = preprocessor.get_data_for_model('random_forest')

    print(f"[TrainBaseline] Loaded TRAIN ({len(y_train)} rows) and VALIDATION ({len(y_val)} rows). TEST set is UNTOUCHED.")

    models_to_train = [
        {
            "name": "DummyClassifier",
            "key": "dummy",
            "model_obj": DummyClassifier(strategy="most_frequent"),
            "use_scaled": False,
            "artifact_path": os.path.join(models_dir, "dummy_classifier.pkl")
        },
        {
            "name": "LogisticRegression",
            "key": "logistic_regression",
            "model_obj": LogisticRegression(random_state=42, max_iter=1000),
            "use_scaled": True,
            "artifact_path": os.path.join(models_dir, "logistic_regression_real.pkl")
        },
        {
            "name": "RandomForest",
            "key": "random_forest",
            "model_obj": RandomForestClassifier(n_estimators=100, random_state=42),
            "use_scaled": False,
            "artifact_path": os.path.join(models_dir, "random_forest_real.pkl")
        },
        {
            "name": "XGBoost",
            "key": "xgboost",
            "model_obj": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
            "use_scaled": False,
            "artifact_path": os.path.join(models_dir, "xgboost_real.pkl")
        }
    ]

    results_report = {
        "metadata": {
            "dataset": "bottle_real_dataset",
            "train_samples": len(y_train),
            "train_positives": int(np.sum(y_train == 1)),
            "train_negatives": int(np.sum(y_train == 0)),
            "val_samples": len(y_val),
            "val_positives": int(np.sum(y_val == 1)),
            "val_negatives": int(np.sum(y_val == 0)),
            "test_used": False,
            "classification_threshold": 0.50
        },
        "models": {}
    }

    print("\n==========================================================================")
    print("      PHASE 4.4 — REAL-DATA BASELINE MODEL TRAINING & VALIDATION         ")
    print("==========================================================================")

    for item in models_to_train:
        m_name = item["name"]
        m_obj = item["model_obj"]
        use_scaled = item["use_scaled"]
        art_path = item["artifact_path"]

        X_tr = X_train_scaled if use_scaled else X_train_raw
        X_va = X_val_scaled if use_scaled else X_val_raw

        # Fit model ONLY on Training split
        m_obj.fit(X_tr, y_train)

        # Save model artifact
        joblib.dump(m_obj, art_path)
        print(f"[TrainBaseline] Saved {m_name} artifact to: {art_path}")

        # Evaluate performance on Train (diagnostic) and Validation
        train_eval = evaluate_model_performance(m_obj, X_tr, y_train, "TRAIN")
        val_eval = evaluate_model_performance(m_obj, X_va, y_val, "VALIDATION")

        # Overfitting check
        train_f1 = train_eval["f1"]
        val_f1 = val_eval["f1"]
        is_overfit_signal = (train_f1 - val_f1) > 0.25
        overfit_note = "possible overfitting signal; requires further validation" if is_overfit_signal else "no extreme overfitting gap"

        results_report["models"][item["key"]] = {
            "model_name": m_name,
            "artifact_path": art_path,
            "features_used": "scaled" if use_scaled else "raw",
            "train_metrics": train_eval,
            "validation_metrics": val_eval,
            "overfitting_diagnostic": overfit_note
        }

        print(f"\n--- {m_name.upper()} RESULTS ---")
        print(f"  Train Acc: {train_eval['accuracy']} | Train F1: {train_eval['f1']} | Train ROC-AUC: {train_eval['roc_auc']}")
        print(f"  Val   Acc: {val_eval['accuracy']} | Val   F1: {val_eval['f1']} | Val   ROC-AUC: {val_eval['roc_auc']}")
        print(f"  Val Precision: {val_eval['precision']} | Val Recall: {val_eval['recall']} | Val Pred Positives: {val_eval['predicted_positive_count']}/{len(y_val)}")
        print(f"  Val Confusion Matrix (TN, FP, FN, TP): {val_eval['confusion_matrix']}")
        print(f"  Val Probability Stats (min, max, mean, median): {val_eval['probability_stats']}")
        print(f"  Overfitting Check: {overfit_note}")

    # Save JSON validation results artifact
    json_path = os.path.join(reports_dir, "real_baseline_validation_results.json")
    with open(json_path, "w") as f:
        json.dump(results_report, f, indent=2)

    print(f"\n[TrainBaseline] Saved validation results artifact to: {json_path}")
    print("==========================================================================\n")

if __name__ == '__main__':
    train_real_baseline_models()
