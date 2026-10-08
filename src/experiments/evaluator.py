"""
Phase 8 Experiment Evaluator.
Computes comprehensive, leakage-safe evaluation metrics for candidate ML models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS, TARGET_COLUMN, THRESHOLD_SWEEP, PRODUCTION_THRESHOLD

def compute_detailed_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Calculates classification, discrimination, calibration, and error matrix metrics at a specific decision threshold.
    """
    y_true_arr = np.asarray(y_true)
    y_pred = (y_prob >= threshold).astype(int)

    total_samples = len(y_true_arr)
    pos_samples = int(np.sum(y_true_arr == 1))
    neg_samples = int(np.sum(y_true_arr == 0))
    pos_rate = round(pos_samples / total_samples, 4) if total_samples > 0 else 0.0

    acc = round(float(accuracy_score(y_true_arr, y_pred)), 4)
    prec = round(float(precision_score(y_true_arr, y_pred, zero_division=0)), 4)
    rec = round(float(recall_score(y_true_arr, y_pred, zero_division=0)), 4)
    f1 = round(float(f1_score(y_true_arr, y_pred, zero_division=0)), 4)

    # ROC-AUC & PR-AUC (Average Precision)
    if len(np.unique(y_true_arr)) > 1 and np.max(y_prob) > np.min(y_prob):
        roc_auc = round(float(roc_auc_score(y_true_arr, y_prob)), 4)
        pr_auc = round(float(average_precision_score(y_true_arr, y_prob)), 4)
    else:
        roc_auc = 0.5000
        pr_auc = pos_rate

    # Brier score (Calibration metric)
    brier = round(float(brier_score_loss(y_true_arr, y_prob)), 4)

    # Confusion matrix
    cm = confusion_matrix(y_true_arr, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    pred_pos_count = int(np.sum(y_pred))
    pred_pos_rate = round(pred_pos_count / total_samples, 4) if total_samples > 0 else 0.0

    return {
        "decision_threshold": round(float(threshold), 4),
        "total_samples": total_samples,
        "positive_samples": pos_samples,
        "negative_samples": neg_samples,
        "positive_prevalence": pos_rate,
        "predicted_positives": pred_pos_count,
        "predicted_positive_rate": pred_pos_rate,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }

def sweep_validation_thresholds(
    y_true: pd.Series,
    y_prob: np.ndarray,
    thresholds: List[float] = THRESHOLD_SWEEP
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Evaluates model performance across a spectrum of decision thresholds on the Validation set
    and identifies explicit operating points (max_f1, high_recall, balanced, high_precision, production_0_35).
    """
    results = [compute_detailed_metrics(y_true, y_prob, t) for t in thresholds]
    df_res = pd.DataFrame(results)

    # 1. Max F1 Operating Point
    max_f1_idx = df_res['f1'].idxmax()
    max_f1_op = df_res.loc[max_f1_idx].to_dict()

    # 2. High Recall Operating Point (Recall >= 0.70 with max F1)
    high_rec_sub = df_res[df_res['recall'] >= 0.70]
    if not high_rec_sub.empty:
        high_rec_op = high_rec_sub.loc[high_rec_sub['f1'].idxmax()].to_dict()
    else:
        high_rec_op = df_res.loc[df_res['recall'].idxmax()].to_dict()

    # 3. Balanced Operating Point (Minimal |Precision - Recall|)
    df_res['diff'] = (df_res['precision'] - df_res['recall']).abs()
    valid_diffs = df_res[(df_res['precision'] > 0) & (df_res['recall'] > 0)]
    if not valid_diffs.empty:
        balanced_op = valid_diffs.loc[valid_diffs['diff'].idxmin()].to_dict()
    else:
        balanced_op = max_f1_op
    df_res.drop(columns=['diff'], inplace=True, errors='ignore')

    # 4. High Precision Operating Point (Precision >= 0.60 with max F1)
    high_prec_sub = df_res[df_res['precision'] >= 0.60]
    if not high_prec_sub.empty:
        high_prec_op = high_prec_sub.loc[high_prec_sub['f1'].idxmax()].to_dict()
    else:
        nonzero_prec = df_res[df_res['predicted_positives'] > 0]
        high_prec_op = nonzero_prec.loc[nonzero_prec['precision'].idxmax()].to_dict() if not nonzero_prec.empty else max_f1_op

    # 5. Fixed Baseline Threshold 0.35 Operating Point
    prod_035_op = compute_detailed_metrics(y_true, y_prob, PRODUCTION_THRESHOLD)

    operating_points = {
        "max_f1": max_f1_op,
        "high_recall": high_rec_op,
        "balanced": balanced_op,
        "high_precision": high_prec_op,
        "production_0_35": prod_035_op
    }

    return results, operating_points

class ModelTrainerEvaluator:
    """
    Leakage-safe model trainer and evaluator for Phase 8.
    Enforces Train-only scaling and handles raw vs scaled data routing for linear and tree models.
    """
    def __init__(self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

        self.X_train_raw = df_train[FEATURE_COLUMNS].copy()
        self.y_train = df_train[TARGET_COLUMN].copy()

        self.X_val_raw = df_val[FEATURE_COLUMNS].copy()
        self.y_val = df_val[TARGET_COLUMN].copy()

        self.X_test_raw = df_test[FEATURE_COLUMNS].copy()
        self.y_test = df_test[TARGET_COLUMN].copy()

        # Fit StandardScaler ONLY on X_train (CRITICAL LEAKAGE PREVENTION RULE)
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train_raw)

        self.X_train_scaled = self.scaler.transform(self.X_train_raw)
        self.X_val_scaled = self.scaler.transform(self.X_val_raw)
        self.X_test_scaled = self.scaler.transform(self.X_test_raw)

    def prepare_data(self, model_type: str) -> Tuple[Any, Any, Any]:
        """
        Routes scaled feature arrays to linear models and raw feature DataFrames to tree models.
        """
        if model_type.lower() in ['logistic_regression', 'logistic', 'linear']:
            return self.X_train_scaled, self.X_val_scaled, self.X_test_scaled
        else:
            return self.X_train_raw, self.X_val_raw, self.X_test_raw
