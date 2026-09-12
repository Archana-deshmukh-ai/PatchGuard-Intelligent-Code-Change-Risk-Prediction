import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
sys.path.insert(0, src_dir)

from generate_dataset import generate_synthetic_dataset

def run_controlled_diagnostic():
    # 1. Load dataset with 10 features generated under seed 42
    df = generate_synthetic_dataset(num_samples=300, random_state=42)
    
    feature_cols_10 = [
        'lines_added', 'lines_deleted', 'code_churn', 'files_changed', 'functions_changed',
        'num_directories_touched', 'is_test_file_modified', 'avg_lines_changed_per_file',
        'max_lines_changed_in_single_file', 'num_source_files_changed'
    ]
    feature_cols_5 = feature_cols_10[:5]
    target_col = 'bug_introduced'
    
    X_full = df[feature_cols_10]
    y_full = df[target_col]
    
    # Stratified Train/Test Split (Same indices for both models!)
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.20, random_state=42, stratify=y_full
    )
    
    # 5-Feature Subset
    X_train_5 = X_train_full[feature_cols_5]
    X_test_5 = X_test_full[feature_cols_5]
    
    # Scaling
    scaler_5 = StandardScaler()
    X_train_5_scaled = scaler_5.fit_transform(X_train_5)
    X_test_5_scaled = scaler_5.transform(X_test_5)
    
    scaler_10 = StandardScaler()
    X_train_10_scaled = scaler_10.fit_transform(X_train_full)
    X_test_10_scaled = scaler_10.transform(X_test_full)
    
    # Train Models
    model_5 = LogisticRegression(random_state=42, solver='lbfgs')
    model_5.fit(X_train_5_scaled, y_train)
    
    model_10 = LogisticRegression(random_state=42, solver='lbfgs')
    model_10.fit(X_train_10_scaled, y_train)
    
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train_10_scaled, y_train)
    
    # Evaluation on Test Set (Default threshold 0.5)
    prob_5 = model_5.predict_proba(X_test_5_scaled)[:, 1]
    pred_5 = (prob_5 >= 0.5).astype(int)
    
    prob_10 = model_10.predict_proba(X_test_10_scaled)[:, 1]
    pred_10 = (prob_10 >= 0.5).astype(int)
    
    pred_dummy = dummy.predict(X_test_10_scaled)
    
    print("================================================================")
    print("         CONTROLLED EXPERIMENT DIAGNOSTIC RESULTS               ")
    print("================================================================")
    print(f"Total Dataset: 300 samples (Clean: {(y_full==0).sum()}, Bug-prone: {(y_full==1).sum()})")
    print(f"Test Split: 60 samples (Clean: {(y_test==0).sum()}, Bug-prone: {(y_test==1).sum()})\n")
    
    def print_metrics(name, y_true, y_pred, y_prob):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_prob)) > 1 else 0.5
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print(f"--- {name} ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        print(f"Confusion Matrix: [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")
        print(f"Positive Predictions (>= 0.5): {y_pred.sum()} / {len(y_pred)}\n")
        return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc, "tn": tn, "fp": fp, "fn": fn, "tp": tp}
        
    m_dummy = print_metrics("Dummy Majority Classifier", y_test, pred_dummy, np.zeros(len(y_test)))
    m_5 = print_metrics("Model A: 5-Feature Logistic Regression (Controlled)", y_test, pred_5, prob_5)
    m_10 = print_metrics("Model B: 10-Feature Logistic Regression (Controlled)", y_test, pred_10, prob_10)
    
    print("----------------------------------------------------------------")
    print("PROBABILITY DISTRIBUTION INSPECTION (Test Set)")
    print("----------------------------------------------------------------")
    print(f"Model A (5-Feature)  Probabilities -> Min: {prob_5.min():.4f}, Mean: {prob_5.mean():.4f}, Max: {prob_5.max():.4f}, Median: {np.median(prob_5):.4f}")
    print(f"Model B (10-Feature) Probabilities -> Min: {prob_10.min():.4f}, Mean: {prob_10.mean():.4f}, Max: {prob_10.max():.4f}, Median: {np.median(prob_10):.4f}\n")
    
    print("----------------------------------------------------------------")
    print("THRESHOLD SENSITIVITY DIAGNOSTIC (Model B: 10-Feature)")
    print("----------------------------------------------------------------")
    print(f"{'Threshold':<10} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'TN':<4}")
    print("-" * 60)
    for thresh in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        t_pred = (prob_10 >= thresh).astype(int)
        t_prec = precision_score(y_test, t_pred, zero_division=0)
        t_rec = recall_score(y_test, t_pred, zero_division=0)
        t_f1 = f1_score(y_test, t_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, t_pred).ravel()
        print(f"{thresh:<10.2f} | {t_prec:<8.4f} | {t_rec:<8.4f} | {t_f1:<8.4f} | {tp:<4} | {fp:<4} | {fn:<4} | {tn:<4}")

if __name__ == '__main__':
    run_controlled_diagnostic()
