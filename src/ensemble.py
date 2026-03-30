"""
Ensemble scoring — combining XGBoost and autoencoder.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from data import load_data, engineer_features, prepare_features, PROJECT_ROOT
from autoencoder import Autoencoder, compute_reconstruction_error


def load_models():
    """
    Load the trained XGBoost model, autoencoder, and scaler from disk.
    """
    # Load XGBoost
    xgb_path = os.path.join(PROJECT_ROOT, 'models', 'xgb_model.pkl')
    with open(xgb_path, 'rb') as f:
        xgb_model = pickle.load(f)

    # Load autoencoder
    scaler_path = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    ae_path = os.path.join(PROJECT_ROOT, 'models', 'autoencoder.pth')
    ae_model = Autoencoder(input_dim=34)  # must match training architecture
    ae_model.load_state_dict(torch.load(ae_path, weights_only=True))
    ae_model.eval()  # set to evaluation mode (disables training behaviors)

    print("Loaded XGBoost model, autoencoder, and scaler")
    return xgb_model, ae_model, scaler


def normalize_errors(errors):
    """
    Normalize reconstruction errors to the 0-1 range.
    """
    min_err = errors.min()
    max_err = errors.max()

    # Avoid division by zero if all errors are identical
    if max_err == min_err:
        return np.zeros_like(errors)

    return (errors - min_err) / (max_err - min_err)


def ensemble_score(xgb_proba, ae_errors, xgb_weight=0.7, ae_weight=0.3):
    """
    Combine XGBoost probability and autoencoder error into a single score.
    The weights can be tuned based on validation performance.
    """
    ae_normalized = normalize_errors(ae_errors)
    combined = (xgb_weight * xgb_proba) + (ae_weight * ae_normalized)
    return combined


def evaluate_ensemble(combined_scores, y_true, threshold=0.5):
    """
    Evaluate the ensemble's performance.
    """
    predictions = (combined_scores >= threshold).astype(int)

    acc = accuracy_score(y_true, predictions)
    prec = precision_score(y_true, predictions)
    rec = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    auc = roc_auc_score(y_true, combined_scores)

    cm = confusion_matrix(y_true, predictions)

    print(f"\n{'='*50}")
    print(f"  ENSEMBLE RESULTS (threshold={threshold})")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    print(f"  AUC-ROC:   {auc:.4f}")

    print(f"\n  Confusion Matrix:")
    print(f"                      Predicted Legit    Predicted Fraud")
    print(f"  Actually Legit      {cm[0][0]:>10}         {cm[0][1]:>10}")
    print(f"  Actually Fraud      {cm[1][0]:>10}         {cm[1][1]:>10}")

    print(f"\n  True Negatives:  {cm[0][0]:>6} (legit correctly left alone)")
    print(f"  False Positives: {cm[0][1]:>6} (legit wrongly flagged)")
    print(f"  False Negatives: {cm[1][0]:>6} (fraud that slipped through)")
    print(f"  True Positives:  {cm[1][1]:>6} (fraud correctly caught)")


def show_threshold_comparison(combined_scores, y_true):
    """
    Show how different thresholds affect precision and recall.
    """
    print(f"\n{'='*50}")
    print(f"  THRESHOLD COMPARISON")
    print(f"{'='*50}")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Flagged':>10}")

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        preds = (combined_scores >= threshold).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        flagged = preds.sum()
        print(f"  {threshold:>10.1f} {prec*100:>9.2f}% {rec*100:>9.2f}% {f1*100:>9.2f}% {flagged:>10}")

if __name__ == '__main__':
    df = load_data()
    df, _ = engineer_features(df)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    xgb_model, ae_model, scaler = load_models()

    print("\nScoring with XGBoost...")
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

    print("Scoring with autoencoder...")
    ae_errors = compute_reconstruction_error(ae_model, scaler, X_test)

    print(f"\n  XGBoost AUC:      {roc_auc_score(y_test, xgb_proba):.4f}")
    print(f"  Autoencoder AUC:  {roc_auc_score(y_test, ae_errors):.4f}")

    combined = ensemble_score(xgb_proba, ae_errors)
    print(f"  Ensemble AUC:     {roc_auc_score(y_test, combined):.4f}")

    evaluate_ensemble(combined, y_test.values)

    show_threshold_comparison(combined, y_test.values)
