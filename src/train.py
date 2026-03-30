#imports
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

from data import load_data, engineer_features, prepare_features, PROJECT_ROOT, save_training_stats


def split_data(X, y, test_size=0.2, seed=42):
    """
    Split features and labels into train and test sets
    """
    # stratify=y ensures both train and test maintain the same fraud/legit ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    print(f"\nTrain set: {X_train.shape[0]} transactions ({y_train.mean()*100:.1f}% fraud)")
    print(f"Test set:  {X_test.shape[0]} transactions ({y_test.mean()*100:.1f}% fraud)")

    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, y_train):
    """
    Train an XGBoost classifier.
    """
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
    )

    print(f"\nTraining XGBoost on {len(X_train)} transactions...")
    model.fit(X_train, y_train)
    print("Training complete.")

    return model


def evaluate(model, X_test, y_test):
    """
    Evaluate the trained model on unseen test data.
    """

    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # fraud probability
    y_pred = model.predict(X_test)                     # hard 0/1

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n{'='*50}")
    print(f"  RESULTS ON {len(y_test)} UNSEEN TRANSACTIONS")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    print(f"  AUC-ROC:   {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                      Predicted Legit    Predicted Fraud")
    print(f"  Actually Legit      {cm[0][0]:>10}         {cm[0][1]:>10}")
    print(f"  Actually Fraud      {cm[1][0]:>10}         {cm[1][1]:>10}")
    print(f"\n  True Negatives:  {cm[0][0]:>6} (legit correctly left alone)")
    print(f"  False Positives: {cm[0][1]:>6} (legit wrongly flagged)")
    print(f"  False Negatives: {cm[1][0]:>6} (fraud that slipped through)")
    print(f"  True Positives:  {cm[1][1]:>6} (fraud correctly caught)")

    # Feature importance
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': importances,
    }).sort_values('Importance', ascending=False)

    print(f"\n  Top 10 Features:")
    for _, row in importance_df.head(10).iterrows():
        bar = '█' * int(row['Importance'] * 150)
        print(f"    {row['Feature']:>15}: {row['Importance']:.4f}  {bar}")

    # Example predictions
    print(f"\n  Example Predictions:")
    print(f"  {'Actual':>8} {'Predicted':>10} {'Fraud Prob':>12}")
    for i in range(10):
        actual = "FRAUD" if y_test.iloc[i] == 1 else "LEGIT"
        predicted = "FRAUD" if y_pred[i] == 1 else "LEGIT"
        print(f"  {actual:>8} {predicted:>10} {y_pred_proba[i]:>11.1%}")


def save_model(model, path=None):
    """Save trained model to disk using pickle."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, 'models', 'xgb_model.pkl')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {path}")


if __name__ == '__main__':
    df = load_data()

    df, training_stats = engineer_features(df)

    X, y = prepare_features(df)
    print(f"\nFeatures: {X.shape[1]} columns")
    print(f"Columns: {list(X.columns)}")

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_xgboost(X_train, y_train)

    evaluate(model, X_test, y_test)

    save_model(model)
    save_training_stats(training_stats)
