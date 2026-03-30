"""
Data loading and feature engineering.
This module handles all data-related tasks, including:
- Loading the raw dataset from CSV
- Creating new features that help the model learn patterns of fraud
- Preparing the final feature matrix (X) and labels (y) for model training
- Saving/loading training statistics (means/stds) for use in the streaming pipeline
"""

import os
import pickle
import pandas as pd
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

V_COLS = [f'V{i}' for i in range(1, 29)]


def load_data(path=None):
    """
    Load the raw credit card transaction dataset.
    """
    if path is None:
        path = os.path.join(PROJECT_ROOT, 'data', 'creditcard.csv')
    df = pd.read_csv(path)

    # Drop rows with missing values: the real Kaggle dataset has some
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows with missing values")

    # Ensure Class column is integer (0 or 1)
    df['Class'] = df['Class'].astype(int)

    print(f"Loaded {df.shape[0]} transactions, {df.shape[1]} columns")
    print(f"Fraud: {(df['Class'] == 1).sum()} ({(df['Class'] == 1).mean() * 100:.1f}%)")
    return df


def engineer_features(df, stats=None):
    """
    Create new features from the raw dataset.
    
    Each feature is designed to surface a specific type of pattern
    that helps the model distinguish fraud from legitimate transactions.
    """

    df = df.copy()

    # These statistics must be identical during training and inference.
    # During training, we compute them from the dataset and save them.
    # During streaming, we load the saved values so new transactions
    # are compared against the same baseline.
    if stats is None:
        stats = {
            'amount_mean': df['Amount'].mean(),
            'amount_std': df['Amount'].std(),
            'v_means': df[V_COLS].mean().to_dict(),
            'v_stds': df[V_COLS].std().to_dict(),
        }


    #Log the amount as some are extremely large values other may be smaller values. This will help the model learn better.
    df['Amount_log'] = np.log1p(df['Amount'])


    # Standardize the amount: how does this transaction's amount compare to the user's typical amounts?
    # A value of 2.0 means "this amount is 2 standard deviations above the mean", which can be a strong signal of fraud.
    df['Amount_zscore'] = (df['Amount'] - stats['amount_mean']) / stats['amount_std']

    # Individual features might not be strong fraud signals alone, but
    # their COMBINATION can be. Multiplying two features creates a single
    # value that captures their joint behavior.
    #
    # NOTE: Ideally, you'd choose which features to combine based on
    # feature importance from an initial training run. We use V1*V2 and
    # V1*V3 because PCA orders components by variance captured (V1 holds
    # the most information), but the proper workflow is:
    #   1. Train with raw features only
    #   2. Check feature importance
    #   3. Create interactions between the top-ranked features
    #   4. Retrain and see if metrics improve
    df['V1_V2'] = df['V1'] * df['V2']
    df['V1_V3'] = df['V1'] * df['V3']


    # How far is this transaction from "normal center" across ALL features?
    # Uses Euclidean distance: sqrt(V1² + V2² + ... + V28²)
    # Fraud transactions tend to be more extreme overall — further from
    # the center of normal behavior in the 28-dimensional feature space.
    df['V_magnitude'] = np.sqrt((df[V_COLS] ** 2).sum(axis=1))

    # How many of the 28 features are outliers (> 2 std from their mean)?
    # A legit transaction might have 1 unusual feature. Fraud tends to
    # have multiple unusual features simultaneously.
    v_means = pd.Series(stats['v_means'])
    v_stds = pd.Series(stats['v_stds'])
    df['n_extreme'] = ((df[V_COLS] - v_means).abs() > 2 * v_stds).sum(axis=1)

    return df, stats


def prepare_features(df):
    """
    Separate engineered DataFrame into model inputs (X) and labels (y).
    
    Drops columns that shouldn't be used as features:
      - 'id': just an identifier, not predictive
      - 'Class': the label we're predicting (can't use the answer as input)
      - 'Amount': replaced by Amount_log and Amount_zscore
    """
    drop_cols = ['id', 'Class', 'Amount']
    # Only drop columns that exist (in case some aren't present)
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df['Class']

    return X, y


def save_training_stats(stats, path=None):
    """
    Save training statistics so the streaming pipeline can reuse them.
    """
    if path is None:
        path = os.path.join(PROJECT_ROOT, 'models', 'training_stats.pkl')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"Training stats saved to {path}")


def load_training_stats(path=None):
    """
    Load training statistics for use in the streaming pipeline.
    """
    if path is None:
        path = os.path.join(PROJECT_ROOT, 'models', 'training_stats.pkl')
    with open(path, 'rb') as f:
        stats = pickle.load(f)
    return stats
