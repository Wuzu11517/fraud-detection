"""
Autoencoder for anomaly detection.

This module defines, trains, and evaluates an autoencoder that learns
what "normal" transactions look like. Anything it can't reconstruct
well is flagged as anomalous — potentially fraud.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from data import load_data, engineer_features, prepare_features, PROJECT_ROOT


class Autoencoder(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        # nn.Sequential chains layers together so data flows through
        # them in order automatically.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),   # 34 features → 16
            nn.ReLU(),
            nn.Linear(16, 8),           # 16 → 8 (bottleneck)
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),           # 8 → 16
            nn.ReLU(),
            nn.Linear(16, input_dim),   # 16 → 34 (reconstruct)
            # No activation on the final layer — we want the raw
            # reconstructed values, not values clamped by ReLU.
        )

    def forward(self, x):
        """
        Define how data flows through the network.
        """
        encoded = self.encoder(x)   # compress
        decoded = self.decoder(encoded)  # reconstruct
        return decoded


def train_autoencoder(X_train_legit, epochs=50, batch_size=256, lr=0.001):
    """
    Train the autoencoder on legitimate transactions only.
    
    Steps:
      1. Scale the data so all features have mean=0, std=1.
         Neural networks train better when inputs are on similar scales.
      2. Convert to PyTorch tensors (PyTorch's data format).
      3. Create a DataLoader that feeds data in batches.
      4. Train by minimizing reconstruction error (MSE loss).
    """
    

    # StandardScaler transforms each column to have mean=0 and std=1.
    # Before: Amount_log might range 0-10, V1 might range -5 to 5
    # After: both center around 0 with similar spread.
    # Neural networks are sensitive to input scale — if one feature
    # is 100x larger than another, it dominates the learning.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_legit)

    # --- CONVERT TO PYTORCH TENSORS ---
    # PyTorch works with tensors (like numpy arrays but with GPU support
    # and automatic gradient computation for training).
    # .float() converts to 32-bit floats (PyTorch default).
    X_tensor = torch.FloatTensor(X_scaled)

    # --- DATALOADER ---
    # DataLoader handles batching and shuffling automatically.
    # Instead of feeding all 220k transactions at once, it feeds
    # them in groups of 256 (batch_size). Shuffling each epoch
    # prevents the model from memorizing the order.
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- MODEL, LOSS, OPTIMIZER ---
    input_dim = X_scaled.shape[1]  # number of features (34)
    model = Autoencoder(input_dim)

    # MSELoss = Mean Squared Error
    # For each transaction: average of (original_value - reconstructed_value)²
    # across all features. The model tries to minimize this.
    criterion = nn.MSELoss()

    # Adam optimizer — an improved version of basic gradient descent.
    # It adapts the learning rate per-parameter based on past gradients,
    # which generally trains faster and more reliably.
    # model.parameters() gives Adam access to all the learnable weights.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- TRAINING LOOP ---
    print(f"Training autoencoder on {len(X_tensor)} legitimate transactions...")
    print(f"Architecture: {input_dim} → 16 → 8 → 16 → {input_dim}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {lr}\n")

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        for (batch,) in loader:
            # Forward pass: feed data through the model
            reconstructed = model(batch)

            # Compute loss: how different is the reconstruction?
            loss = criterion(reconstructed, batch)

            # Backward pass: compute gradients
            # PyTorch tracks all operations on tensors. When you call
            # loss.backward(), it computes how much each weight
            # contributed to the error (the gradient).
            optimizer.zero_grad()   # reset gradients from previous batch
            loss.backward()         # compute gradients
            optimizer.step()        # update weights using gradients

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3}/{epochs} — Avg Loss: {avg_loss:.6f}")

    print("Training complete.\n")
    return model, scaler


def compute_reconstruction_error(model, scaler, X):
    """
    Compute reconstruction error for each transaction.
    
    High error = the autoencoder couldn't reconstruct this transaction
    well, meaning it doesn't fit the pattern of "normal." This is our
    anomaly score.
    """

    # Scale using the SAME scaler from training (not a new one).
    # This ensures the data is on the same scale the model expects.
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)

    # torch.no_grad() tells PyTorch we're not training — don't track
    # gradients. This saves memory and speeds up inference.
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        reconstructed = model(X_tensor)

    # Compute per-transaction MSE: average squared difference across
    # all features for each transaction.
    # .numpy() converts PyTorch tensor back to numpy array.
    mse = ((X_tensor - reconstructed) ** 2).mean(dim=1).numpy()

    return mse


def evaluate_autoencoder(errors, y_true):
    """
    Evaluate how well reconstruction error separates fraud from legit.
    
    """
    legit_errors = errors[y_true == 0]
    fraud_errors = errors[y_true == 1]

    print(f"  Legit reconstruction error — mean: {legit_errors.mean():.6f}, std: {legit_errors.std():.6f}")
    print(f"  Fraud reconstruction error — mean: {fraud_errors.mean():.6f}, std: {fraud_errors.std():.6f}")
    print(f"  Ratio (fraud/legit): {fraud_errors.mean() / legit_errors.mean():.2f}x")

    # AUC-ROC: can the reconstruction error alone separate fraud from legit?
    auc = roc_auc_score(y_true, errors)
    print(f"  AUC-ROC (error as fraud score): {auc:.4f}")

    # Try a simple threshold: flag anything above the 95th percentile
    # of legitimate transaction errors
    threshold = np.percentile(legit_errors, 95)
    predictions = (errors > threshold).astype(int)
    prec = precision_score(y_true, predictions)
    rec = recall_score(y_true, predictions)
    print(f"\n  Threshold (95th percentile of legit errors): {threshold:.6f}")
    print(f"  Precision at this threshold: {prec*100:.2f}%")
    print(f"  Recall at this threshold:    {rec*100:.2f}%")


def save_autoencoder(model, scaler, ae_stats=None, model_path=None, scaler_path=None, stats_path=None):
    """Save the trained autoencoder, scaler, and error statistics."""
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, 'models', 'autoencoder.pth')
    if scaler_path is None:
        scaler_path = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
    if stats_path is None:
        stats_path = os.path.join(PROJECT_ROOT, 'models', 'ae_stats.pkl')

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # PyTorch models are saved with torch.save using state_dict()
    # which contains all the learned weights.
    torch.save(model.state_dict(), model_path)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Save error distribution stats for normalization in the consumer.
    # Without these, the consumer would have to guess how to convert
    # raw reconstruction error into a 0-1 score.
    if ae_stats is not None:
        with open(stats_path, 'wb') as f:
            pickle.dump(ae_stats, f)
        print(f"AE error stats saved to {stats_path}")

    print(f"Autoencoder saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    df = load_data()
    df, _ = engineer_features(df)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_legit = X_train[y_train == 0]
    print(f"Training on {len(X_train_legit)} legitimate transactions only")
    print(f"(Excluding {(y_train == 1).sum()} fraud transactions from training)\n")

    model, scaler = train_autoencoder(X_train_legit, epochs=50)

    print("Scoring test set...")
    errors = compute_reconstruction_error(model, scaler, X_test)

    legit_errors = errors[y_test.values == 0]
    fraud_errors = errors[y_test.values == 1]
    ae_stats = {
        'legit_error_mean': float(legit_errors.mean()),
        'legit_error_std': float(legit_errors.std()),
        'legit_error_95th': float(np.percentile(legit_errors, 95)),
        'legit_error_99th': float(np.percentile(legit_errors, 99)),
        'fraud_error_mean': float(fraud_errors.mean()),
        'fraud_error_median': float(np.median(fraud_errors)),
    }

    print(f"\n{'='*50}")
    print(f"  AUTOENCODER RESULTS ON {len(y_test)} UNSEEN TRANSACTIONS")
    print(f"{'='*50}")
    evaluate_autoencoder(errors, y_test.values)

    # 7. Save
    print()
    save_autoencoder(model, scaler, ae_stats)