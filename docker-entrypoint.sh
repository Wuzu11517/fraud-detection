#!/bin/bash
set -e

echo "=================================="
echo "  Fraud Detection System"
echo "=================================="

# Generate data if no CSV exists
if [ ! -f data/creditcard.csv ]; then
    echo ""
    echo "[1/4] Generating synthetic data..."
    python src/generate_data.py
else
    echo ""
    echo "[1/4] Dataset found"
fi

# Train models if they don't exist
if [ ! -f models/xgb_model.pkl ]; then
    echo ""
    echo "[2/4] Training XGBoost..."
    python src/train.py
else
    echo ""
    echo "[2/4] XGBoost model found"
fi

if [ ! -f models/autoencoder.pth ]; then
    echo ""
    echo "[3/4] Training Autoencoder..."
    python src/autoencoder.py
else
    echo ""
    echo "[3/4] Autoencoder model found"
fi

echo ""
echo "[4/4] Starting dashboard..."
echo ""
echo "  Open http://localhost:5000 in your browser"
echo "  Submit transactions via the form or the API"
echo ""
echo "=================================="

python src/dashboard.py
