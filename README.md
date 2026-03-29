# Distributed Fraud Detection System

Real-time credit card fraud detection using an ensemble of XGBoost and an autoencoder,
served over a Redis Streams pipeline.

## Setup

```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate it
#    macOS/Linux:
source venv/bin/activate
#    Windows:
#    venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your dataset
#    Download from: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023
#    Save the CSV as: data/creditcard.csv
#
#    OR generate synthetic data:
python src/generate_data.py
```

## Project Structure

```
fraud-detection/
├── data/                    # Dataset lives here (gitignored)
│   └── creditcard.csv
├── models/                  # Trained models saved here (gitignored)
│   ├── xgb_model.pkl
│   ├── autoencoder.pth
│   └── scaler.pkl
├── src/
│   ├── data.py              # Data loading + feature engineering
│   ├── train.py             # Train/test split, XGBoost training, evaluation
│   ├── autoencoder.py       # Autoencoder model, training, anomaly scoring
│   ├── ensemble.py          # Combines both models into a final fraud score
│   ├── feature_store.py     # Per-user aggregates in Redis
│   ├── producer.py          # Pushes transactions into Redis Stream
│   ├── consumer.py          # Reads stream, scores transactions in real-time
│   └── generate_data.py     # Generate synthetic data for testing
├── requirements.txt
├── .gitignore
└── README.md
```

## Usage

```bash
# Step 1: Generate synthetic data (or use real Kaggle data)
python src/generate_data.py

# Step 2: Train and evaluate XGBoost
python src/train.py

# Step 3: Train and evaluate autoencoder
python src/autoencoder.py

# Step 4: Run the ensemble (combines both models)
python src/ensemble.py

# Step 5: Start Redis, then run the streaming pipeline
redis-server --daemonize yes
python src/producer.py    # terminal 1: push transactions
python src/consumer.py    # terminal 2: score in real-time
```

## Dataset

Using the [Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
from Kaggle. 550,000+ transactions from European cardholders with PCA-anonymized features.
