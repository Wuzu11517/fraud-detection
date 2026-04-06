"""
Dashboard — Flask web app for monitoring and interacting with the
fraud detection pipeline.

Features:
  - Live feed of scored transactions
  - Aggregated stats (catch rate, false positives, model performance)
  - Interactive form to submit transactions and see scores in real-time

The dashboard loads both trained models on startup so it can score
transactions directly — no need for the separate consumer process
when using the submission form.
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
import torch
import redis
from flask import Flask, render_template, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import V_COLS, PROJECT_ROOT, load_training_stats
from autoencoder import Autoencoder, compute_reconstruction_error
from consumer import build_feature_vector, score_transaction, make_decision
from feature_store import get_user_aggregate, update_user_aggregate, compute_user_features

app = Flask(__name__)

models = {}

def load_all_models():
    """Load all models and stats into memory on startup."""
    global models

    # XGBoost
    xgb_path = os.path.join(PROJECT_ROOT, 'models', 'xgb_model.pkl')
    with open(xgb_path, 'rb') as f:
        models['xgb'] = pickle.load(f)

    # Scaler
    scaler_path = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        models['scaler'] = pickle.load(f)

    # Autoencoder
    ae_path = os.path.join(PROJECT_ROOT, 'models', 'autoencoder.pth')
    ae_model = Autoencoder(input_dim=34)
    ae_model.load_state_dict(torch.load(ae_path, weights_only=True))
    ae_model.eval()
    models['ae'] = ae_model

    # Training stats
    models['training_stats'] = load_training_stats()

    # AE error stats
    ae_stats_path = os.path.join(PROJECT_ROOT, 'models', 'ae_stats.pkl')
    with open(ae_stats_path, 'rb') as f:
        models['ae_stats'] = pickle.load(f)

    # Load a sample of real transactions for the submission form.
    # Sampling from real data preserves the correlations between features
    # that the autoencoder learned. Generating features independently
    # breaks those correlations and makes everything look anomalous.
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'creditcard.csv')
    df = pd.read_csv(csv_path).dropna()
    df['Class'] = df['Class'].astype(int)
    models['legit_sample'] = df[df['Class'] == 0].reset_index(drop=True)
    models['fraud_sample'] = df[df['Class'] == 1].reset_index(drop=True)

    print("Models loaded successfully")


def get_redis():
    host = os.environ.get('REDIS_HOST', 'localhost')
    return redis.Redis(host=host, port=6379, decode_responses=True)

@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/transactions')
def get_transactions():
    """Return the most recent scored transactions."""
    r = get_redis()
    raw = r.lrange('results', 0, 49)
    transactions = [json.loads(item) for item in raw]
    return jsonify(transactions)


@app.route('/api/stats')
def get_stats():
    """Return aggregated statistics."""
    r = get_redis()
    raw = r.lrange('results', 0, 499)

    if not raw:
        return jsonify({
            'total': 0, 'blocked': 0, 'reviewed': 0, 'allowed': 0,
            'actual_fraud': 0, 'fraud_caught': 0, 'false_blocks': 0,
            'catch_rate': 0, 'false_positive_rate': 0,
            'avg_xgb_score_fraud': 0, 'avg_xgb_score_legit': 0,
            'avg_ae_score_fraud': 0, 'avg_ae_score_legit': 0,
        })

    results = [json.loads(item) for item in raw]

    total = len(results)
    blocked = sum(1 for r in results if r['decision'] == 'BLOCK')
    reviewed = sum(1 for r in results if r['decision'] == 'REVIEW')
    allowed = sum(1 for r in results if r['decision'] == 'ALLOW')
    actual_fraud = sum(1 for r in results if r['actual'] == 1)
    actual_legit = sum(1 for r in results if r['actual'] == 0)
    fraud_caught = sum(1 for r in results if r['actual'] == 1 and r['decision'] == 'BLOCK')
    false_blocks = sum(1 for r in results if r['actual'] == 0 and r['decision'] == 'BLOCK')

    catch_rate = (fraud_caught / actual_fraud * 100) if actual_fraud > 0 else 0
    false_positive_rate = (false_blocks / actual_legit * 100) if actual_legit > 0 else 0

    fraud_results = [r for r in results if r['actual'] == 1]
    legit_results = [r for r in results if r['actual'] == 0]

    avg_xgb_fraud = sum(r['xgb_score'] for r in fraud_results) / len(fraud_results) if fraud_results else 0
    avg_xgb_legit = sum(r['xgb_score'] for r in legit_results) / len(legit_results) if legit_results else 0
    avg_ae_fraud = sum(r['ae_score'] for r in fraud_results) / len(fraud_results) if fraud_results else 0
    avg_ae_legit = sum(r['ae_score'] for r in legit_results) / len(legit_results) if legit_results else 0

    return jsonify({
        'total': total, 'blocked': blocked, 'reviewed': reviewed,
        'allowed': allowed, 'actual_fraud': actual_fraud,
        'fraud_caught': fraud_caught, 'false_blocks': false_blocks,
        'catch_rate': round(catch_rate, 1),
        'false_positive_rate': round(false_positive_rate, 2),
        'avg_xgb_score_fraud': round(avg_xgb_fraud, 3),
        'avg_xgb_score_legit': round(avg_xgb_legit, 3),
        'avg_ae_score_fraud': round(avg_ae_fraud, 3),
        'avg_ae_score_legit': round(avg_ae_legit, 3),
    })


@app.route('/api/score', methods=['POST'])
def score():
    data = request.get_json()
    r = get_redis()

    amount = float(data.get('amount', 50.0))
    user_id = str(data.get('user_id', np.random.randint(1, 1001)))
    profile = data.get('profile', 'random')  # 'legit', 'fraud', or 'random'

    # Generate V features based on profile
    if 'features' in data:
        features = {k: float(v) for k, v in data['features'].items()}
    else:
        features = generate_features(profile)

    # Look up user aggregate
    user_agg = get_user_aggregate(r, user_id)
    user_features = compute_user_features(user_agg, amount)

    # Build feature vector and score
    feature_vector = build_feature_vector(features, amount, models['training_stats'])
    xgb_score, ae_score, combined_score = score_transaction(
        models['xgb'], models['ae'], models['scaler'],
        models['ae_stats'], feature_vector
    )

    decision, risk = make_decision(combined_score)

    # Update user aggregate
    update_user_aggregate(r, user_id, amount)

    # Store in results for the live feed
    result = {
        'msg_id': f'manual-{int(time.time()*1000)}',
        'user_id': user_id,
        'amount': round(float(amount), 2),
        'xgb_score': round(float(xgb_score), 3),
        'ae_score': round(float(ae_score), 3),
        'combined_score': round(float(combined_score), 3),
        'decision': decision,
        'risk': risk,
        'actual': -1,  # -1 means "submitted manually, no ground truth"
        'timestamp': time.time(),
    }
    r.lpush('results', json.dumps(result))
    r.ltrim('results', 0, 499)

    return jsonify(result)


def generate_features(profile='random'):
    """
    Sample V1-V28 features from a real transaction in the dataset.
    
    Instead of generating features independently (which breaks the
    correlations the autoencoder learned), we pick an actual transaction
    from the dataset. This preserves the natural feature relationships
    and gives realistic scores.
    
    'legit' samples from legitimate transactions,
    'fraud' samples from fraud transactions,
    'random' is a coin flip between the two.
    """
    if profile == 'random':
        profile = np.random.choice(['legit', 'fraud'])

    if profile == 'fraud':
        sample_df = models['fraud_sample']
    else:
        sample_df = models['legit_sample']

    # Pick a random row
    row = sample_df.sample(1).iloc[0]

    features = {}
    for i in range(1, 29):
        features[f'V{i}'] = float(row[f'V{i}'])

    return features

if __name__ == '__main__':
    load_all_models()
    print("\nDashboard running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)