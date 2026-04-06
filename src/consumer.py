import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import V_COLS, PROJECT_ROOT, load_training_stats
from autoencoder import Autoencoder, compute_reconstruction_error
from feature_store import get_user_aggregate, update_user_aggregate, compute_user_features
from producer import STREAM_NAME, GROUP_NAME, get_redis, setup_stream


def load_models():
    """
    Load XGBoost, autoencoder, scaler, and stats from disk.
    
    This happens once when the consumer starts up. The models
    stay in memory for fast scoring — loading from disk on every
    transaction would be far too slow.
    """
    # XGBoost
    xgb_path = os.path.join(PROJECT_ROOT, 'models', 'xgb_model.pkl')
    with open(xgb_path, 'rb') as f:
        xgb_model = pickle.load(f)

    # Scaler
    scaler_path = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Autoencoder
    ae_path = os.path.join(PROJECT_ROOT, 'models', 'autoencoder.pth')
    ae_model = Autoencoder(input_dim=34)
    ae_model.load_state_dict(torch.load(ae_path, weights_only=True))
    ae_model.eval()

    # Training stats (amount_mean, amount_std, v_means, v_stds)
    training_stats = load_training_stats()

    # Autoencoder error stats (for normalizing reconstruction error)
    ae_stats_path = os.path.join(PROJECT_ROOT, 'models', 'ae_stats.pkl')
    with open(ae_stats_path, 'rb') as f:
        ae_stats = pickle.load(f)

    print("Loaded: XGBoost, Autoencoder, Scaler, Training stats, AE error stats")
    return {
        'xgb_model': xgb_model,
        'ae_model': ae_model,
        'scaler': scaler,
        'training_stats': training_stats,
        'ae_stats': ae_stats,
    }

def parse_message(msg_data):
    """
    Convert a Redis Stream message back into usable data.
    
    Redis stores everything as strings, so we convert back to
    the proper types. The message comes in as a dictionary like:
    {'user_id': '103', 'amount': '49.99', 'V1': '-1.34', ...}
    """
    user_id = msg_data['user_id']
    amount = float(msg_data['amount'])
    actual_class = int(msg_data['class'])

    # Extract V features
    features = {}
    for i in range(1, 29):
        features[f'V{i}'] = float(msg_data[f'V{i}'])

    return user_id, amount, features, actual_class


def build_feature_vector(features, amount, training_stats):
    """
    Build the full feature vector that our models expect.
    
    This replicates the same feature engineering from data.py,
    but for a single transaction instead of a whole DataFrame.
    The models were trained on these exact features, so we must
    produce them in the same way.
    """
    row = dict(features)  # copy V1-V28
    row['Amount'] = amount

    # Create a single-row DataFrame
    df = pd.DataFrame([row])

    # Apply the same feature engineering as training
    df['Amount_log'] = np.log1p(df['Amount'])

    # Z-score using the SAME mean/std from training
    df['Amount_zscore'] = (df['Amount'] - training_stats['amount_mean']) / training_stats['amount_std']

    # Interaction features
    df['V1_V2'] = df['V1'] * df['V2']
    df['V1_V3'] = df['V1'] * df['V3']

    # Magnitude
    v_cols = [f'V{i}' for i in range(1, 29)]
    df['V_magnitude'] = np.sqrt((df[v_cols] ** 2).sum(axis=1))

    # Extreme count using saved training stats
    v_means = pd.Series(training_stats['v_means'])
    v_stds = pd.Series(training_stats['v_stds'])
    df['n_extreme'] = ((df[v_cols] - v_means).abs() > 2 * v_stds).sum(axis=1)

    # Select the same columns in the same order as training
    feature_cols = v_cols + ['Amount_log', 'Amount_zscore', 'V1_V2', 'V1_V3', 'V_magnitude', 'n_extreme']
    return df[feature_cols]


def score_transaction(xgb_model, ae_model, scaler, ae_stats, feature_vector):
    # XGBoost: get fraud probability
    xgb_score = xgb_model.predict_proba(feature_vector)[:, 1][0]

    # Autoencoder: get reconstruction error
    ae_error = compute_reconstruction_error(ae_model, scaler, feature_vector)[0]

    # Normalize autoencoder error to 0-1 using the actual error distribution.
    # We map the range [legit_mean, fraud_median] to [0, 1].
    # Errors at the legit mean score ~0 (normal), errors at the fraud
    # median score ~1 (anomalous). This uses the full dynamic range
    # instead of clamping everything above a low threshold to 1.0.
    low = ae_stats['legit_error_mean']
    high = ae_stats['fraud_error_median']
    if high > low:
        ae_score = (ae_error - low) / (high - low)
        ae_score = max(0.0, min(1.0, ae_score))
    else:
        ae_score = 0.0

    # Combine: 70% XGBoost, 30% autoencoder
    combined = 0.7 * xgb_score + 0.3 * ae_score

    return xgb_score, ae_score, combined


def make_decision(combined_score):
    """
    Apply tiered thresholds to make the final decision.
    
    In production, these tiers would trigger different actions:
      BLOCK:   reject the transaction outright
      REVIEW:  send a text/push notification to the cardholder
      ALLOW:   let it through silently
    
    Args:
        combined_score: ensemble fraud score (0 to 1)
    
    Returns:
        decision string and risk level
    """
    if combined_score >= 0.7:
        return 'BLOCK', 'HIGH'
    elif combined_score >= 0.4:
        return 'REVIEW', 'MEDIUM'
    else:
        return 'ALLOW', 'LOW'


def consume(max_messages=None):
    """
    Main consumer loop. Reads from the stream and processes
    transactions until stopped or max_messages is reached.
    """
    r = get_redis()
    setup_stream(r)
    models = load_models()

    xgb_model = models['xgb_model']
    ae_model = models['ae_model']
    scaler = models['scaler']
    training_stats = models['training_stats']
    ae_stats = models['ae_stats']

    consumer_name = 'worker-1'  # unique ID for this consumer
    processed = 0
    results = []  # track results for summary

    print(f"\nConsumer '{consumer_name}' listening on stream '{STREAM_NAME}'...")
    print(f"{'='*80}")

    while True:
        if max_messages and processed >= max_messages:
            break

        # XREADGROUP: read the next unprocessed message
        #   GROUP_NAME: our consumer group
        #   consumer_name: our worker ID
        #   count=1: read one message at a time
        #   block=5000: wait up to 5 seconds if no messages
        #   streams={STREAM_NAME: '>'}: read NEW messages only
        #
        # Returns a list of [stream_name, [(msg_id, msg_data), ...]]
        # or an empty list if the block timeout expires.
        response = r.xreadgroup(
            GROUP_NAME,
            consumer_name,
            {STREAM_NAME: '>'},
            count=1,
            block=5000,
        )

        # No messages available (timeout expired)
        if not response:
            print("  Waiting for transactions...")
            continue

        # Unpack the response
        # response = [['transactions', [('1711234567890-0', {field: value, ...})]]]
        stream, messages = response[0]
        msg_id, msg_data = messages[0]

        # 1. Parse the message
        user_id, amount, features, actual_class = parse_message(msg_data)

        # 2. Get user aggregate from feature store
        user_agg = get_user_aggregate(r, user_id)
        user_features = compute_user_features(user_agg, amount)

        # 3. Build feature vector (same features as training)
        feature_vector = build_feature_vector(features, amount, training_stats)

        # 4. Score with both models
        xgb_score, ae_score, combined_score = score_transaction(
            xgb_model, ae_model, scaler, ae_stats, feature_vector
        )

        # 5. Make decision
        decision, risk = make_decision(combined_score)

        # 6. Update user aggregate for next time
        update_user_aggregate(r, user_id, amount)

        # 7. Acknowledge the message — tells Redis we're done with it
        r.xack(STREAM_NAME, GROUP_NAME, msg_id)

        # 8. Store result in Redis for the dashboard to read
        result = {
            'msg_id': msg_id,
            'user_id': user_id,
            'amount': round(float(amount), 2),
            'xgb_score': round(float(xgb_score), 3),
            'ae_score': round(float(ae_score), 3),
            'combined_score': round(float(combined_score), 3),
            'decision': decision,
            'risk': risk,
            'actual': int(actual_class),
            'timestamp': time.time(),
        }
        # LPUSH adds to the front of a list. The dashboard reads
        # the most recent N results with LRANGE.
        r.lpush('results', json.dumps(result))
        # Keep only the last 500 results to avoid unbounded growth
        r.ltrim('results', 0, 499)

        # 9. Log the result
        actual = "FRAUD" if actual_class == 1 else "LEGIT"
        correct = "✓" if (decision == 'BLOCK' and actual_class == 1) or \
                         (decision == 'ALLOW' and actual_class == 0) else \
                  "~" if decision == 'REVIEW' else "✗"

        print(f"  {msg_id} | user={user_id:>4} | ${amount:>9.2f} | "
              f"XGB={xgb_score:.3f} AE={ae_score:.3f} → {combined_score:.3f} | "
              f"{decision:>6} ({risk:>6}) | actual={actual:>5} {correct}")

        results.append({
            'actual': actual_class,
            'decision': decision,
            'combined_score': combined_score,
            'user_features': user_features,
        })
        processed += 1

    # Print summary
    if results:
        print(f"\n{'='*80}")
        print(f"  SUMMARY: {processed} transactions processed")
        print(f"{'='*80}")

        blocked = sum(1 for r in results if r['decision'] == 'BLOCK')
        reviewed = sum(1 for r in results if r['decision'] == 'REVIEW')
        allowed = sum(1 for r in results if r['decision'] == 'ALLOW')
        actual_fraud = sum(1 for r in results if r['actual'] == 1)
        fraud_caught = sum(1 for r in results if r['actual'] == 1 and r['decision'] == 'BLOCK')
        false_blocks = sum(1 for r in results if r['actual'] == 0 and r['decision'] == 'BLOCK')

        print(f"  Decisions:  {blocked} blocked | {reviewed} reviewed | {allowed} allowed")
        print(f"  Actual fraud: {actual_fraud} / {processed}")
        if actual_fraud > 0:
            print(f"  Fraud caught (blocked): {fraud_caught} / {actual_fraud} = {fraud_caught/actual_fraud*100:.1f}%")
        if blocked > 0:
            print(f"  False blocks: {false_blocks} / {blocked}")


if __name__ == '__main__':
    consume(max_messages=50)