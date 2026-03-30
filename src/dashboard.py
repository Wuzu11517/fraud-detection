"""
Dashboard — Flask web app for monitoring the fraud detection pipeline.

Reads scored transaction results from Redis (stored by the consumer)
and displays them in a live-updating browser dashboard.

Architecture:
  Consumer scores transactions → stores results in Redis list 'results'
  Dashboard reads from Redis → serves via Flask API → frontend polls

Run: python src/dashboard.py
Then open http://localhost:5000 in your browser.
"""

import os
import sys
import json
import redis
from flask import Flask, render_template, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

def get_redis():
    return redis.Redis(host='localhost', port=6379, decode_responses=True)


@app.route('/')
def index():
    """Serve the dashboard HTML page."""
    return render_template('dashboard.html')


@app.route('/api/transactions')
def get_transactions():
    """
    Return the most recent scored transactions.
    
    Reads from the Redis 'results' list that the consumer populates.
    LRANGE 0 49 returns the 50 most recent (LPUSH puts newest first).
    """
    r = get_redis()
    raw = r.lrange('results', 0, 49)
    transactions = [json.loads(item) for item in raw]
    return jsonify(transactions)


@app.route('/api/stats')
def get_stats():
    """
    Return aggregated statistics across all processed transactions.
    
    Computes totals, catch rates, and score distributions from
    the stored results.
    """
    r = get_redis()
    raw = r.lrange('results', 0, 499)  # up to 500 recent results
    
    if not raw:
        return jsonify({
            'total': 0,
            'blocked': 0,
            'reviewed': 0,
            'allowed': 0,
            'actual_fraud': 0,
            'fraud_caught': 0,
            'false_blocks': 0,
            'catch_rate': 0,
            'false_positive_rate': 0,
            'avg_xgb_score_fraud': 0,
            'avg_xgb_score_legit': 0,
            'avg_ae_score_fraud': 0,
            'avg_ae_score_legit': 0,
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
        'total': total,
        'blocked': blocked,
        'reviewed': reviewed,
        'allowed': allowed,
        'actual_fraud': actual_fraud,
        'fraud_caught': fraud_caught,
        'false_blocks': false_blocks,
        'catch_rate': round(catch_rate, 1),
        'false_positive_rate': round(false_positive_rate, 2),
        'avg_xgb_score_fraud': round(avg_xgb_fraud, 3),
        'avg_xgb_score_legit': round(avg_xgb_legit, 3),
        'avg_ae_score_fraud': round(avg_ae_fraud, 3),
        'avg_ae_score_legit': round(avg_ae_legit, 3),
    })


if __name__ == '__main__':
    print("Dashboard running at http://localhost:5000")
    print("Make sure the consumer is running to see live data.")
    app.run(host='0.0.0.0', port=5000, debug=True)
