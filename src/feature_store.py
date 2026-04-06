"""
Feature Store — per-user aggregates stored in Redis.

This module manages the behavioral profile for each user. When a
transaction comes in, we need to answer questions like "is this
amount unusual for THIS user?" To do that, we need to know their
history: average spend, last transaction time, typical patterns.

HOW IT WORKS:

  For each user, we store a small summary (aggregate) in Redis
  as a hash map. A Redis hash is like a Python dictionary stored
  under a single key:

    Key: "user:123"
    Value: {
        "txn_count": "47",
        "amount_sum": "1523.45",
        "amount_avg": "32.41",
        "amount_max": "245.00",
        "last_amount": "18.50",
        "last_time": "1711234567"
    }

  When a new transaction arrives for user 123:
    1. Read their current aggregate from Redis (fast — in memory)
    2. Compute features by comparing the transaction to the aggregate
    3. Score the transaction
    4. Update the aggregate with the new transaction's data

  REDIS HASHES:
    HSET key field value — set one field in the hash
    HGETALL key          — get all fields as a dictionary
    HINCRBY key field N  — increment a field by N (atomic)

    Why hashes instead of storing a JSON string?
    Hashes let you read/write individual fields without loading
    the entire object. If you only need the amount_avg, you can
    HGET just that field instead of parsing a JSON blob.
"""

import time
import numpy as np

USER_PREFIX = 'user:'


def get_user_aggregate(r, user_id):
    """
    Read a user's aggregate from Redis.
    
    HGETALL returns all field-value pairs for a key as a dictionary.
    If the user doesn't exist yet (first transaction), it returns
    an empty dictionary {}.
    
    All values come back as strings (Redis stores everything as
    strings), so we convert them to floats/ints as needed.
    """
    key = f"{USER_PREFIX}{user_id}"
    data = r.hgetall(key)

    if not data:
        return {}

    # Convert string values back to numbers
    return {
        'txn_count': int(data.get('txn_count', 0)),
        'amount_sum': float(data.get('amount_sum', 0)),
        'amount_avg': float(data.get('amount_avg', 0)),
        'amount_max': float(data.get('amount_max', 0)),
        'last_amount': float(data.get('last_amount', 0)),
        'last_time': float(data.get('last_time', 0)),
    }


def update_user_aggregate(r, user_id, amount):
    """
    Update a user's aggregate after processing a transaction.
    
    We use a Redis pipeline here. A pipeline batches multiple
    Redis commands into a single round-trip to the server:
    """
    key = f"{USER_PREFIX}{user_id}"
    now = time.time()

    # Get current values (or defaults for new users)
    current = get_user_aggregate(r, user_id)
    txn_count = current.get('txn_count', 0) + 1
    amount_sum = current.get('amount_sum', 0) + amount
    amount_avg = amount_sum / txn_count
    amount_max = max(current.get('amount_max', 0), amount)

    # Pipeline: batch all updates into one network call
    pipe = r.pipeline()
    pipe.hset(key, mapping={
        'txn_count': str(txn_count),
        'amount_sum': str(amount_sum),
        'amount_avg': str(amount_avg),
        'amount_max': str(amount_max),
        'last_amount': str(amount),
        'last_time': str(now),
    })
    pipe.execute()


def compute_user_features(aggregate, amount):
    """
    Compute features by comparing this transaction against the
    user's historical behavior.
    
    These are the features we COULDN'T compute during model training
    because the dataset didn't have user IDs. In the streaming
    pipeline, we simulate them. They're the most valuable fraud
    signals because they capture individual behavior patterns.
    """
    if not aggregate or aggregate.get('txn_count', 0) == 0:
        # First transaction for this user — no history to compare
        return {
            'amount_vs_avg': 0.0,      # no baseline yet
            'amount_vs_max': 0.0,      # no baseline yet
            'time_since_last': -1.0,   # -1 signals "no previous transaction"
            'is_new_user': 1.0,        # flag for first-time users
        }

    # How does this amount compare to their average?
    # A value of 5.0 means "5x their usual spending"
    avg = aggregate['amount_avg']
    if avg > 0:
        amount_vs_avg = amount / avg
    else:
        amount_vs_avg = 0.0

    # How does this compare to their largest ever transaction?
    # A value > 1.0 means this is their biggest purchase ever
    max_amt = aggregate['amount_max']
    if max_amt > 0:
        amount_vs_max = amount / max_amt
    else:
        amount_vs_max = 0.0

    # How long since their last transaction? (in seconds)
    # Very short gaps between transactions can signal fraud —
    # someone rapidly draining a stolen card.
    last_time = aggregate.get('last_time', 0)
    if last_time > 0:
        time_since_last = time.time() - last_time
    else:
        time_since_last = -1.0

    return {
        'amount_vs_avg': amount_vs_avg,
        'amount_vs_max': amount_vs_max,
        'time_since_last': time_since_last,
        'is_new_user': 0.0,
    }
