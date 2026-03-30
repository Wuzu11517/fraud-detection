import os
import sys
import time
import json
import redis
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_data, PROJECT_ROOT

def get_redis():
    """Create a connection to Redis."""
    return redis.Redis(host='localhost', port=6379, decode_responses=True)


STREAM_NAME = 'transactions'     
GROUP_NAME = 'fraud_detectors'

def setup_stream(r):
    """
    Create the stream and consumer group if they don't exist.
    
    A consumer group lets multiple workers share the load. If you
    had 3 workers, Redis ensures each transaction is only sent to
    ONE worker.
    """
    try:
        r.xgroup_create(STREAM_NAME, GROUP_NAME, id='0', mkstream=True)
        print(f"Created stream '{STREAM_NAME}' and group '{GROUP_NAME}'")
    except redis.exceptions.ResponseError as e:
        # "BUSYGROUP" means the group already exists
        if 'BUSYGROUP' in str(e):
            print(f"Stream '{STREAM_NAME}' and group '{GROUP_NAME}' already exist")
        else:
            raise

def produce_transactions(r, df, count=100, delay=0.1):
    """
    Push transactions into the Redis Stream one at a time.
    
    For each transaction, we:
      1. Convert the row to a dictionary
      2. Assign a fake user_id (simulating real users)
      3. Add a timestamp
      4. Push it into the stream with XADD
    """
    np.random.seed(42)

    # Assign fake user IDs from a pool of 1000 users
    # In reality, this would come from the payment processor
    user_ids = np.random.randint(1, 1001, size=len(df))

    print(f"\nProducing {count} transactions into stream '{STREAM_NAME}'...")
    print(f"Delay between transactions: {delay}s\n")

    for i in range(min(count, len(df))):
        row = df.iloc[i]

        message = {
            'user_id': str(user_ids[i]),
            'amount': str(row['Amount']),
            'class': str(int(row['Class'])), 
        }

        # Add all V features
        for v in range(1, 29):
            message[f'V{v}'] = str(row[f'V{v}'])

        msg_id = r.xadd(STREAM_NAME, message)

        label = "FRAUD" if int(row['Class']) == 1 else "LEGIT"
        print(f"  [{i+1}/{count}] Produced txn {msg_id} | user={user_ids[i]} | ${row['Amount']:.2f} | {label}")

        time.sleep(delay)

    print(f"\nDone. Produced {count} transactions.")
    # Show how many messages are in the stream now
    stream_info = r.xinfo_stream(STREAM_NAME)
    print(f"Stream length: {stream_info['length']}")

if __name__ == '__main__':
    r = get_redis()
    setup_stream(r)

    df = load_data()
    
    produce_transactions(r, df, count=50, delay=0.05)
