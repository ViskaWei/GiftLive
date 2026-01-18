#!/usr/bin/env python3
"""
Quick test to verify rolling feature fix is leakage-free.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/home/swei20/GiftLive")
DATA_DIR = BASE_DIR / "data" / "KuaiLive"

print("Loading data...")
gift = pd.read_csv(DATA_DIR / "gift.csv")
click = pd.read_csv(DATA_DIR / "click.csv")

print(f"Gift: {len(gift):,} records")
print(f"Click: {len(click):,} records")

# Test the vectorized rolling feature function
print("\n" + "="*60)
print("Testing create_past_only_features_rolling_vectorized")
print("="*60)

# Prepare a small sample for testing
gift_sorted = gift.sort_values('timestamp').copy()
click_sorted = click.sort_values('timestamp').copy()
# Use LAST 1000 clicks (more likely to have history)
click_sorted = click_sorted.tail(1000).reset_index(drop=True)

# Compute cumulative stats per pair
gift_sorted['pair_gift_count_cum'] = gift_sorted.groupby(['user_id', 'streamer_id']).cumcount() + 1
gift_sorted['pair_gift_sum_cum'] = gift_sorted.groupby(['user_id', 'streamer_id'])['gift_price'].cumsum()

# Build lookup structure
pair_lookup = {}
for (user_id, streamer_id), grp in gift_sorted.groupby(['user_id', 'streamer_id']):
    grp = grp.sort_values('timestamp')
    pair_lookup[(user_id, streamer_id)] = {
        'ts': grp['timestamp'].values,
        'count': grp['pair_gift_count_cum'].values,
        'sum': grp['pair_gift_sum_cum'].values
    }

print(f"Built lookup for {len(pair_lookup):,} unique pairs")

# Test the binary search approach
df = click_sorted.copy()
pair_count = np.zeros(len(df))
pair_sum = np.zeros(len(df))

for idx, row in df.iterrows():
    key = (row['user_id'], row['streamer_id'])
    click_ts = row['timestamp']

    if key in pair_lookup:
        lookup = pair_lookup[key]
        ts_arr = lookup['ts']
        # Find position: strictly less than click_ts
        pos = np.searchsorted(ts_arr, click_ts, side='left') - 1
        if pos >= 0:
            pair_count[idx] = lookup['count'][pos]
            pair_sum[idx] = lookup['sum'][pos]

df['pair_gift_count_past'] = pair_count
df['pair_gift_sum_past'] = pair_sum

print(f"\nComputed rolling features for {len(df):,} clicks")
print(f"Non-zero pair_count rate: {(pair_count > 0).mean()*100:.1f}%")

# VERIFICATION: Check against ground truth
print("\n" + "="*60)
print("VERIFICATION: Checking for leakage...")
print("="*60)

n_tests = 100
tests_passed = 0
tests_failed = 0
errors = []

sample_indices = np.random.choice(len(df), size=min(n_tests, len(df)), replace=False)

for i, idx in enumerate(sample_indices):
    row = df.iloc[idx]
    click_ts = row['timestamp']
    user_id = row['user_id']
    streamer_id = row['streamer_id']

    # Ground truth: count gifts STRICTLY BEFORE click
    past_gifts = gift_sorted[
        (gift_sorted['user_id'] == user_id) &
        (gift_sorted['streamer_id'] == streamer_id) &
        (gift_sorted['timestamp'] < click_ts)
    ]
    true_count = len(past_gifts)

    # Compare with computed value
    computed_count = int(row['pair_gift_count_past'])

    if computed_count == true_count:
        tests_passed += 1
    else:
        tests_failed += 1
        errors.append({
            'idx': idx,
            'click_ts': click_ts,
            'user_id': user_id,
            'streamer_id': streamer_id,
            'expected': true_count,
            'got': computed_count,
            'diff': computed_count - true_count
        })

print(f"\nResults: {tests_passed}/{n_tests} passed, {tests_failed}/{n_tests} failed")

if tests_failed == 0:
    print("\n✅ VERIFICATION PASSED: Rolling features are leakage-free!")
else:
    print(f"\n❌ VERIFICATION FAILED: {tests_failed} samples have leakage")
    print("\nFirst 5 errors:")
    for err in errors[:5]:
        print(f"  idx={err['idx']}: expected={err['expected']}, got={err['got']}, diff={err['diff']}")

    # Diagnose the issue
    print("\nDiagnosing the first error...")
    err = errors[0]
    key = (err['user_id'], err['streamer_id'])
    if key in pair_lookup:
        lookup = pair_lookup[key]
        print(f"  Pair {key} has {len(lookup['ts'])} gifts")
        print(f"  Gift timestamps: {lookup['ts'][:5]}...")
        print(f"  Click timestamp: {err['click_ts']}")
        pos = np.searchsorted(lookup['ts'], err['click_ts'], side='left') - 1
        print(f"  searchsorted position: {pos}")
        if pos >= 0:
            print(f"  Gift at pos {pos}: ts={lookup['ts'][pos]}, count={lookup['count'][pos]}")

print("\nDone!")
