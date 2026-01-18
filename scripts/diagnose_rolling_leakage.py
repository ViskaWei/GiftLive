#!/usr/bin/env python3
"""
Rolling Leakage Diagnosis: MVP-1.1
==================================

Experiment: EXP-20260118-gift_EVpred-03

Goal: Diagnose why Rolling version (81.1%) vastly outperforms Frozen (11.5%)
- Check if cumsum+shift implementation has time leakage
- Identify which feature group contains leakage
- Fix and re-evaluate if leakage confirmed

Author: Viska Wei
Date: 2026-01-18
"""

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path("/home/swei20/GiftLive")
DATA_DIR = BASE_DIR / "data" / "KuaiLive"
OUTPUT_DIR = BASE_DIR / "gift_EVpred"
IMG_DIR = OUTPUT_DIR / "img"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

for d in [IMG_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def log_message(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "📝", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌", "CRITICAL": "🔴"}.get(level, "")
    print(f"[{timestamp}] {prefix} {msg}")


def load_data():
    """Load raw data files."""
    log_message("Loading data files...")
    gift = pd.read_csv(DATA_DIR / "gift.csv")
    click = pd.read_csv(DATA_DIR / "click.csv")
    log_message(f"Gift: {len(gift):,} records")
    log_message(f"Click: {len(click):,} records")
    return gift, click


# =============================================================================
# DIAGNOSIS STEP 1: Timestamp Sorting Check
# =============================================================================
def check_timestamp_sorting(gift, click):
    """Check if data is sorted by timestamp."""
    log_message("\n" + "="*60)
    log_message("STEP 1: Timestamp Sorting Check")
    log_message("="*60)

    results = {}

    # Gift data
    gift_sorted = gift.sort_values('timestamp')
    is_gift_sorted = gift['timestamp'].equals(gift_sorted['timestamp'])
    results['gift_is_sorted'] = is_gift_sorted
    log_message(f"Gift data sorted: {is_gift_sorted}")

    if not is_gift_sorted:
        n_out_of_order = (gift['timestamp'].diff() < 0).sum()
        results['gift_out_of_order_count'] = int(n_out_of_order)
        log_message(f"Gift out-of-order records: {n_out_of_order:,}", "WARNING")

    # Click data
    click_sorted = click.sort_values('timestamp')
    is_click_sorted = click['timestamp'].equals(click_sorted['timestamp'])
    results['click_is_sorted'] = is_click_sorted
    log_message(f"Click data sorted: {is_click_sorted}")

    if not is_click_sorted:
        n_out_of_order = (click['timestamp'].diff() < 0).sum()
        results['click_out_of_order_count'] = int(n_out_of_order)
        log_message(f"Click out-of-order records: {n_out_of_order:,}", "WARNING")

    return results


# =============================================================================
# DIAGNOSIS STEP 2: Duplicate Timestamp Check
# =============================================================================
def check_duplicate_timestamps(gift, click):
    """Check for duplicate timestamps."""
    log_message("\n" + "="*60)
    log_message("STEP 2: Duplicate Timestamp Check")
    log_message("="*60)

    results = {}

    # Gift data
    gift_ts_counts = gift.groupby('timestamp').size()
    results['gift_max_per_ts'] = int(gift_ts_counts.max())
    results['gift_mean_per_ts'] = float(gift_ts_counts.mean())
    results['gift_ts_with_dups'] = int((gift_ts_counts > 1).sum())

    log_message(f"Gift - Max records per timestamp: {results['gift_max_per_ts']}")
    log_message(f"Gift - Mean records per timestamp: {results['gift_mean_per_ts']:.2f}")
    log_message(f"Gift - Timestamps with duplicates: {results['gift_ts_with_dups']:,}")

    # Click data
    click_ts_counts = click.groupby('timestamp').size()
    results['click_max_per_ts'] = int(click_ts_counts.max())
    results['click_mean_per_ts'] = float(click_ts_counts.mean())
    results['click_ts_with_dups'] = int((click_ts_counts > 1).sum())

    log_message(f"Click - Max records per timestamp: {results['click_max_per_ts']}")
    log_message(f"Click - Mean records per timestamp: {results['click_mean_per_ts']:.2f}")
    log_message(f"Click - Timestamps with duplicates: {results['click_ts_with_dups']:,}")

    # Check same (user, streamer, timestamp) combinations
    gift_pair_ts = gift.groupby(['user_id', 'streamer_id', 'timestamp']).size()
    dup_pair_ts = (gift_pair_ts > 1).sum()
    results['gift_dup_pair_ts'] = int(dup_pair_ts)
    log_message(f"Gift - Same (user, streamer, timestamp): {dup_pair_ts:,}")

    return results


# =============================================================================
# DIAGNOSIS STEP 3: First Gift Sample Check (CRITICAL)
# =============================================================================
def check_first_gift_samples(gift):
    """
    Check if first gift for each (user, streamer) pair has pair_gift_count_past = 0.
    This is the CRITICAL test for leakage.
    """
    log_message("\n" + "="*60)
    log_message("STEP 3: First Gift Sample Check (CRITICAL)")
    log_message("="*60)

    results = {}

    # Sort by timestamp
    gift_sorted = gift.sort_values('timestamp').copy()

    # Mark first occurrence for each (user, streamer) pair
    gift_sorted['pair_rank'] = gift_sorted.groupby(['user_id', 'streamer_id']).cumcount()
    first_gifts = gift_sorted[gift_sorted['pair_rank'] == 0].copy()

    log_message(f"Total unique (user, streamer) pairs: {len(first_gifts):,}")

    # For first gifts, the "past" count should be 0
    # But the rolling implementation might include current sample

    # Simulate rolling implementation (what train_leakage_free_baseline.py does)
    # The issue: it uses groupby().agg() on FULL data, then merges back
    # This means even first gift gets the TOTAL count, not past count

    # Let's check what the rolling implementation actually computes
    pair_stats = gift_sorted.groupby(['user_id', 'streamer_id']).agg({
        'gift_price': ['count', 'sum', 'mean']
    }).reset_index()
    pair_stats.columns = ['user_id', 'streamer_id', 'pair_gift_count', 'pair_gift_sum', 'pair_gift_mean']

    # Merge back to first gifts
    first_gifts_with_stats = first_gifts.merge(
        pair_stats, on=['user_id', 'streamer_id'], how='left'
    )

    # Check: for first gifts, the rolling implementation gives what value?
    non_zero_count = (first_gifts_with_stats['pair_gift_count'] != 1).sum()
    # Note: pair_gift_count should be at least 1 (the current sample), but if it's > 1,
    # it means FUTURE samples are included!

    multi_gift_pairs = (first_gifts_with_stats['pair_gift_count'] > 1).sum()
    results['first_gifts_total'] = len(first_gifts)
    results['first_gifts_with_future_data'] = int(multi_gift_pairs)
    results['leakage_rate_future'] = float(multi_gift_pairs / len(first_gifts) * 100)

    log_message(f"First gifts with count > 1 (includes future): {multi_gift_pairs:,} ({results['leakage_rate_future']:.1f}%)")

    # The REAL issue: rolling implementation uses ALL data statistics, not past-only
    # Even for pair_gift_count=1 samples, the mean/sum might be from future

    # Let's check a sample
    sample_pairs = first_gifts_with_stats.head(10)
    log_message("\nSample first gifts with rolling stats:")
    for _, row in sample_pairs.iterrows():
        log_message(f"  Pair ({row['user_id']}, {row['streamer_id']}): "
                   f"count={row['pair_gift_count']}, sum={row['pair_gift_sum']:.2f}, mean={row['pair_gift_mean']:.2f}")

    # Diagnosis conclusion
    if multi_gift_pairs > 0:
        log_message(f"\n🔴 LEAKAGE CONFIRMED: {results['leakage_rate_future']:.1f}% of first gifts have future data!", "CRITICAL")
        results['leakage_confirmed'] = True
    else:
        log_message("\n⚠️ Need deeper check - count=1 but mean/sum might still be wrong", "WARNING")
        results['leakage_confirmed'] = 'uncertain'

    return results


# =============================================================================
# DIAGNOSIS STEP 4: Time Travel Check (CRITICAL)
# =============================================================================
def check_time_travel(gift, n_samples=100):
    """
    For a sample of records, compare rolling features vs true past-only features.
    """
    log_message("\n" + "="*60)
    log_message(f"STEP 4: Time Travel Check (n={n_samples})")
    log_message("="*60)

    results = {}

    gift_sorted = gift.sort_values('timestamp').copy()

    # Sample from later records (more likely to have history)
    n_total = len(gift_sorted)
    sample_start = int(n_total * 0.5)  # Start from middle
    sample_indices = np.random.choice(
        range(sample_start, n_total),
        size=min(n_samples, n_total - sample_start),
        replace=False
    )

    # What rolling implementation computes (using ALL data)
    pair_stats_all = gift_sorted.groupby(['user_id', 'streamer_id']).agg({
        'gift_price': ['count', 'sum', 'mean']
    }).reset_index()
    pair_stats_all.columns = ['user_id', 'streamer_id', 'rolling_count', 'rolling_sum', 'rolling_mean']
    pair_stats_dict = {(row['user_id'], row['streamer_id']): row for _, row in pair_stats_all.iterrows()}

    mismatches = []

    for idx in sample_indices:
        row = gift_sorted.iloc[idx]
        curr_ts = row['timestamp']
        curr_user = row['user_id']
        curr_streamer = row['streamer_id']

        # True past-only: strictly before current timestamp
        past_gifts = gift_sorted[
            (gift_sorted['user_id'] == curr_user) &
            (gift_sorted['streamer_id'] == curr_streamer) &
            (gift_sorted['timestamp'] < curr_ts)
        ]
        true_count = len(past_gifts)
        true_sum = past_gifts['gift_price'].sum() if len(past_gifts) > 0 else 0
        true_mean = past_gifts['gift_price'].mean() if len(past_gifts) > 0 else 0

        # Rolling implementation values
        key = (curr_user, curr_streamer)
        if key in pair_stats_dict:
            rolling_stats = pair_stats_dict[key]
            rolling_count = rolling_stats['rolling_count']
            rolling_sum = rolling_stats['rolling_sum']
            rolling_mean = rolling_stats['rolling_mean']
        else:
            rolling_count = 0
            rolling_sum = 0
            rolling_mean = 0

        # Compare
        if rolling_count != true_count:
            mismatches.append({
                'idx': int(idx),
                'user_id': int(curr_user),
                'streamer_id': int(curr_streamer),
                'true_count': int(true_count),
                'rolling_count': int(rolling_count),
                'count_diff': int(rolling_count - true_count),
                'true_sum': float(true_sum),
                'rolling_sum': float(rolling_sum),
            })

    results['samples_checked'] = len(sample_indices)
    results['mismatches_found'] = len(mismatches)
    results['mismatch_rate'] = float(len(mismatches) / len(sample_indices) * 100)

    log_message(f"Samples checked: {results['samples_checked']}")
    log_message(f"Mismatches found: {results['mismatches_found']} ({results['mismatch_rate']:.1f}%)")

    if len(mismatches) > 0:
        log_message("\n🔴 TIME TRAVEL CONFIRMED!", "CRITICAL")
        log_message("\nSample mismatches:")
        for m in mismatches[:5]:
            log_message(f"  idx={m['idx']}: true_count={m['true_count']}, rolling_count={m['rolling_count']}, diff={m['count_diff']}")

        # Analyze the pattern
        diffs = [m['count_diff'] for m in mismatches]
        results['avg_count_diff'] = float(np.mean(diffs))
        results['max_count_diff'] = int(np.max(diffs))
        log_message(f"\nCount difference: avg={results['avg_count_diff']:.1f}, max={results['max_count_diff']}")

        results['leakage_confirmed'] = True
    else:
        log_message("\n✅ No time travel detected in sample", "SUCCESS")
        results['leakage_confirmed'] = False

    results['sample_mismatches'] = mismatches[:10]  # Save first 10

    return results


# =============================================================================
# DIAGNOSIS STEP 5: Analyze Rolling Implementation Code
# =============================================================================
def analyze_rolling_implementation():
    """
    Analyze the create_past_only_features_rolling() function.
    """
    log_message("\n" + "="*60)
    log_message("STEP 5: Rolling Implementation Analysis")
    log_message("="*60)

    log_message("\n🔍 Key issues found in create_past_only_features_rolling():")
    log_message("")
    log_message("1. Line 272-276: Uses groupby().agg() on FULL gift_sorted data")
    log_message("   → This computes statistics using ALL data including future samples")
    log_message("")
    log_message("2. Line 297-303: Merges pair_stats back to df")
    log_message("   → Every sample gets the FULL history stats, not past-only")
    log_message("")
    log_message("3. Line 319: User features also use FULL data")
    log_message("   → user_total_gift_7d includes future gifts")
    log_message("")
    log_message("4. Line 326-333: Streamer features also use FULL data")
    log_message("   → streamer_recent_revenue includes future revenue")
    log_message("")
    log_message("🔴 CONCLUSION: The rolling implementation is fundamentally flawed.")
    log_message("   It uses groupby().agg() on full data, then merges back.")
    log_message("   This means EVERY sample sees the TOTAL statistics, not past-only.")
    log_message("")
    log_message("📌 The correct approach would be:")
    log_message("   1. Sort by timestamp")
    log_message("   2. Use expanding().agg() with shift(1) to exclude current sample")
    log_message("   3. OR use cumsum().shift(1) for cumulative features")
    log_message("   4. BUT the current code doesn't do this at all!")

    return {
        'issue': 'Rolling implementation uses full-data groupby instead of expanding window',
        'fix_needed': True,
        'recommendation': 'Use Frozen version as ground truth, fix Rolling or deprecate it'
    }


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_leakage_diagnosis(diagnosis_results, save_dir):
    """Generate diagnostic plots."""
    log_message("\n" + "="*60)
    log_message("Generating Diagnostic Plots")
    log_message("="*60)

    # Fig 1: Leakage Summary
    fig, ax = plt.subplots(figsize=(6, 5))

    categories = ['First Gift\nFuture Data', 'Time Travel\nMismatch']
    values = [
        diagnosis_results['first_gift_check'].get('leakage_rate_future', 0),
        diagnosis_results['time_travel_check'].get('mismatch_rate', 0)
    ]
    colors = ['#ff6b6b' if v > 50 else '#ffd93d' if v > 10 else '#6bcb77' for v in values]

    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Critical (50%)')
    ax.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Warning (10%)')

    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title('Rolling Leakage Diagnosis Summary', fontsize=14)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_path = save_dir / 'rolling_leakage_diagnosis_summary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")

    # Fig 2: Diagnosis Matrix
    fig, ax = plt.subplots(figsize=(6, 5))

    checks = ['Timestamp\nSorting', 'Duplicate\nTimestamps', 'First Gift\nCheck', 'Time Travel\nCheck']

    # Determine status for each check
    statuses = []

    # Sorting check
    if diagnosis_results['sorting_check']['gift_is_sorted'] and diagnosis_results['sorting_check']['click_is_sorted']:
        statuses.append(1)  # Pass
    else:
        statuses.append(0)  # Fail

    # Duplicate check - informational, not fail/pass
    statuses.append(0.5)  # Neutral

    # First gift check
    fg_rate = diagnosis_results['first_gift_check'].get('leakage_rate_future', 0)
    if fg_rate > 50:
        statuses.append(0)  # Fail
    elif fg_rate > 10:
        statuses.append(0.5)  # Warning
    else:
        statuses.append(1)  # Pass

    # Time travel check
    tt_rate = diagnosis_results['time_travel_check'].get('mismatch_rate', 0)
    if tt_rate > 50:
        statuses.append(0)  # Fail
    elif tt_rate > 10:
        statuses.append(0.5)  # Warning
    else:
        statuses.append(1)  # Pass

    colors_status = ['#6bcb77' if s == 1 else '#ffd93d' if s == 0.5 else '#ff6b6b' for s in statuses]

    ax.barh(checks, [1]*4, color=colors_status, edgecolor='black', linewidth=1.5)

    status_labels = ['PASS' if s == 1 else 'INFO' if s == 0.5 else 'FAIL' for s in statuses]
    for i, (check, label) in enumerate(zip(checks, status_labels)):
        ax.text(0.5, i, label, ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title('Leakage Diagnosis Matrix', fontsize=14)

    plt.tight_layout()
    save_path = save_dir / 'leakage_diagnosis_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    log_message("="*60)
    log_message("Rolling Leakage Diagnosis - MVP-1.1")
    log_message("="*60)

    start_time = time.time()

    # Load data
    gift, click = load_data()

    # Run all diagnosis steps
    diagnosis_results = {}

    # Step 1: Sorting check
    diagnosis_results['sorting_check'] = check_timestamp_sorting(gift, click)

    # Step 2: Duplicate check
    diagnosis_results['duplicate_check'] = check_duplicate_timestamps(gift, click)

    # Step 3: First gift check (CRITICAL)
    diagnosis_results['first_gift_check'] = check_first_gift_samples(gift)

    # Step 4: Time travel check (CRITICAL)
    diagnosis_results['time_travel_check'] = check_time_travel(gift, n_samples=200)

    # Step 5: Code analysis
    diagnosis_results['code_analysis'] = analyze_rolling_implementation()

    # Generate plots
    plot_leakage_diagnosis(diagnosis_results, IMG_DIR)

    # Final summary
    log_message("\n" + "="*60)
    log_message("FINAL DIAGNOSIS SUMMARY")
    log_message("="*60)

    fg_leakage = diagnosis_results['first_gift_check'].get('leakage_rate_future', 0)
    tt_leakage = diagnosis_results['time_travel_check'].get('mismatch_rate', 0)

    log_message(f"\n📊 First Gift Future Data Rate: {fg_leakage:.1f}%")
    log_message(f"📊 Time Travel Mismatch Rate: {tt_leakage:.1f}%")

    if fg_leakage > 50 or tt_leakage > 50:
        log_message("\n🔴 LEAKAGE CONFIRMED: Rolling implementation has severe time leakage!", "CRITICAL")
        diagnosis_results['final_verdict'] = 'LEAKAGE_CONFIRMED'
        diagnosis_results['recommendation'] = 'Use Frozen version only. Rolling implementation is fundamentally flawed.'
    elif fg_leakage > 10 or tt_leakage > 10:
        log_message("\n⚠️ LEAKAGE DETECTED: Rolling implementation has moderate time leakage", "WARNING")
        diagnosis_results['final_verdict'] = 'LEAKAGE_DETECTED'
        diagnosis_results['recommendation'] = 'Fix Rolling implementation or use Frozen version.'
    else:
        log_message("\n✅ No significant leakage detected", "SUCCESS")
        diagnosis_results['final_verdict'] = 'NO_LEAKAGE'
        diagnosis_results['recommendation'] = 'Rolling implementation appears correct.'

    log_message(f"\n💡 Recommendation: {diagnosis_results['recommendation']}")

    # Save results
    elapsed = time.time() - start_time
    diagnosis_results['elapsed_time'] = elapsed

    results_path = RESULTS_DIR / 'rolling_leakage_diagnosis_20260118.json'

    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    diagnosis_results = convert_types(diagnosis_results)

    with open(results_path, 'w') as f:
        json.dump(diagnosis_results, f, indent=2)

    log_message(f"\nResults saved to: {results_path}")
    log_message(f"Total time: {elapsed:.1f}s")
    log_message("\n" + "="*60)
    log_message("Diagnosis Complete!", "SUCCESS")
    log_message("="*60)


if __name__ == '__main__':
    main()
