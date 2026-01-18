#!/usr/bin/env python3
"""
Test watch_time truncation impact on label construction.

Compare:
- Original: label_window = [t_click, t_click + H]
- Fixed: label_window = [t_click, min(t_click + watch_time, t_click + H)]

Author: Viska Wei
Date: 2026-01-18
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path("/home/swei20/GiftLive")
DATA_DIR = BASE_DIR / "data" / "KuaiLive"
RESULTS_DIR = BASE_DIR / "gift_EVpred" / "results"


def log_message(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "📝", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "")
    print(f"[{timestamp}] {prefix} {msg}")


def prepare_click_level_data_original(gift, click, label_window_hours=1):
    """Original implementation: window = [t_click, t_click + H]"""
    click_base = click.copy()
    click_base['timestamp_dt'] = pd.to_datetime(click_base['timestamp'], unit='ms')

    gift = gift.copy()
    gift['timestamp_dt'] = pd.to_datetime(gift['timestamp'], unit='ms')

    # Original: fixed window
    click_base['click_end_ts'] = click_base['timestamp_dt'] + pd.Timedelta(hours=label_window_hours)

    merged = click_base[['user_id', 'streamer_id', 'live_id', 'timestamp_dt', 'click_end_ts']].merge(
        gift[['user_id', 'streamer_id', 'live_id', 'timestamp_dt', 'gift_price']],
        on=['user_id', 'streamer_id', 'live_id'],
        how='left',
        suffixes=('_click', '_gift')
    )

    merged = merged[
        (merged['timestamp_dt_gift'] >= merged['timestamp_dt_click']) &
        (merged['timestamp_dt_gift'] <= merged['click_end_ts'])
    ]

    gift_agg = merged.groupby(['user_id', 'streamer_id', 'live_id', 'timestamp_dt_click']).agg({
        'gift_price': 'sum'
    }).reset_index().rename(columns={
        'timestamp_dt_click': 'timestamp_dt',
        'gift_price': 'gift_price_label'
    })

    click_base = click_base.merge(gift_agg, on=['user_id', 'streamer_id', 'live_id', 'timestamp_dt'], how='left')
    click_base['gift_price_label'] = click_base['gift_price_label'].fillna(0)

    return click_base


def prepare_click_level_data_fixed(gift, click, label_window_hours=1):
    """Fixed implementation: window = [t_click, min(t_click + watch_time, t_click + H)]"""
    click_base = click.copy()
    click_base['timestamp_dt'] = pd.to_datetime(click_base['timestamp'], unit='ms')

    gift = gift.copy()
    gift['timestamp_dt'] = pd.to_datetime(gift['timestamp'], unit='ms')

    # Fixed: use min(watch_time, H) for truncation
    max_window_ms = label_window_hours * 3600 * 1000  # Convert hours to ms
    click_base['effective_window_ms'] = np.minimum(click_base['watch_live_time'], max_window_ms)
    click_base['click_end_ts'] = click_base['timestamp_dt'] + pd.to_timedelta(click_base['effective_window_ms'], unit='ms')

    merged = click_base[['user_id', 'streamer_id', 'live_id', 'timestamp_dt', 'click_end_ts']].merge(
        gift[['user_id', 'streamer_id', 'live_id', 'timestamp_dt', 'gift_price']],
        on=['user_id', 'streamer_id', 'live_id'],
        how='left',
        suffixes=('_click', '_gift')
    )

    merged = merged[
        (merged['timestamp_dt_gift'] >= merged['timestamp_dt_click']) &
        (merged['timestamp_dt_gift'] <= merged['click_end_ts'])
    ]

    gift_agg = merged.groupby(['user_id', 'streamer_id', 'live_id', 'timestamp_dt_click']).agg({
        'gift_price': 'sum'
    }).reset_index().rename(columns={
        'timestamp_dt_click': 'timestamp_dt',
        'gift_price': 'gift_price_label'
    })

    click_base = click_base.merge(gift_agg, on=['user_id', 'streamer_id', 'live_id', 'timestamp_dt'], how='left')
    click_base['gift_price_label'] = click_base['gift_price_label'].fillna(0)

    return click_base


def main():
    log_message("="*60)
    log_message("Watch Time Truncation Impact Analysis")
    log_message("="*60)

    # Load data
    log_message("Loading data...")
    gift = pd.read_csv(DATA_DIR / "gift.csv")
    click = pd.read_csv(DATA_DIR / "click.csv")

    log_message(f"Gift: {len(gift):,} records")
    log_message(f"Click: {len(click):,} records")

    # Watch time statistics
    log_message("\n--- Watch Time Statistics ---")
    watch_time_sec = click['watch_live_time'] / 1000  # Convert to seconds
    log_message(f"Watch time (seconds): min={watch_time_sec.min():.1f}, median={watch_time_sec.median():.1f}, "
               f"mean={watch_time_sec.mean():.1f}, max={watch_time_sec.max():.1f}")

    # How many sessions are shorter than 1 hour?
    one_hour_ms = 3600 * 1000
    short_sessions = (click['watch_live_time'] < one_hour_ms).mean() * 100
    log_message(f"Sessions shorter than 1h: {short_sessions:.1f}%")

    # Prepare data with both methods
    log_message("\n--- Preparing Original Labels (H=1h fixed window) ---")
    click_original = prepare_click_level_data_original(gift, click, label_window_hours=1)

    log_message("\n--- Preparing Fixed Labels (min(watch_time, H)) ---")
    click_fixed = prepare_click_level_data_fixed(gift, click, label_window_hours=1)

    # Compare results
    log_message("\n" + "="*60)
    log_message("COMPARISON RESULTS")
    log_message("="*60)

    results = {}

    # 1. Gift rate comparison
    orig_gift_rate = (click_original['gift_price_label'] > 0).mean() * 100
    fixed_gift_rate = (click_fixed['gift_price_label'] > 0).mean() * 100

    results['original_gift_rate'] = orig_gift_rate
    results['fixed_gift_rate'] = fixed_gift_rate
    results['gift_rate_diff'] = orig_gift_rate - fixed_gift_rate

    log_message(f"\n📊 Gift Rate:")
    log_message(f"   Original (fixed H): {orig_gift_rate:.3f}%")
    log_message(f"   Fixed (min truncation): {fixed_gift_rate:.3f}%")
    log_message(f"   Difference: {orig_gift_rate - fixed_gift_rate:.3f}pp")

    # 2. Total gift amount comparison
    orig_total = click_original['gift_price_label'].sum()
    fixed_total = click_fixed['gift_price_label'].sum()

    results['original_total_gift'] = float(orig_total)
    results['fixed_total_gift'] = float(fixed_total)
    results['total_gift_diff_pct'] = (orig_total - fixed_total) / orig_total * 100

    log_message(f"\n📊 Total Gift Amount:")
    log_message(f"   Original: {orig_total:,.0f}")
    log_message(f"   Fixed: {fixed_total:,.0f}")
    log_message(f"   Difference: {(orig_total - fixed_total) / orig_total * 100:.2f}%")

    # 3. Per-sample comparison
    # How many samples have different labels?
    diff_mask = click_original['gift_price_label'] != click_fixed['gift_price_label']
    diff_count = diff_mask.sum()
    diff_pct = diff_count / len(click_original) * 100

    results['samples_with_diff_labels'] = int(diff_count)
    results['diff_pct'] = diff_pct

    log_message(f"\n📊 Label Differences:")
    log_message(f"   Samples with different labels: {diff_count:,} ({diff_pct:.3f}%)")

    # 4. Cases where original > fixed (gifts outside watch window)
    outside_window = (click_original['gift_price_label'] > click_fixed['gift_price_label'])
    outside_count = outside_window.sum()
    outside_pct = outside_count / len(click_original) * 100

    results['gifts_outside_watch_window'] = int(outside_count)
    results['outside_pct'] = outside_pct

    log_message(f"   Gifts attributed outside watch window: {outside_count:,} ({outside_pct:.3f}%)")

    # 5. Revenue attributed outside watch window
    outside_revenue = (click_original['gift_price_label'] - click_fixed['gift_price_label']).clip(lower=0).sum()
    outside_revenue_pct = outside_revenue / orig_total * 100

    results['revenue_outside_watch_window'] = float(outside_revenue)
    results['outside_revenue_pct'] = outside_revenue_pct

    log_message(f"   Revenue outside watch window: {outside_revenue:,.0f} ({outside_revenue_pct:.2f}%)")

    # Verdict
    log_message("\n" + "="*60)
    log_message("VERDICT")
    log_message("="*60)

    if outside_revenue_pct < 1:
        log_message(f"✅ Impact is MINIMAL ({outside_revenue_pct:.2f}% revenue difference)", "SUCCESS")
        log_message("   The current fixed-window implementation is acceptable.")
        results['verdict'] = 'MINIMAL_IMPACT'
    elif outside_revenue_pct < 5:
        log_message(f"⚠️ Impact is MODERATE ({outside_revenue_pct:.2f}% revenue difference)", "WARNING")
        log_message("   Consider using watch_time truncation for more accuracy.")
        results['verdict'] = 'MODERATE_IMPACT'
    else:
        log_message(f"🔴 Impact is SIGNIFICANT ({outside_revenue_pct:.2f}% revenue difference)", "ERROR")
        log_message("   Watch_time truncation is recommended.")
        results['verdict'] = 'SIGNIFICANT_IMPACT'

    # Save results
    results_path = RESULTS_DIR / 'watch_time_truncation_analysis_20260118.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    log_message(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
