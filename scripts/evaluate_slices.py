#!/usr/bin/env python3
"""
Slice Evaluation - MVP-1.4
==========================

Evaluate model performance across different slices:
- Cold-start pair (train pair_gift_count=0)
- Cold-start streamer (train streamer_revenue=0)
- Cold-start user (train user_gift=0)
- Top-1% user, Top-10% user, Tail user

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
IMG_DIR = OUTPUT_DIR / "exp" / "img"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"

for d in [IMG_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


def log_message(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "📝", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "")
    print(f"[{timestamp}] {prefix} {msg}")


def compute_revenue_capture_at_k(y_true, y_pred, k_pct=0.01):
    """Compute Revenue Capture@K."""
    n = len(y_true)
    k = max(1, int(n * k_pct))

    pred_order = np.argsort(-y_pred)
    top_k_indices = pred_order[:k]

    revenue_top_k = y_true[top_k_indices].sum()
    total_revenue = y_true.sum()

    if total_revenue == 0:
        return 0.0

    return revenue_top_k / total_revenue


def compute_top_k_capture(y_true, y_pred, k_pct=0.01):
    """Compute Top-K% capture rate (set overlap)."""
    n = len(y_true)
    k = max(1, int(n * k_pct))

    true_rank = np.argsort(np.argsort(-y_true))
    pred_rank = np.argsort(np.argsort(-y_pred))

    true_topk = set(np.where(true_rank < k)[0])
    pred_topk = set(np.where(pred_rank < k)[0])

    return len(true_topk & pred_topk) / len(true_topk) if len(true_topk) > 0 else 0


def evaluate_slice(y_true, y_pred, slice_name):
    """Evaluate a single slice and return metrics."""
    n = len(y_true)
    if n == 0:
        return None

    # Basic stats
    gift_rate = (y_true > 0).mean() * 100
    revenue_share = y_true.sum()

    # Spearman correlation
    if (y_true > 0).sum() > 1:
        spearman, _ = stats.spearmanr(y_true, y_pred)
    else:
        spearman = np.nan

    # Revenue capture metrics
    rev_cap_1pct = compute_revenue_capture_at_k(y_true, y_pred, 0.01)
    rev_cap_5pct = compute_revenue_capture_at_k(y_true, y_pred, 0.05)

    # Top-K capture
    top_1pct = compute_top_k_capture(y_true, y_pred, 0.01)
    top_5pct = compute_top_k_capture(y_true, y_pred, 0.05)

    return {
        'slice_name': slice_name,
        'n_samples': n,
        'gift_rate': gift_rate,
        'revenue_share': revenue_share,
        'spearman': float(spearman) if not np.isnan(spearman) else None,
        'rev_cap_1pct': rev_cap_1pct,
        'rev_cap_5pct': rev_cap_5pct,
        'top_1pct': top_1pct,
        'top_5pct': top_5pct,
    }


def create_slice_masks(test_df, train_df, gift_train):
    """Create boolean masks for different slices."""
    slices = {}

    # Full dataset
    slices['all'] = np.ones(len(test_df), dtype=bool)

    # === Cold-start slices ===

    # 1. Cold-start pair: (user, streamer) pairs not in train
    train_pairs = set(zip(train_df['user_id'], train_df['streamer_id']))
    test_pairs = list(zip(test_df['user_id'], test_df['streamer_id']))
    cold_pair_mask = np.array([p not in train_pairs for p in test_pairs])
    slices['cold_pair'] = cold_pair_mask
    slices['warm_pair'] = ~cold_pair_mask

    # 2. Cold-start streamer: streamers with no gifts in train
    gift_train_streamers = set(gift_train['streamer_id'].unique())
    cold_streamer_mask = ~test_df['streamer_id'].isin(gift_train_streamers).values
    slices['cold_streamer'] = cold_streamer_mask
    slices['warm_streamer'] = ~cold_streamer_mask

    # 3. Cold-start user: users with no gifts in train
    gift_train_users = set(gift_train['user_id'].unique())
    cold_user_mask = ~test_df['user_id'].isin(gift_train_users).values
    slices['cold_user'] = cold_user_mask
    slices['warm_user'] = ~cold_user_mask

    # === User tier slices (based on train gift history) ===

    # Compute user gift totals from training period
    user_gift_train = gift_train.groupby('user_id')['gift_price'].sum()

    # Get test user gift history
    test_user_gift = test_df['user_id'].map(user_gift_train).fillna(0).values

    # Define tiers
    if len(user_gift_train) > 0:
        threshold_1pct = np.percentile(user_gift_train.values, 99)
        threshold_10pct = np.percentile(user_gift_train.values, 90)
        threshold_50pct = np.percentile(user_gift_train.values, 50)
    else:
        threshold_1pct = threshold_10pct = threshold_50pct = 0

    # Top-1% users (whales)
    slices['top1pct_user'] = test_user_gift >= threshold_1pct

    # Top-10% users (high value)
    slices['top10pct_user'] = (test_user_gift >= threshold_10pct) & (test_user_gift < threshold_1pct)

    # Middle users (10-50%)
    slices['middle_user'] = (test_user_gift >= threshold_50pct) & (test_user_gift < threshold_10pct)

    # Tail users (bottom 50%)
    slices['tail_user'] = test_user_gift < threshold_50pct

    return slices


def plot_slice_performance(results, save_path):
    """Plot slice performance comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Prepare data
    slice_names = [r['slice_name'] for r in results]
    rev_cap_1pct = [r['rev_cap_1pct'] * 100 for r in results]
    rev_cap_5pct = [r['rev_cap_5pct'] * 100 for r in results]
    spearman = [r['spearman'] if r['spearman'] else 0 for r in results]

    x = np.arange(len(slice_names))
    width = 0.35

    # Plot 1: Revenue Capture
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, rev_cap_1pct, width, label='RevCap@1%', color='steelblue')
    bars2 = ax1.bar(x + width/2, rev_cap_5pct, width, label='RevCap@5%', color='lightsteelblue')
    ax1.set_ylabel('Revenue Capture (%)')
    ax1.set_title('Revenue Capture by Slice')
    ax1.set_xticks(x)
    ax1.set_xticklabels(slice_names, rotation=45, ha='right')
    ax1.legend()
    ax1.axhline(y=rev_cap_1pct[0], color='red', linestyle='--', alpha=0.5, label='All baseline')

    # Plot 2: Spearman
    ax2 = axes[1]
    colors = ['steelblue' if s > spearman[0]*0.5 else 'salmon' for s in spearman]
    bars = ax2.bar(slice_names, spearman, color=colors)
    ax2.set_ylabel('Spearman Correlation')
    ax2.set_title('Ranking Quality by Slice')
    ax2.set_xticklabels(slice_names, rotation=45, ha='right')
    ax2.axhline(y=spearman[0], color='red', linestyle='--', alpha=0.5)

    # Plot 3: Sample distribution and revenue share
    ax3 = axes[2]
    n_samples = [r['n_samples'] for r in results]
    revenue_share = [r['revenue_share'] for r in results]
    total_revenue = results[0]['revenue_share']
    revenue_pct = [r / total_revenue * 100 for r in revenue_share]

    ax3_twin = ax3.twinx()
    bars1 = ax3.bar(x - width/2, [n/1000 for n in n_samples], width, label='Samples (K)', color='steelblue', alpha=0.7)
    bars2 = ax3_twin.bar(x + width/2, revenue_pct, width, label='Revenue %', color='green', alpha=0.7)
    ax3.set_ylabel('Samples (K)', color='steelblue')
    ax3_twin.set_ylabel('Revenue Share (%)', color='green')
    ax3.set_title('Slice Distribution')
    ax3.set_xticks(x)
    ax3.set_xticklabels(slice_names, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    log_message(f"Saved slice performance plot to {save_path}")


def plot_coldstart_analysis(test_df, y_true, y_pred, train_pairs_count, save_path):
    """Plot cold-start analysis: performance vs pair history."""
    # Bin by pair gift count in train
    bins = [0, 1, 5, 20, 100, np.inf]
    labels = ['0', '1-5', '6-20', '21-100', '100+']

    test_df = test_df.copy()
    test_df['pair_history_bin'] = pd.cut(train_pairs_count, bins=bins, labels=labels, right=False)
    test_df['y_true'] = y_true
    test_df['y_pred'] = y_pred

    results = []
    for label in labels:
        mask = test_df['pair_history_bin'] == label
        if mask.sum() > 0:
            y_t = test_df.loc[mask, 'y_true'].values
            y_p = test_df.loc[mask, 'y_pred'].values

            rev_cap = compute_revenue_capture_at_k(y_t, y_p, 0.01)
            spearman, _ = stats.spearmanr(y_t, y_p) if (y_t > 0).sum() > 1 else (np.nan, None)

            results.append({
                'bin': label,
                'n_samples': mask.sum(),
                'gift_rate': (y_t > 0).mean() * 100,
                'rev_cap_1pct': rev_cap * 100,
                'spearman': spearman if not np.isnan(spearman) else 0
            })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: RevCap by history bin
    ax1 = axes[0]
    bins_plot = [r['bin'] for r in results]
    rev_caps = [r['rev_cap_1pct'] for r in results]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(results)))
    ax1.bar(bins_plot, rev_caps, color=colors)
    ax1.set_xlabel('Pair Gift Count in Train')
    ax1.set_ylabel('Revenue Capture@1% (%)')
    ax1.set_title('Performance vs Pair History')

    # Add sample counts as text
    for i, r in enumerate(results):
        ax1.text(i, rev_caps[i] + 0.5, f'n={r["n_samples"]//1000}K', ha='center', fontsize=8)

    # Plot 2: Gift rate by history bin
    ax2 = axes[1]
    gift_rates = [r['gift_rate'] for r in results]
    ax2.bar(bins_plot, gift_rates, color=colors)
    ax2.set_xlabel('Pair Gift Count in Train')
    ax2.set_ylabel('Gift Rate (%)')
    ax2.set_title('Gift Rate vs Pair History')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    log_message(f"Saved cold-start analysis plot to {save_path}")

    return results


def main():
    log_message("="*60)
    log_message("MVP-1.4: Slice Evaluation")
    log_message("="*60)

    start_time = time.time()

    # Load model
    model_path = MODELS_DIR / 'direct_frozen_v2_20260118.pkl'
    if not model_path.exists():
        log_message(f"Model not found at {model_path}. Please run train_leakage_free_baseline_v2.py first.", "ERROR")
        return

    log_message("Loading model...")
    model = pickle.load(open(model_path, 'rb'))

    # Load data
    log_message("Loading data...")
    gift = pd.read_csv(DATA_DIR / "gift.csv")
    user = pd.read_csv(DATA_DIR / "user.csv")
    streamer = pd.read_csv(DATA_DIR / "streamer.csv")
    room = pd.read_csv(DATA_DIR / "room.csv")
    click = pd.read_csv(DATA_DIR / "click.csv")

    # Prepare click-level data (same as training)
    log_message("Preparing data...")
    click_base = click.copy()
    click_base['timestamp_dt'] = pd.to_datetime(click_base['timestamp'], unit='ms')
    click_base['hour'] = click_base['timestamp_dt'].dt.hour
    click_base['day_of_week'] = click_base['timestamp_dt'].dt.dayofweek
    click_base['is_weekend'] = (click_base['day_of_week'] >= 5).astype(int)

    # V2: Watch time truncation
    max_window_ms = 1 * 3600 * 1000
    click_base['effective_window_ms'] = np.minimum(click_base['watch_live_time'], max_window_ms)
    click_base['click_end_ts'] = click_base['timestamp_dt'] + pd.to_timedelta(click_base['effective_window_ms'], unit='ms')

    gift_ts = gift.copy()
    gift_ts['timestamp_dt'] = pd.to_datetime(gift_ts['timestamp'], unit='ms')

    merged = click_base[['user_id', 'streamer_id', 'live_id', 'timestamp_dt', 'click_end_ts']].merge(
        gift_ts[['user_id', 'streamer_id', 'live_id', 'timestamp_dt', 'gift_price']],
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
    }).reset_index().rename(columns={'timestamp_dt_click': 'timestamp_dt', 'gift_price': 'gift_price_label'})

    click_base = click_base.merge(gift_agg, on=['user_id', 'streamer_id', 'live_id', 'timestamp_dt'], how='left')
    click_base['gift_price_label'] = click_base['gift_price_label'].fillna(0)

    # Temporal split
    click_base_sorted = click_base.sort_values('timestamp').reset_index(drop=True)
    train_ratio, val_ratio = 0.70, 0.15
    n = len(click_base_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = click_base_sorted.iloc[:train_end].copy()
    test_df = click_base_sorted.iloc[val_end:].copy()

    log_message(f"Train: {len(train_df):,}, Test: {len(test_df):,}")

    # Gift data in train period
    train_max_ts = train_df['timestamp'].max()
    gift_train = gift[gift['timestamp'] <= train_max_ts].copy()
    log_message(f"Gift records in train period: {len(gift_train):,}")

    # Prepare features for test (using train lookups)
    log_message("Preparing features...")

    # Import feature preparation functions from baseline script
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from train_leakage_free_baseline_v2 import (
        create_static_features,
        create_past_only_features_frozen,
        apply_frozen_features,
        get_feature_columns
    )

    user_features, streamer_features, room_info = create_static_features(user, streamer, room)

    test_df = test_df.merge(user_features, on='user_id', how='left')
    test_df = test_df.merge(streamer_features, on='streamer_id', how='left')
    test_df = test_df.merge(room_info, on='live_id', how='left')

    lookups = create_past_only_features_frozen(gift, click, train_df)
    test_df = apply_frozen_features(test_df, lookups)

    # Fill NaN and convert categorical
    numeric_cols = test_df.select_dtypes(include=[np.number]).columns
    test_df[numeric_cols] = test_df[numeric_cols].fillna(0)

    cat_columns = ['age', 'gender', 'device_brand', 'device_price', 'fans_num', 'follow_num',
                   'accu_watch_live_cnt', 'accu_watch_live_duration',
                   'streamer_gender', 'streamer_age', 'streamer_device_brand', 'streamer_device_price',
                   'live_operation_tag', 'fans_user_num', 'fans_group_fans_num', 'follow_user_num',
                   'accu_live_cnt', 'accu_live_duration', 'accu_play_cnt', 'accu_play_duration',
                   'live_type', 'live_content_category']

    for col in cat_columns:
        if col in test_df.columns and test_df[col].dtype == 'object':
            test_df[col] = test_df[col].fillna('unknown')
            test_df[col] = test_df[col].astype('category').cat.codes

    test_df['target'] = np.log1p(test_df['gift_price_label'])
    test_df['target_raw'] = test_df['gift_price_label']

    # Get predictions
    feature_cols = get_feature_columns(test_df)
    X_test = test_df[feature_cols]
    y_pred_log = model.predict(X_test)
    y_pred_raw = np.expm1(y_pred_log)
    y_true = test_df['target_raw'].values

    log_message(f"Predictions generated for {len(test_df):,} test samples")

    # Create slice masks
    log_message("\nCreating slice masks...")
    slices = create_slice_masks(test_df, train_df, gift_train)

    # Evaluate each slice
    log_message("\n" + "="*60)
    log_message("SLICE EVALUATION RESULTS")
    log_message("="*60)

    results = []

    # Define evaluation order
    slice_order = ['all', 'cold_pair', 'warm_pair', 'cold_streamer', 'warm_streamer',
                   'cold_user', 'warm_user', 'top1pct_user', 'top10pct_user', 'middle_user', 'tail_user']

    for slice_name in slice_order:
        if slice_name not in slices:
            continue

        mask = slices[slice_name]
        if mask.sum() == 0:
            log_message(f"  {slice_name}: No samples", "WARNING")
            continue

        slice_result = evaluate_slice(y_true[mask], y_pred_raw[mask], slice_name)
        if slice_result:
            results.append(slice_result)

            log_message(f"\n📊 {slice_name}:")
            log_message(f"   Samples: {slice_result['n_samples']:,} ({slice_result['n_samples']/len(test_df)*100:.1f}%)")
            log_message(f"   Gift Rate: {slice_result['gift_rate']:.2f}%")
            log_message(f"   RevCap@1%: {slice_result['rev_cap_1pct']*100:.1f}%")
            log_message(f"   RevCap@5%: {slice_result['rev_cap_5pct']*100:.1f}%")
            if slice_result['spearman']:
                log_message(f"   Spearman: {slice_result['spearman']:.4f}")

    # Compute pair history for cold-start analysis
    log_message("\n" + "="*60)
    log_message("COLD-START ANALYSIS (by pair history)")
    log_message("="*60)

    # Get pair gift counts from train
    pair_gift_counts = gift_train.groupby(['user_id', 'streamer_id']).size()
    test_pairs = list(zip(test_df['user_id'], test_df['streamer_id']))
    train_pairs_count = np.array([pair_gift_counts.get(p, 0) for p in test_pairs])

    coldstart_results = plot_coldstart_analysis(
        test_df, y_true, y_pred_raw, train_pairs_count,
        IMG_DIR / 'coldstart_analysis.png'
    )

    # Plot slice performance
    plot_slice_performance(results, IMG_DIR / 'slice_performance_comparison.png')

    # Save results (convert numpy types to Python native)
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    output = convert_numpy({
        'timestamp': datetime.now().isoformat(),
        'model': 'direct_frozen_v2',
        'n_test_samples': len(test_df),
        'slice_results': results,
        'coldstart_analysis': coldstart_results,
        'baseline_revcap_1pct': results[0]['rev_cap_1pct'] if results else None,
    })

    results_path = RESULTS_DIR / 'slice_evaluation_20260118.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - start_time

    log_message("\n" + "="*60)
    log_message("SUMMARY")
    log_message("="*60)

    # Key findings
    all_result = next((r for r in results if r['slice_name'] == 'all'), None)
    cold_pair_result = next((r for r in results if r['slice_name'] == 'cold_pair'), None)
    warm_pair_result = next((r for r in results if r['slice_name'] == 'warm_pair'), None)

    if all_result and cold_pair_result:
        cold_vs_all = cold_pair_result['rev_cap_1pct'] / all_result['rev_cap_1pct'] if all_result['rev_cap_1pct'] > 0 else 0
        log_message(f"Cold-pair RevCap@1%: {cold_pair_result['rev_cap_1pct']*100:.1f}% ({cold_vs_all*100:.0f}% of baseline)")

    if warm_pair_result:
        warm_vs_all = warm_pair_result['rev_cap_1pct'] / all_result['rev_cap_1pct'] if all_result['rev_cap_1pct'] > 0 else 0
        log_message(f"Warm-pair RevCap@1%: {warm_pair_result['rev_cap_1pct']*100:.1f}% ({warm_vs_all*100:.0f}% of baseline)")

    log_message(f"\nTotal time: {elapsed:.1f}s")
    log_message(f"Results saved to: {results_path}")
    log_message("="*60, "SUCCESS")


if __name__ == '__main__':
    main()
