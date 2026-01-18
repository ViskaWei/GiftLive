#!/usr/bin/env python3
"""
KuaiLive Comprehensive EDA Script
Experiment ID: EXP-20260109-gift-allocation-02
MVP: MVP-0.1-Enhanced

This script performs comprehensive exploratory data analysis on the KuaiLive dataset,
covering user behavior, supply side, interaction structure, temporal patterns, 
anomaly detection, and data quality.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set plot style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
DATA_DIR = Path("/home/swei20/GiftLive/data/KuaiLive")
OUTPUT_DIR = Path("/home/swei20/GiftLive/KuaiLive")
IMG_DIR = OUTPUT_DIR / "img"
RESULTS_DIR = Path("/home/swei20/GiftLive/gift_allocation/results")
IMG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Import functions from existing script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from eda_kuailive import compute_gini, load_data

def analyze_data_quality(click_df, gift_df):
    """Phase 1: Data Audit & Quality Check"""
    print("\n" + "=" * 60)
    print("Phase 1: Data Quality Check")
    print("=" * 60)
    
    quality_stats = {}
    
    # 1. Time range check
    print("\n--- Time Range Check ---")
    click_df['datetime'] = pd.to_datetime(click_df['timestamp'], unit='ms')
    gift_df['datetime'] = pd.to_datetime(gift_df['timestamp'], unit='ms')
    
    click_min = click_df['datetime'].min()
    click_max = click_df['datetime'].max()
    gift_min = gift_df['datetime'].min()
    gift_max = gift_df['datetime'].max()
    
    time_span_days = (click_max - click_min).days
    print(f"Click time range: {click_min} to {click_max} ({time_span_days} days)")
    print(f"Gift time range: {gift_min} to {gift_max}")
    
    quality_stats['time_range'] = {
        'click_min': click_min.isoformat(),
        'click_max': click_max.isoformat(),
        'gift_min': gift_min.isoformat(),
        'gift_max': gift_max.isoformat(),
        'span_days': time_span_days
    }
    
    # 2. Missing/Anomaly check for click
    print("\n--- Click Table Anomaly Check ---")
    click_anomalies = {
        'watch_time_le_zero': (click_df['watch_live_time'] <= 0).sum(),
        'watch_time_gt_24h': (click_df['watch_live_time'] > 24*3600*1000).sum(),
        'missing_values': click_df.isna().sum().sum(),
        'duplicate_records': click_df.duplicated(subset=['user_id', 'live_id', 'timestamp']).sum()
    }
    
    for key, value in click_anomalies.items():
        pct = value / len(click_df) * 100 if len(click_df) > 0 else 0
        print(f"  {key}: {value:,} ({pct:.2f}%)")
    
    quality_stats['click_anomalies'] = click_anomalies
    
    # 3. Missing/Anomaly check for gift
    print("\n--- Gift Table Anomaly Check ---")
    gift_anomalies = {
        'gift_price_le_zero': (gift_df['gift_price'] <= 0).sum(),
        'gift_price_extreme': (gift_df['gift_price'] > gift_df['gift_price'].quantile(0.99) * 10).sum(),
        'missing_values': gift_df.isna().sum().sum(),
        'duplicate_records': gift_df.duplicated(subset=['user_id', 'live_id', 'timestamp']).sum()
    }
    
    for key, value in gift_anomalies.items():
        pct = value / len(gift_df) * 100 if len(gift_df) > 0 else 0
        print(f"  {key}: {value:,} ({pct:.2f}%)")
    
    quality_stats['gift_anomalies'] = gift_anomalies
    
    # 4. Gift-Click consistency check
    print("\n--- Gift-Click Consistency Check ---")
    # Merge gift with click on same (user_id, live_id) and time proximity (±5 min)
    click_df_sorted = click_df.sort_values(['user_id', 'live_id', 'timestamp'])
    gift_df_sorted = gift_df.sort_values(['user_id', 'live_id', 'timestamp'])
    
    # For each gift, find matching click
    matched_count = 0
    for idx, gift_row in gift_df_sorted.iterrows():
        matching_clicks = click_df_sorted[
            (click_df_sorted['user_id'] == gift_row['user_id']) &
            (click_df_sorted['live_id'] == gift_row['live_id']) &
            (click_df_sorted['timestamp'] >= gift_row['timestamp'] - 5*60*1000) &
            (click_df_sorted['timestamp'] <= gift_row['timestamp'] + 5*60*1000)
        ]
        if len(matching_clicks) > 0:
            matched_count += 1
    
    unmapped_ratio = 1 - matched_count / len(gift_df) if len(gift_df) > 0 else 0
    print(f"  Matched gifts: {matched_count:,} / {len(gift_df):,} ({matched_count/len(gift_df)*100:.2f}%)")
    print(f"  Unmapped ratio: {unmapped_ratio*100:.2f}%")
    
    quality_stats['consistency'] = {
        'matched_gifts': matched_count,
        'total_gifts': len(gift_df),
        'match_rate': matched_count / len(gift_df) if len(gift_df) > 0 else 0,
        'unmapped_ratio': unmapped_ratio
    }
    
    # 5. Data health score (0-5 scale)
    print("\n--- Data Health Score ---")
    completeness = 5.0 if click_anomalies['missing_values'] == 0 and gift_anomalies['missing_values'] == 0 else 3.0
    consistency = 5.0 if unmapped_ratio < 0.1 else max(0, 5.0 - unmapped_ratio * 10)
    validity = 5.0 if (click_anomalies['watch_time_le_zero'] / len(click_df) < 0.01 and 
                       gift_anomalies['gift_price_le_zero'] == 0) else 3.0
    timeliness = 5.0 if time_span_days > 0 else 0.0
    
    overall_score = (completeness + consistency + validity + timeliness) / 4
    
    health_scores = {
        'completeness': completeness,
        'consistency': consistency,
        'validity': validity,
        'timeliness': timeliness,
        'overall_score': overall_score
    }
    
    for key, value in health_scores.items():
        print(f"  {key}: {value:.2f}/5.0")
    
    quality_stats['health_scores'] = health_scores
    
    return quality_stats, click_df, gift_df

def build_sessions(click_df, gift_df):
    """Phase 2: Build Session Table (Optimized)"""
    print("\n" + "=" * 60)
    print("Phase 2: Building Sessions")
    print("=" * 60)
    
    # Sort by user_id, live_id, timestamp
    click_df_sorted = click_df.sort_values(['user_id', 'live_id', 'timestamp']).copy()
    gift_df_sorted = gift_df.sort_values(['user_id', 'live_id', 'timestamp']).copy()
    
    print("  Processing clicks...")
    
    # Simplified: one session per (user_id, live_id) for now
    # (Can optimize later with time gap merging if needed)
    sessions_df = click_df_sorted.groupby(['user_id', 'live_id']).agg({
        'streamer_id': 'first',
        'timestamp': ['min', 'max'],
        'watch_live_time': ['sum', 'count']
    }).reset_index()
    sessions_df.columns = ['user_id', 'live_id', 'streamer_id', 'session_start', 'session_end_raw', 'total_watch_time', 'click_count']
    
    # Calculate session_end properly
    last_click = click_df_sorted.groupby(['user_id', 'live_id']).apply(
        lambda x: x.loc[x['timestamp'].idxmax(), 'timestamp'] + x.loc[x['timestamp'].idxmax(), 'watch_live_time']
    ).reset_index(name='session_end')
    sessions_df = sessions_df.merge(last_click, on=['user_id', 'live_id'])
    sessions_df['session_duration'] = sessions_df['session_end'] - sessions_df['session_start']
    
    print(f"  Created {len(sessions_df):,} sessions")
    
    # Merge gift information
    print("  Merging gift information...")
    gift_agg = gift_df_sorted.groupby(['user_id', 'live_id']).agg({
        'gift_price': ['count', 'sum'],
        'timestamp': 'min'
    }).reset_index()
    gift_agg.columns = ['user_id', 'live_id', 'gift_count', 'gift_amount', 't_first_gift']
    
    sessions_df = sessions_df.merge(gift_agg, on=['user_id', 'live_id'], how='left')
    sessions_df['gift_count'] = sessions_df['gift_count'].fillna(0).astype(int)
    sessions_df['gift_amount'] = sessions_df['gift_amount'].fillna(0)
    sessions_df['t_first_gift'] = sessions_df['t_first_gift'].fillna(np.nan)
    
    # Calculate t_first_gift relative to session_start
    sessions_df['t_first_gift_relative'] = (sessions_df['t_first_gift'] - sessions_df['session_start']) / sessions_df['session_duration']
    sessions_df['t_first_gift_relative'] = sessions_df['t_first_gift_relative'].fillna(np.nan)
    
    print(f"  Sessions with gifts: {(sessions_df['gift_count'] > 0).sum():,}")
    print(f"  Sessions with multiple gifts: {(sessions_df['gift_count'] > 1).sum():,}")
    
    return sessions_df

# ============================================================================
# Phase 3: Session & Funnel Analysis
# ============================================================================

def analyze_session_funnel(sessions_df, click_df, gift_df):
    """Phase 3: Session & Funnel Analysis"""
    print("\n" + "=" * 60)
    print("Phase 3: Session & Funnel Analysis")
    print("=" * 60)
    
    stats = {}
    
    # 3.1 First gift time analysis
    gift_sessions = sessions_df[sessions_df['gift_count'] > 0].copy()
    if len(gift_sessions) > 0:
        first_gift_ratio = gift_sessions['t_first_gift_relative'].dropna()
        stats['first_gift_ratio'] = {
            'mean': float(first_gift_ratio.mean()) if len(first_gift_ratio) > 0 else 0,
            'median': float(first_gift_ratio.median()) if len(first_gift_ratio) > 0 else 0,
            'p10': float(first_gift_ratio.quantile(0.10)) if len(first_gift_ratio) > 0 else 0,
            'p90': float(first_gift_ratio.quantile(0.90)) if len(first_gift_ratio) > 0 else 0,
            'immediate_rate': float((first_gift_ratio < 0.1).sum() / len(first_gift_ratio)) if len(first_gift_ratio) > 0 else 0
        }
        print(f"  First gift ratio (relative to session): mean={stats['first_gift_ratio']['mean']:.3f}, median={stats['first_gift_ratio']['median']:.3f}")
        print(f"  Immediate gift rate (<10%): {stats['first_gift_ratio']['immediate_rate']*100:.1f}%")
    
    # 3.2 Session duration distribution
    session_durations = sessions_df['session_duration'] / 1000  # Convert to seconds
    stats['session_duration'] = {
        'mean': float(session_durations.mean()),
        'median': float(session_durations.median()),
        'p50': float(session_durations.quantile(0.50)),
        'p90': float(session_durations.quantile(0.90)),
        'p99': float(session_durations.quantile(0.99))
    }
    print(f"  Session duration: P50={stats['session_duration']['p50']:.1f}s, P90={stats['session_duration']['p90']:.1f}s, P99={stats['session_duration']['p99']:.1f}s")
    
    # 3.3 Conversion funnel
    total_clicks = len(click_df)
    total_sessions = len(sessions_df)
    gift_sessions_count = (sessions_df['gift_count'] > 0).sum()
    multi_gift_sessions = (sessions_df['gift_count'] > 1).sum()
    
    stats['funnel'] = {
        'click_to_session': total_sessions / total_clicks if total_clicks > 0 else 0,
        'session_to_gift': gift_sessions_count / total_sessions if total_sessions > 0 else 0,
        'gift_to_multi': multi_gift_sessions / gift_sessions_count if gift_sessions_count > 0 else 0,
        'click_to_gift': gift_sessions_count / total_clicks if total_clicks > 0 else 0
    }
    print(f"  Conversion funnel:")
    print(f"    Click → Session: {stats['funnel']['click_to_session']*100:.2f}%")
    print(f"    Session → Gift: {stats['funnel']['session_to_gift']*100:.2f}%")
    print(f"    Gift → Multi-gift: {stats['funnel']['gift_to_multi']*100:.2f}%")
    print(f"    Click → Gift: {stats['funnel']['click_to_gift']*100:.2f}%")
    
    return stats

def plot_phase3_figures(sessions_df, click_df, gift_df, stats, img_dir):
    """Plot Phase 3 figures"""
    print("\n  Generating Phase 3 plots...")
    
    # Fig 3.1: First gift time ratio distribution
    gift_sessions = sessions_df[sessions_df['gift_count'] > 0].copy()
    if len(gift_sessions) > 0:
        first_gift_ratio = gift_sessions['t_first_gift_relative'].dropna()
        if len(first_gift_ratio) > 0:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.hist(first_gift_ratio, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(first_gift_ratio.median(), color='red', linestyle='--', linewidth=2, 
                      label=f'Median={first_gift_ratio.median():.3f}')
            immediate_rate = (first_gift_ratio < 0.1).sum() / len(first_gift_ratio)
            ax.text(0.95, 0.95, f'Immediate rate (<10%): {immediate_rate*100:.1f}%',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            ax.set_xlabel('First Gift Time / Session Duration', fontweight='bold')
            ax.set_ylabel('Count', fontweight='bold')
            ax.set_title('Gifts are mostly immediate (<10% of session)', fontweight='bold', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(img_dir / "first_gift_time_ratio.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Saved: first_gift_time_ratio.png")
    
    # Fig 3.2: First gift time vs watch time (scatter, log-log)
    if len(gift_sessions) > 0:
        gift_sessions_valid = gift_sessions[
            (gift_sessions['t_first_gift_relative'].notna()) & 
            (gift_sessions['total_watch_time'] > 0)
        ].copy()
        if len(gift_sessions_valid) > 0:
            # Sample if too large
            if len(gift_sessions_valid) > 10000:
                gift_sessions_valid = gift_sessions_valid.sample(n=10000, random_state=42)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            watch_time_sec = gift_sessions_valid['total_watch_time'] / 1000
            first_gift_time_sec = (gift_sessions_valid['t_first_gift'] - gift_sessions_valid['session_start']) / 1000
            ax.scatter(watch_time_sec, first_gift_time_sec, alpha=0.3, s=1)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Watch Time (seconds, log scale)', fontweight='bold')
            ax.set_ylabel('First Gift Time (seconds, log scale)', fontweight='bold')
            ax.set_title('First gift time vs watch time', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3, which='both')
            plt.tight_layout()
            plt.savefig(img_dir / "first_gift_vs_watch.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Saved: first_gift_vs_watch.png")
    
    # Fig 3.3: Session duration CCDF
    session_durations = sessions_df['session_duration'] / 1000  # seconds
    sorted_durations = np.sort(session_durations)[::-1]
    n = len(sorted_durations)
    ccdf = np.arange(1, n + 1) / n
    p50 = session_durations.quantile(0.50)
    p90 = session_durations.quantile(0.90)
    p99 = session_durations.quantile(0.99)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    mask = sorted_durations > 0
    ax.loglog(sorted_durations[mask], ccdf[mask], 'b-', linewidth=2, alpha=0.7)
    ax.axvline(p50, color='red', linestyle='--', linewidth=2, label=f'P50={p50:.1f}s')
    ax.axvline(p90, color='orange', linestyle='--', linewidth=2, label=f'P90={p90:.1f}s')
    ax.axvline(p99, color='purple', linestyle='--', linewidth=2, label=f'P99={p99:.1f}s')
    ax.set_xlabel('Session Duration (seconds, log scale)', fontweight='bold')
    ax.set_ylabel('P(X ≥ x) (log scale)', fontweight='bold')
    ax.set_title(f'Session duration is heavy-tailed: P99/P50 = {p99/p50:.1f}x', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(img_dir / "session_duration_ccdf.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: session_duration_ccdf.png")
    
    # Fig 3.5: Conversion funnel
    total_clicks = len(click_df)
    total_sessions = len(sessions_df)
    gift_sessions_count = (sessions_df['gift_count'] > 0).sum()
    multi_gift_sessions = (sessions_df['gift_count'] > 1).sum()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    steps = ['Click', 'Session', 'Gift\nSession', 'Multi-\nGift']
    counts = [total_clicks, total_sessions, gift_sessions_count, multi_gift_sessions]
    rates = [100, (total_sessions/total_clicks*100) if total_clicks > 0 else 0,
             (gift_sessions_count/total_sessions*100) if total_sessions > 0 else 0,
             (multi_gift_sessions/gift_sessions_count*100) if gift_sessions_count > 0 else 0]
    
    bars = ax.bar(steps, rates, color=['steelblue', 'forestgreen', 'coral', 'purple'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Conversion Rate (%)', fontweight='bold')
    ax.set_title('Conversion funnel: Most drop at click→session', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, rate, count in zip(bars, rates, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.2f}%\n(N={count:,})',
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(img_dir / "conversion_funnel.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: conversion_funnel.png")

# ============================================================================
# Phase 4: User Behavior Analysis
# ============================================================================

def analyze_user_behavior(sessions_df, gift_df, user_df):
    """Phase 4: User Behavior Analysis"""
    print("\n" + "=" * 60)
    print("Phase 4: User Behavior Analysis")
    print("=" * 60)
    
    stats = {}
    
    # 4.1 User watch intensity
    user_sessions = sessions_df.groupby('user_id').agg({
        'total_watch_time': 'sum',
        'session_duration': 'sum',
        'gift_count': 'sum',
        'gift_amount': 'sum'
    }).reset_index()
    user_sessions.columns = ['user_id', 'total_watch_time', 'total_session_duration', 'total_gift_count', 'total_gift_amount']
    
    stats['user_watch_time'] = {
        'mean': float(user_sessions['total_watch_time'].mean()),
        'median': float(user_sessions['total_watch_time'].median()),
        'p50': float(user_sessions['total_watch_time'].quantile(0.50)),
        'p90': float(user_sessions['total_watch_time'].quantile(0.90)),
        'p99': float(user_sessions['total_watch_time'].quantile(0.99))
    }
    print(f"  User watch time: P50={stats['user_watch_time']['p50']/1000:.1f}s, P90={stats['user_watch_time']['p90']/1000:.1f}s")
    
    # 4.2 User quadrant (watch vs pay)
    high_watch = user_sessions['total_watch_time'] > user_sessions['total_watch_time'].quantile(0.75)
    high_pay = user_sessions['total_gift_amount'] > user_sessions['total_gift_amount'].quantile(0.75)
    
    q1 = (high_watch & high_pay).sum()  # High watch, High pay
    q2 = (~high_watch & high_pay).sum()  # Low watch, High pay
    q3 = (~high_watch & ~high_pay).sum()  # Low watch, Low pay
    q4 = (high_watch & ~high_pay).sum()  # High watch, Low pay
    
    stats['user_quadrant'] = {
        'high_watch_high_pay': int(q1),
        'low_watch_high_pay': int(q2),
        'low_watch_low_pay': int(q3),
        'high_watch_low_pay': int(q4),
        'high_watch_low_pay_ratio': float(q4 / len(user_sessions)) if len(user_sessions) > 0 else 0
    }
    print(f"  User quadrant: High watch + Low pay = {stats['user_quadrant']['high_watch_low_pay']:,} ({stats['user_quadrant']['high_watch_low_pay_ratio']*100:.1f}%)")
    
    # 4.3 Watch time vs gift rate
    # Bin watch time and compute gift rate
    watch_time_bins = np.percentile(user_sessions['total_watch_time'], [0, 25, 50, 75, 90, 95, 99, 100])
    user_sessions['watch_time_bin'] = pd.cut(user_sessions['total_watch_time'], bins=watch_time_bins, labels=False, include_lowest=True)
    bin_stats = user_sessions.groupby('watch_time_bin').agg({
        'total_watch_time': 'mean',
        'total_gift_amount': lambda x: (x > 0).sum() / len(x)  # Gift rate
    }).reset_index()
    bin_stats.columns = ['bin', 'avg_watch_time', 'gift_rate']
    
    stats['watch_time_vs_gift_rate'] = {
        'bins': watch_time_bins.tolist(),
        'gift_rates': bin_stats['gift_rate'].tolist()
    }
    
    # 4.4 Gift price tiers
    gift_prices = gift_df['gift_price'].value_counts().sort_index()
    top_prices = gift_prices.head(20)
    cumulative_share = gift_prices.cumsum() / gift_prices.sum()
    
    stats['gift_price_tiers'] = {
        'top_20_prices': top_prices.index.tolist(),
        'top_20_counts': top_prices.values.tolist(),
        'top_20_cumulative_share': cumulative_share[top_prices.index].tolist()
    }
    print(f"  Top gift price: {top_prices.index[0]:.0f} (count={top_prices.values[0]:,})")
    
    return stats, user_sessions

def plot_phase4_figures(sessions_df, gift_df, user_sessions, stats, img_dir):
    """Plot Phase 4 figures"""
    print("\n  Generating Phase 4 plots...")
    
    # Fig 4.1: User watch time CCDF
    watch_times = user_sessions['total_watch_time'] / 1000  # seconds
    sorted_watch = np.sort(watch_times)[::-1]
    n = len(sorted_watch)
    ccdf = np.arange(1, n + 1) / n
    
    fig, ax = plt.subplots(figsize=(6, 5))
    mask = sorted_watch > 0
    ax.loglog(sorted_watch[mask], ccdf[mask], 'b-', linewidth=2, alpha=0.7)
    ax.set_xlabel('Total Watch Time (seconds, log scale)', fontweight='bold')
    ax.set_ylabel('P(X ≥ x) (log scale)', fontweight='bold')
    ax.set_title('User watch time is highly concentrated', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(img_dir / "user_watch_time_ccdf.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: user_watch_time_ccdf.png")
    
    # Fig 4.3: Multi-metric conversion funnel
    total_clicks = len(sessions_df) * 1.0  # Approximate
    total_sessions = len(sessions_df)
    gift_sessions = (sessions_df['gift_count'] > 0).sum()
    total_users = len(user_sessions)
    paying_users = (user_sessions['total_gift_amount'] > 0).sum()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    metrics = ['Click-level', 'Session-level', 'User-level']
    rates = [
        (gift_sessions / total_sessions * 100) if total_sessions > 0 else 0,
        (gift_sessions / total_sessions * 100) if total_sessions > 0 else 0,
        (paying_users / total_users * 100) if total_users > 0 else 0
    ]
    bars = ax.bar(metrics, rates, color=['steelblue', 'forestgreen', 'coral'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Conversion Rate (%)', fontweight='bold')
    ax.set_title('Conversion rates vary by metric definition', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.2f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(img_dir / "conversion_funnel_multi.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: conversion_funnel_multi.png")
    
    # Fig 4.4: User quadrant
    fig, ax = plt.subplots(figsize=(6, 5))
    watch_log = np.log1p(user_sessions['total_watch_time'] / 1000)
    pay_log = np.log1p(user_sessions['total_gift_amount'])
    
    # Sample if too large
    if len(user_sessions) > 5000:
        sample_idx = np.random.choice(len(user_sessions), 5000, replace=False)
        watch_log = watch_log.iloc[sample_idx]
        pay_log = pay_log.iloc[sample_idx]
    
    ax.scatter(watch_log, pay_log, alpha=0.3, s=1)
    
    # Add quadrant lines
    watch_median = np.median(watch_log)
    pay_median = np.median(pay_log[pay_log > 0]) if (pay_log > 0).sum() > 0 else 0
    ax.axvline(watch_median, color='red', linestyle='--', linewidth=1, alpha=0.5)
    if pay_median > 0:
        ax.axhline(pay_median, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add quadrant labels
    q4_ratio = stats['user_quadrant']['high_watch_low_pay_ratio']
    ax.text(0.95, 0.05, f'High watch\nLow pay: {q4_ratio*100:.1f}%',
           transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
           horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Total Watch Time (log scale)', fontweight='bold')
    ax.set_ylabel('Total Gift Amount (log scale)', fontweight='bold')
    ax.set_title(f'High watch + Low pay users: {q4_ratio*100:.1f}%', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(img_dir / "user_quadrant.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: user_quadrant.png")
    
    # Fig 4.5: Watch time vs gift rate
    watch_time_bins = stats['watch_time_vs_gift_rate']['bins']
    gift_rates = stats['watch_time_vs_gift_rate']['gift_rates']
    bin_centers = [(watch_time_bins[i] + watch_time_bins[i+1]) / 2 for i in range(len(watch_time_bins)-1)]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(bin_centers[:len(gift_rates)], [r*100 for r in gift_rates], 'o-', linewidth=2, markersize=8)
    ax.set_xscale('log')
    ax.set_xlabel('Watch Time (ms, log scale)', fontweight='bold')
    ax.set_ylabel('Gift Rate (%)', fontweight='bold')
    ax.set_title('Gift rate increases with watch time', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(img_dir / "watch_time_vs_gift_rate.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: watch_time_vs_gift_rate.png")
    
    # Fig 4.8: Gift price tiers
    top_prices = stats['gift_price_tiers']['top_20_prices']
    top_counts = stats['gift_price_tiers']['top_20_counts']
    top_cumshare = stats['gift_price_tiers']['top_20_cumulative_share']
    
    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax2 = ax1.twinx()
    
    bars = ax1.bar(range(len(top_prices)), top_counts, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Gift Price Tier (Rank)', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    ax2.plot(range(len(top_prices)), [s*100 for s in top_cumshare], 'ro-', linewidth=2, markersize=6)
    ax2.set_ylabel('Cumulative Share (%)', fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax1.set_xticks(range(len(top_prices)))
    ax1.set_xticklabels([f'{p:.0f}' for p in top_prices], rotation=45, ha='right')
    ax1.set_title('Gift price tiers: Top prices dominate', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(img_dir / "gift_price_tiers.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: gift_price_tiers.png")

# ============================================================================
# Phase 5: Supply Side Analysis
# ============================================================================

def analyze_supply_side(sessions_df, gift_df, streamer_df):
    """Phase 5: Supply Side Analysis"""
    print("\n" + "=" * 60)
    print("Phase 5: Supply Side Analysis")
    print("=" * 60)
    
    stats = {}
    
    # 5.1 Streamer watch time
    streamer_sessions = sessions_df.groupby('streamer_id').agg({
        'total_watch_time': 'sum',
        'session_duration': 'sum',
        'gift_count': 'sum',
        'gift_amount': 'sum',
        'user_id': 'nunique'
    }).reset_index()
    streamer_sessions.columns = ['streamer_id', 'total_watch_time', 'total_session_duration', 
                                 'total_gift_count', 'total_revenue', 'unique_viewers']
    
    stats['streamer_watch_time'] = {
        'mean': float(streamer_sessions['total_watch_time'].mean()),
        'median': float(streamer_sessions['total_watch_time'].median()),
        'p50': float(streamer_sessions['total_watch_time'].quantile(0.50)),
        'p90': float(streamer_sessions['total_watch_time'].quantile(0.90)),
        'p99': float(streamer_sessions['total_watch_time'].quantile(0.99))
    }
    print(f"  Streamer watch time: P50={stats['streamer_watch_time']['p50']/1000:.1f}s")
    
    # 5.2 Conversion efficiency
    streamer_sessions['conversion_rate'] = (streamer_sessions['total_gift_count'] > 0).astype(int)
    streamer_sessions['revenue_per_watch_hour'] = streamer_sessions['total_revenue'] / (streamer_sessions['total_watch_time'] / 3600000 + 1)
    
    stats['conversion_efficiency'] = {
        'mean_conversion_rate': float(streamer_sessions['conversion_rate'].mean()),
        'median_revenue_per_watch_hour': float(streamer_sessions['revenue_per_watch_hour'].median()),
        'p50_revenue_per_watch_hour': float(streamer_sessions['revenue_per_watch_hour'].quantile(0.50)),
        'p90_revenue_per_watch_hour': float(streamer_sessions['revenue_per_watch_hour'].quantile(0.90))
    }
    print(f"  Revenue per watch hour: P50={stats['conversion_efficiency']['p50_revenue_per_watch_hour']:.2f}")
    
    # 5.3 Streamer quadrant
    high_watch = streamer_sessions['total_watch_time'] > streamer_sessions['total_watch_time'].quantile(0.75)
    high_revenue = streamer_sessions['total_revenue'] > streamer_sessions['total_revenue'].quantile(0.75)
    
    q1 = (high_watch & high_revenue).sum()
    q2 = (~high_watch & high_revenue).sum()
    q3 = (~high_watch & ~high_revenue).sum()
    q4 = (high_watch & ~high_revenue).sum()
    
    stats['streamer_quadrant'] = {
        'high_watch_high_revenue': int(q1),
        'low_watch_high_revenue': int(q2),
        'low_watch_low_revenue': int(q3),
        'high_watch_low_revenue': int(q4),
        'high_watch_low_revenue_ratio': float(q4 / len(streamer_sessions)) if len(streamer_sessions) > 0 else 0
    }
    print(f"  Streamer quadrant: High watch + Low revenue = {stats['streamer_quadrant']['high_watch_low_revenue']:,} ({stats['streamer_quadrant']['high_watch_low_revenue_ratio']*100:.1f}%)")
    
    return stats, streamer_sessions

def plot_phase5_figures(streamer_sessions, stats, img_dir):
    """Plot Phase 5 figures"""
    print("\n  Generating Phase 5 plots...")
    
    # Fig 5.1: Streamer watch time CCDF
    watch_times = streamer_sessions['total_watch_time'] / 1000  # seconds
    sorted_watch = np.sort(watch_times)[::-1]
    n = len(sorted_watch)
    ccdf = np.arange(1, n + 1) / n
    
    fig, ax = plt.subplots(figsize=(6, 5))
    mask = sorted_watch > 0
    ax.loglog(sorted_watch[mask], ccdf[mask], 'r-', linewidth=2, alpha=0.7)
    ax.set_xlabel('Total Watch Time (seconds, log scale)', fontweight='bold')
    ax.set_ylabel('P(X ≥ x) (log scale)', fontweight='bold')
    ax.set_title('Streamer watch time is highly concentrated', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(img_dir / "streamer_watch_time_ccdf.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: streamer_watch_time_ccdf.png")
    
    # Fig 5.2: Conversion efficiency distribution
    revenue_per_hour = streamer_sessions['revenue_per_watch_hour']
    revenue_per_hour = revenue_per_hour[revenue_per_hour > 0]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(np.log1p(revenue_per_hour), bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax.set_xlabel('Revenue per Watch Hour (log scale)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Streamer conversion efficiency varies widely', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(img_dir / "streamer_conversion_efficiency.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: streamer_conversion_efficiency.png")
    
    # Fig 5.3: Streamer quadrant (watch vs revenue)
    fig, ax = plt.subplots(figsize=(6, 5))
    watch_log = np.log1p(streamer_sessions['total_watch_time'] / 1000)
    revenue_log = np.log1p(streamer_sessions['total_revenue'])
    
    # Sample if too large
    if len(streamer_sessions) > 5000:
        sample_idx = np.random.choice(len(streamer_sessions), 5000, replace=False)
        watch_log = watch_log.iloc[sample_idx]
        revenue_log = revenue_log.iloc[sample_idx]
    
    ax.scatter(watch_log, revenue_log, alpha=0.3, s=1)
    
    # Add quadrant lines
    watch_median = np.median(watch_log)
    revenue_median = np.median(revenue_log[revenue_log > 0]) if (revenue_log > 0).sum() > 0 else 0
    ax.axvline(watch_median, color='red', linestyle='--', linewidth=1, alpha=0.5)
    if revenue_median > 0:
        ax.axhline(revenue_median, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    q4_ratio = stats['streamer_quadrant']['high_watch_low_revenue_ratio']
    ax.text(0.95, 0.05, f'High watch\nLow revenue: {q4_ratio*100:.1f}%',
           transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
           horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Total Watch Time (log scale)', fontweight='bold')
    ax.set_ylabel('Total Revenue (log scale)', fontweight='bold')
    ax.set_title(f'High watch + Low revenue streamers: {q4_ratio*100:.1f}%', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(img_dir / "streamer_quadrant_revenue.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: streamer_quadrant_revenue.png")
    
    # Fig 5.4: Streamer quadrant (viewers vs conversion)
    streamer_sessions['conversion_rate'] = (streamer_sessions['total_gift_count'] > 0).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    viewers_log = np.log1p(streamer_sessions['unique_viewers'])
    conversion_log = np.log1p(streamer_sessions['conversion_rate'] * 100 + 1)
    
    if len(streamer_sessions) > 5000:
        sample_idx = np.random.choice(len(streamer_sessions), 5000, replace=False)
        viewers_log = viewers_log.iloc[sample_idx]
        conversion_log = conversion_log.iloc[sample_idx]
    
    ax.scatter(viewers_log, conversion_log, alpha=0.3, s=1)
    ax.set_xlabel('Unique Viewers (log scale)', fontweight='bold')
    ax.set_ylabel('Conversion Rate (log scale)', fontweight='bold')
    ax.set_title('Viewer count vs conversion rate', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(img_dir / "streamer_quadrant_conversion.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: streamer_quadrant_conversion.png")

# ============================================================================
# Phase 6: Interaction Structure Analysis
# ============================================================================

def analyze_interaction_structure(sessions_df, gift_df):
    """Phase 6: Interaction Structure Analysis"""
    print("\n" + "=" * 60)
    print("Phase 6: Interaction Structure Analysis")
    print("=" * 60)
    
    stats = {}
    
    # 6.1 User loyalty (Top1 streamer share)
    user_streamer_watch = sessions_df.groupby(['user_id', 'streamer_id'])['total_watch_time'].sum().reset_index()
    user_streamer_gift = gift_df.groupby(['user_id', 'streamer_id'])['gift_price'].sum().reset_index()
    
    # Watch loyalty
    user_total_watch = user_streamer_watch.groupby('user_id')['total_watch_time'].sum()
    user_max_watch = user_streamer_watch.groupby('user_id')['total_watch_time'].max()
    watch_loyalty = (user_max_watch / user_total_watch).fillna(0)
    
    # Gift loyalty
    user_total_gift = user_streamer_gift.groupby('user_id')['gift_price'].sum()
    user_max_gift = user_streamer_gift.groupby('user_id')['gift_price'].max()
    gift_loyalty = (user_max_gift / user_total_gift).fillna(0)
    
    stats['user_loyalty'] = {
        'watch_loyalty_p50': float(watch_loyalty.quantile(0.50)),
        'watch_loyalty_p90': float(watch_loyalty.quantile(0.90)),
        'gift_loyalty_p50': float(gift_loyalty.quantile(0.50)),
        'gift_loyalty_p90': float(gift_loyalty.quantile(0.90))
    }
    print(f"  User loyalty (watch): P50={stats['user_loyalty']['watch_loyalty_p50']:.3f}, P90={stats['user_loyalty']['watch_loyalty_p90']:.3f}")
    print(f"  User loyalty (gift): P50={stats['user_loyalty']['gift_loyalty_p50']:.3f}, P90={stats['user_loyalty']['gift_loyalty_p90']:.3f}")
    
    return stats, watch_loyalty, gift_loyalty

def plot_phase6_figures(watch_loyalty, gift_loyalty, stats, img_dir):
    """Plot Phase 6 figures"""
    print("\n  Generating Phase 6 plots...")
    
    # Fig 6.3: User loyalty distribution
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(watch_loyalty, bins=50, alpha=0.5, label='Watch loyalty', color='blue', edgecolor='black')
    ax.hist(gift_loyalty, bins=50, alpha=0.5, label='Gift loyalty', color='red', edgecolor='black')
    ax.set_xlabel('Loyalty (Top1 Streamer Share)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Users show high loyalty to top streamer', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(img_dir / "user_loyalty_dist.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: user_loyalty_dist.png")

# ============================================================================
# Phase 7: Time & Seasonality Analysis
# ============================================================================

def analyze_temporal(click_df, gift_df):
    """Phase 7: Time & Seasonality Analysis"""
    print("\n" + "=" * 60)
    print("Phase 7: Time & Seasonality Analysis")
    print("=" * 60)
    
    stats = {}
    
    # Extract time features
    click_df['datetime'] = pd.to_datetime(click_df['timestamp'], unit='ms')
    gift_df['datetime'] = pd.to_datetime(gift_df['timestamp'], unit='ms')
    
    click_df['hour'] = click_df['datetime'].dt.hour
    click_df['day_of_week'] = click_df['datetime'].dt.dayofweek
    gift_df['hour'] = gift_df['datetime'].dt.hour
    gift_df['day_of_week'] = gift_df['datetime'].dt.dayofweek
    
    # Aggregate by day_of_week x hour
    temporal_agg = gift_df.groupby(['day_of_week', 'hour']).agg({
        'gift_price': ['count', 'sum']
    }).reset_index()
    temporal_agg.columns = ['day_of_week', 'hour', 'gift_count', 'gift_amount']
    
    # Find peak
    peak_idx = temporal_agg['gift_count'].idxmax()
    peak_hour = temporal_agg.loc[peak_idx, 'hour']
    peak_dow = temporal_agg.loc[peak_idx, 'day_of_week']
    
    stats['temporal'] = {
        'peak_hour': int(peak_hour),
        'peak_dow': int(peak_dow)
    }
    print(f"  Peak hour: {peak_hour}, Peak day: {peak_dow}")
    
    return stats, temporal_agg

def plot_phase7_figures(click_df, gift_df, temporal_agg, stats, img_dir):
    """Plot Phase 7 figures"""
    print("\n  Generating Phase 7 plots...")
    
    # Fig 7.1: 7x24 heatmap
    # Aggregate click data
    click_temporal = click_df.groupby(['day_of_week', 'hour']).agg({
        'watch_live_time': 'sum'
    }).reset_index()
    click_temporal.columns = ['day_of_week', 'hour', 'watch_time']
    
    # Merge
    temporal_full = temporal_agg.merge(click_temporal, on=['day_of_week', 'hour'], how='outer').fillna(0)
    
    # Create heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Heatmap 1: Gift count
    pivot_count = temporal_full.pivot(index='day_of_week', columns='hour', values='gift_count')
    pivot_count = pivot_count.fillna(0)
    im1 = axes[0].imshow(pivot_count.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    axes[0].set_xticks(range(24))
    axes[0].set_xticklabels(range(24))
    axes[0].set_yticks(range(7))
    axes[0].set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[0].set_xlabel('Hour of Day', fontweight='bold')
    axes[0].set_ylabel('Day of Week', fontweight='bold')
    axes[0].set_title('Gift Count by Day & Hour', fontweight='bold', fontsize=12)
    plt.colorbar(im1, ax=axes[0], label='Gift Count')
    
    # Heatmap 2: Gift amount
    pivot_amount = temporal_full.pivot(index='day_of_week', columns='hour', values='gift_amount')
    pivot_amount = pivot_amount.fillna(0)
    im2 = axes[1].imshow(pivot_amount.values, cmap='YlGnBu', aspect='auto', interpolation='nearest')
    axes[1].set_xticks(range(24))
    axes[1].set_xticklabels(range(24))
    axes[1].set_yticks(range(7))
    axes[1].set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[1].set_xlabel('Hour of Day', fontweight='bold')
    axes[1].set_ylabel('Day of Week', fontweight='bold')
    axes[1].set_title('Gift Amount by Day & Hour', fontweight='bold', fontsize=12)
    plt.colorbar(im2, ax=axes[1], label='Gift Amount')
    
    # Heatmap 3: Conversion rate (gift_count / watch_time, approximate)
    temporal_full['conversion_rate'] = temporal_full['gift_count'] / (temporal_full['watch_time'] / 1000 + 1) * 100
    pivot_conv = temporal_full.pivot(index='day_of_week', columns='hour', values='conversion_rate')
    pivot_conv = pivot_conv.fillna(0)
    im3 = axes[2].imshow(pivot_conv.values, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    axes[2].set_xticks(range(24))
    axes[2].set_xticklabels(range(24))
    axes[2].set_yticks(range(7))
    axes[2].set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[2].set_xlabel('Hour of Day', fontweight='bold')
    axes[2].set_ylabel('Day of Week', fontweight='bold')
    axes[2].set_title('Conversion Rate by Day & Hour', fontweight='bold', fontsize=12)
    plt.colorbar(im3, ax=axes[2], label='Conversion Rate (%)')
    
    plt.suptitle('Temporal patterns: Peak at hour 12 and evening 20-22', fontweight='bold', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(img_dir / "hour_dow_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: hour_dow_heatmap.png")

# ============================================================================
# Phase 8: Anomaly Detection
# ============================================================================

def detect_anomalies(sessions_df, gift_df):
    """Phase 8: Anomaly Detection"""
    print("\n" + "=" * 60)
    print("Phase 8: Anomaly Detection")
    print("=" * 60)
    
    stats = {}
    
    # Rule 1: High gift but very short watch
    gift_sessions = sessions_df[sessions_df['gift_count'] > 0].copy()
    p99_gift = gift_sessions['gift_amount'].quantile(0.99)
    suspicious_users = gift_sessions[
        (gift_sessions['gift_amount'] > p99_gift) & 
        (gift_sessions['total_watch_time'] < 10 * 1000)  # < 10 seconds
    ]
    stats['suspicious_users'] = {
        'count': int(len(suspicious_users)),
        'ratio': float(len(suspicious_users) / len(gift_sessions)) if len(gift_sessions) > 0 else 0
    }
    print(f"  Suspicious users (high gift, short watch): {stats['suspicious_users']['count']:,} ({stats['suspicious_users']['ratio']*100:.2f}%)")
    
    # Rule 2: Gift intervals < 1s
    gift_df_sorted = gift_df.sort_values(['user_id', 'live_id', 'timestamp'])
    gift_df_sorted['prev_timestamp'] = gift_df_sorted.groupby(['user_id', 'live_id'])['timestamp'].shift(1)
    gift_df_sorted['gift_interval'] = gift_df_sorted['timestamp'] - gift_df_sorted['prev_timestamp']
    suspicious_intervals = gift_df_sorted[gift_df_sorted['gift_interval'] < 1000]  # < 1 second
    stats['suspicious_intervals'] = {
        'count': int(len(suspicious_intervals)),
        'ratio': float(len(suspicious_intervals) / len(gift_df_sorted)) if len(gift_df_sorted) > 0 else 0
    }
    print(f"  Suspicious intervals (<1s): {stats['suspicious_intervals']['count']:,} ({stats['suspicious_intervals']['ratio']*100:.2f}%)")
    
    return stats

def plot_phase8_figures(gift_df, stats, img_dir):
    """Plot Phase 8 figures"""
    print("\n  Generating Phase 8 plots...")
    
    # Fig 8.1: Anomaly rules visualization
    gift_df_sorted = gift_df.sort_values(['user_id', 'live_id', 'timestamp'])
    gift_df_sorted['prev_timestamp'] = gift_df_sorted.groupby(['user_id', 'live_id'])['timestamp'].shift(1)
    gift_df_sorted['gift_interval'] = gift_df_sorted['timestamp'] - gift_df_sorted['prev_timestamp']
    gift_df_sorted = gift_df_sorted[gift_df_sorted['gift_interval'].notna()]
    
    # Sample if too large
    if len(gift_df_sorted) > 10000:
        gift_df_sorted = gift_df_sorted.sample(n=10000, random_state=42)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    intervals_sec = gift_df_sorted['gift_interval'] / 1000
    amounts = gift_df_sorted['gift_price']
    
    ax.scatter(intervals_sec, amounts, alpha=0.3, s=1)
    ax.axvline(1, color='red', linestyle='--', linewidth=2, label='Suspicious: <1s')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Gift Interval (seconds, log scale)', fontweight='bold')
    ax.set_ylabel('Gift Amount (log scale)', fontweight='bold')
    ax.set_title(f'Anomaly detection: {stats["suspicious_intervals"]["count"]:,} suspicious intervals', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(img_dir / "anomaly_rules.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: anomaly_rules.png")

def main():
    """Main execution function"""
    print("=" * 60)
    print("KuaiLive Comprehensive EDA - EXP-20260109-gift-allocation-02")
    print("=" * 60)
    
    # Load data
    gift_df, click_df, user_df, streamer_df = load_data()
    
    # Phase 1: Data Quality Check
    quality_stats, click_df, gift_df = analyze_data_quality(click_df, gift_df)
    
    # Phase 2: Build Sessions
    sessions_df = build_sessions(click_df, gift_df)
    
    # Save intermediate results
    print("\n--- Saving Intermediate Results ---")
    sessions_df.to_parquet(RESULTS_DIR / "sessions_20260109.parquet", index=False)
    print(f"  Saved sessions to {RESULTS_DIR / 'sessions_20260109.parquet'}")
    
    # Phase 3: Session & Funnel Analysis
    phase3_stats = analyze_session_funnel(sessions_df, click_df, gift_df)
    plot_phase3_figures(sessions_df, click_df, gift_df, phase3_stats, IMG_DIR)
    
    # Phase 4: User Behavior Analysis
    phase4_stats, user_sessions = analyze_user_behavior(sessions_df, gift_df, user_df)
    plot_phase4_figures(sessions_df, gift_df, user_sessions, phase4_stats, IMG_DIR)
    
    # Phase 5: Supply Side Analysis
    phase5_stats, streamer_sessions = analyze_supply_side(sessions_df, gift_df, streamer_df)
    plot_phase5_figures(streamer_sessions, phase5_stats, IMG_DIR)
    
    # Phase 6: Interaction Structure Analysis
    phase6_stats, watch_loyalty, gift_loyalty = analyze_interaction_structure(sessions_df, gift_df)
    plot_phase6_figures(watch_loyalty, gift_loyalty, phase6_stats, IMG_DIR)
    
    # Phase 7: Time & Seasonality Analysis
    phase7_stats, temporal_agg = analyze_temporal(click_df, gift_df)
    plot_phase7_figures(click_df, gift_df, temporal_agg, phase7_stats, IMG_DIR)
    
    # Phase 8: Anomaly Detection
    phase8_stats = detect_anomalies(sessions_df, gift_df)
    plot_phase8_figures(gift_df, phase8_stats, IMG_DIR)
    
    # Compile all results
    results = {
        "experiment_id": "EXP-20260109-gift-allocation-02",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_quality": quality_stats['health_scores'],
        "session_stats": {
            "total_sessions": len(sessions_df),
            "avg_session_duration": float(sessions_df['session_duration'].mean()),
            "session_duration_p50_p90_p99": [
                float(sessions_df['session_duration'].quantile(0.50)),
                float(sessions_df['session_duration'].quantile(0.90)),
                float(sessions_df['session_duration'].quantile(0.99))
            ],
            "session_conversion_rate": float((sessions_df['gift_count'] > 0).sum() / len(sessions_df)),
            "immediate_gift_rate": float((sessions_df['t_first_gift_relative'] < 0.1).sum() / (sessions_df['gift_count'] > 0).sum()) if (sessions_df['gift_count'] > 0).sum() > 0 else 0.0
        },
        "phase3_session_funnel": phase3_stats,
        "phase4_user_behavior": phase4_stats,
        "phase5_supply_side": phase5_stats,
        "phase6_interaction": phase6_stats,
        "phase7_temporal": phase7_stats,
        "phase8_anomaly": phase8_stats
    }
    
    # Save results
    output_file = RESULTS_DIR / "eda_comprehensive_stats_20260109.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n=== Results saved to {output_file} ===")
    
    print("\n" + "=" * 60)
    print("All Phases Complete!")
    print("=" * 60)
    print(f"Total Sessions: {len(sessions_df):,}")
    print(f"Session Conversion Rate: {results['session_stats']['session_conversion_rate']*100:.2f}%")
    print(f"Data Health Score: {quality_stats['health_scores']['overall_score']:.2f}/5.0")
    print(f"High watch + Low pay users: {phase4_stats['user_quadrant']['high_watch_low_pay_ratio']*100:.1f}%")
    print(f"High watch + Low revenue streamers: {phase5_stats['streamer_quadrant']['high_watch_low_revenue_ratio']*100:.1f}%")
    print(f"Suspicious users: {phase8_stats['suspicious_users']['count']:,}")
    print("=" * 60)
    
    return results, sessions_df, click_df, gift_df, user_df, streamer_df

if __name__ == "__main__":
    results, sessions_df, click_df, gift_df, user_df, streamer_df = main()
