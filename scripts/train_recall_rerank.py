#!/usr/bin/env python3
"""
Recall-Rerank Pipeline: Direct + Two-Stage Stage2
==================================================

Experiment: EXP-20260109-gift-allocation-51 (MVP-5.1)

Goal: Verify if Direct recall + Stage2 rerank can maintain Top-1% while improving NDCG.

Hypotheses:
- Direct Regression excels at Top-1% (54.5%) but NDCG@100 is weak (21.7%)
- Stage2 has Spearman=0.89 on gift subset, good for head ranking
- Combining them may get best of both worlds

Pipeline:
1. Direct model scores all samples → take Top-M candidates
2. Stage2 model re-ranks Top-M candidates
3. Evaluate final ranking vs ground truth

Sweep: Top_M = [50, 100, 200, 500, 1000]

Author: Viska Wei
Date: 2026-01-09
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

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# Paths
BASE_DIR = Path("/home/swei20/GiftLive")
DATA_DIR = BASE_DIR / "data" / "KuaiLive"
OUTPUT_DIR = BASE_DIR / "gift_allocation"
IMG_DIR = OUTPUT_DIR / "img"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

for d in [IMG_DIR, RESULTS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def log_message(msg: str, level: str = "INFO"):
    """Print log message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "📝", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "")
    print(f"[{timestamp}] {prefix} {msg}")


# ============ Feature Engineering (reused from train_fair_comparison.py) ============

def create_user_features(gift, click, user):
    user_gift_stats = gift.groupby('user_id').agg({
        'gift_price': ['count', 'sum', 'mean', 'std', 'max'],
        'streamer_id': 'nunique',
        'live_id': 'nunique'
    }).reset_index()
    user_gift_stats.columns = [
        'user_id', 'user_gift_count', 'user_gift_sum', 'user_gift_mean', 
        'user_gift_std', 'user_gift_max', 'user_unique_streamers', 'user_unique_rooms'
    ]
    user_gift_stats['user_gift_std'] = user_gift_stats['user_gift_std'].fillna(0)
    
    user_click_stats = click.groupby('user_id').agg({
        'watch_live_time': ['count', 'sum', 'mean'],
        'streamer_id': 'nunique',
        'live_id': 'nunique'
    }).reset_index()
    user_click_stats.columns = [
        'user_id', 'user_watch_count', 'user_watch_time_sum', 'user_watch_time_mean',
        'user_watch_unique_streamers', 'user_watch_unique_rooms'
    ]
    
    user_features = user[['user_id', 'age', 'gender', 'device_brand', 'device_price',
                           'fans_num', 'follow_num', 'accu_watch_live_cnt', 
                           'accu_watch_live_duration', 'is_live_streamer', 'is_photo_author',
                           'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3',
                           'onehot_feat4', 'onehot_feat5', 'onehot_feat6']].copy()
    
    user_features = user_features.merge(user_gift_stats, on='user_id', how='left')
    user_features = user_features.merge(user_click_stats, on='user_id', how='left')
    
    numeric_cols = user_features.select_dtypes(include=[np.number]).columns
    user_features[numeric_cols] = user_features[numeric_cols].fillna(0)
    return user_features


def create_streamer_features(gift, room, streamer):
    streamer_gift_stats = gift.groupby('streamer_id').agg({
        'gift_price': ['count', 'sum', 'mean', 'std', 'max'],
        'user_id': 'nunique',
        'live_id': 'nunique'
    }).reset_index()
    streamer_gift_stats.columns = [
        'streamer_id', 'streamer_gift_count', 'streamer_gift_sum', 'streamer_gift_mean',
        'streamer_gift_std', 'streamer_gift_max', 'streamer_unique_givers', 'streamer_unique_rooms'
    ]
    streamer_gift_stats['streamer_gift_std'] = streamer_gift_stats['streamer_gift_std'].fillna(0)
    
    streamer_room_stats = room.groupby('streamer_id').agg({
        'live_id': 'count',
        'live_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0,
    }).reset_index()
    streamer_room_stats.columns = ['streamer_id', 'streamer_room_count', 'streamer_main_live_type']
    
    streamer_features = streamer[['streamer_id', 'gender', 'age', 'device_brand', 'device_price',
                                   'live_operation_tag', 'fans_user_num', 'fans_group_fans_num',
                                   'follow_user_num', 'accu_live_cnt', 'accu_live_duration',
                                   'accu_play_cnt', 'accu_play_duration',
                                   'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3',
                                   'onehot_feat4', 'onehot_feat5', 'onehot_feat6']].copy()
    
    streamer_features = streamer_features.rename(columns={
        'gender': 'streamer_gender', 'age': 'streamer_age',
        'device_brand': 'streamer_device_brand', 'device_price': 'streamer_device_price',
        'onehot_feat0': 'streamer_onehot_feat0', 'onehot_feat1': 'streamer_onehot_feat1',
        'onehot_feat2': 'streamer_onehot_feat2', 'onehot_feat3': 'streamer_onehot_feat3',
        'onehot_feat4': 'streamer_onehot_feat4', 'onehot_feat5': 'streamer_onehot_feat5',
        'onehot_feat6': 'streamer_onehot_feat6',
    })
    
    streamer_features = streamer_features.merge(streamer_gift_stats, on='streamer_id', how='left')
    streamer_features = streamer_features.merge(streamer_room_stats, on='streamer_id', how='left')
    
    numeric_cols = streamer_features.select_dtypes(include=[np.number]).columns
    streamer_features[numeric_cols] = streamer_features[numeric_cols].fillna(0)
    return streamer_features


def create_interaction_features(gift, click):
    user_streamer_click = click.groupby(['user_id', 'streamer_id']).agg({
        'watch_live_time': ['count', 'sum', 'mean']
    }).reset_index()
    user_streamer_click.columns = ['user_id', 'streamer_id', 'pair_watch_count', 'pair_watch_time_sum', 'pair_watch_time_mean']
    
    user_streamer_gift = gift.groupby(['user_id', 'streamer_id']).agg({
        'gift_price': ['count', 'sum', 'mean']
    }).reset_index()
    user_streamer_gift.columns = ['user_id', 'streamer_id', 'pair_gift_count', 'pair_gift_sum', 'pair_gift_mean']
    
    interaction = user_streamer_click.merge(user_streamer_gift, on=['user_id', 'streamer_id'], how='outer')
    interaction = interaction.fillna(0)
    return interaction


def encode_categorical(df, cat_columns):
    df = df.copy()
    for col in cat_columns:
        if col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                df[col] = df[col].fillna('unknown').astype(str)
                df[col] = df[col].astype('category').cat.codes
    # Encode any remaining object columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('unknown').astype(str).astype('category').cat.codes
    return df


def prepare_features(gift, user, streamer, room, click):
    """Prepare all features for model evaluation."""
    log_message("Preparing features...")
    
    user_features = create_user_features(gift, click, user)
    streamer_features = create_streamer_features(gift, room, streamer)
    interaction = create_interaction_features(gift, click)
    
    click_base = click.copy()
    click_base['timestamp_dt'] = pd.to_datetime(click_base['timestamp'], unit='ms')
    click_base['hour'] = click_base['timestamp_dt'].dt.hour
    click_base['day_of_week'] = click_base['timestamp_dt'].dt.dayofweek
    click_base['is_weekend'] = (click_base['day_of_week'] >= 5).astype(int)
    click_base['date'] = click_base['timestamp_dt'].dt.date
    
    room_info = room[['live_id', 'live_type', 'live_content_category']].drop_duplicates('live_id')
    click_base = click_base.merge(room_info, on='live_id', how='left')
    
    gift_agg = gift.groupby(['user_id', 'streamer_id', 'live_id']).agg({
        'gift_price': 'sum'
    }).reset_index().rename(columns={'gift_price': 'total_gift_price'})
    
    click_base = click_base.merge(gift_agg, on=['user_id', 'streamer_id', 'live_id'], how='left')
    click_base['total_gift_price'] = click_base['total_gift_price'].fillna(0)
    click_base['gift_price'] = click_base['total_gift_price']
    
    df = click_base.merge(user_features, on='user_id', how='left')
    df = df.merge(streamer_features, on='streamer_id', how='left')
    df = df.merge(interaction, on=['user_id', 'streamer_id'], how='left')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    cat_columns = ['age', 'gender', 'device_brand', 'device_price', 'fans_num', 'follow_num',
                   'accu_watch_live_cnt', 'accu_watch_live_duration',
                   'streamer_gender', 'streamer_age', 'streamer_device_brand', 'streamer_device_price',
                   'live_operation_tag', 'fans_user_num', 'fans_group_fans_num', 'follow_user_num',
                   'accu_live_cnt', 'accu_live_duration', 'accu_play_cnt', 'accu_play_duration',
                   'live_type', 'live_content_category']
    df = encode_categorical(df, cat_columns)
    
    df['target'] = np.log1p(df['gift_price'])
    df['is_gift'] = (df['gift_price'] > 0).astype(int)
    
    log_message(f"Final dataset: {len(df):,} records")
    log_message(f"Gift rate: {df['is_gift'].mean()*100:.2f}%")
    
    return df


def get_feature_columns(df):
    exclude_cols = ['user_id', 'live_id', 'streamer_id', 'timestamp', 'timestamp_dt',
                    'date', 'gift_price', 'total_gift_price', 'target', 'is_gift', 'watch_live_time']
    return [c for c in df.columns if c not in exclude_cols]


def temporal_split(df):
    """Split data by time: 80% train, 10% val, 10% test."""
    log_message("Performing temporal split...")
    df = df.sort_values('timestamp').reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    log_message(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test


# ============ Evaluation Metrics ============

def compute_top_k_capture(y_true, y_pred, k_pct=0.01):
    """Compute Top-K% capture rate."""
    n = len(y_true)
    k = max(1, int(n * k_pct))
    
    true_rank = np.argsort(np.argsort(-y_true))
    pred_rank = np.argsort(np.argsort(-y_pred))
    
    true_topk = set(np.where(true_rank < k)[0])
    pred_topk = set(np.where(pred_rank < k)[0])
    
    return len(true_topk & pred_topk) / len(true_topk) if len(true_topk) > 0 else 0


def compute_ndcg(y_true, y_pred, k=100):
    """Compute NDCG@K."""
    pred_order = np.argsort(-y_pred)[:k]
    
    dcg = 0.0
    for i, idx in enumerate(pred_order):
        rel = y_true[idx]
        dcg += rel / np.log2(i + 2)
    
    idcg = 0.0
    sorted_true = np.sort(y_true)[::-1][:k]
    for i, rel in enumerate(sorted_true):
        idcg += rel / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0


def compute_recall_coverage(y_true, direct_pred, top_m, k_pct=0.01):
    """
    Compute recall coverage: what fraction of true Top-K% samples are 
    captured in the Direct's Top-M candidates.
    """
    n = len(y_true)
    k = max(1, int(n * k_pct))
    
    # True Top-K% indices
    true_rank = np.argsort(np.argsort(-y_true))
    true_topk = set(np.where(true_rank < k)[0])
    
    # Direct's Top-M indices
    direct_rank = np.argsort(np.argsort(-direct_pred))
    direct_topm = set(np.where(direct_rank < top_m)[0])
    
    # Coverage = intersection / true_topk
    return len(true_topk & direct_topm) / len(true_topk) if len(true_topk) > 0 else 0


def compute_spearman(y_true, y_pred):
    """Compute Spearman correlation."""
    corr, _ = stats.spearmanr(y_true, y_pred)
    return corr


# ============ Recall-Rerank Pipeline ============

def recall_rerank_evaluate(direct_model, stage2_model, test, feature_cols, top_m_list):
    """
    Evaluate Recall-Rerank pipeline with different Top-M values.
    
    Pipeline:
    1. Direct model scores all samples → take Top-M candidates
    2. Stage2 model re-ranks Top-M candidates  
    3. Evaluate final ranking vs ground truth
    """
    log_message("=" * 60)
    log_message("RECALL-RERANK EVALUATION")
    log_message("=" * 60)
    
    X_test = test[feature_cols]
    y_true_raw = test['gift_price'].values
    n_test = len(y_true_raw)
    
    # Get Direct predictions
    direct_pred_log = direct_model.predict(X_test)
    direct_pred_raw = np.expm1(np.maximum(direct_pred_log, 0))
    
    # Get Stage2 predictions (will be used for reranking)
    stage2_pred_log = stage2_model.predict(X_test)
    stage2_pred_raw = np.expm1(np.maximum(stage2_pred_log, 0))
    
    results = {
        'top_m_list': top_m_list,
        'metrics': []
    }
    
    log_message("-" * 40)
    log_message("Baseline: Direct-only (full ranking)")
    direct_top1 = compute_top_k_capture(y_true_raw, direct_pred_raw, 0.01)
    direct_top5 = compute_top_k_capture(y_true_raw, direct_pred_raw, 0.05)
    direct_ndcg = compute_ndcg(y_true_raw, direct_pred_raw, 100)
    direct_spearman = compute_spearman(y_true_raw, direct_pred_raw)
    
    log_message(f"  Top-1%: {direct_top1*100:.2f}%")
    log_message(f"  Top-5%: {direct_top5*100:.2f}%")
    log_message(f"  NDCG@100: {direct_ndcg:.4f}")
    log_message(f"  Spearman: {direct_spearman:.4f}")
    
    results['baseline_direct'] = {
        'top_1pct': float(direct_top1),
        'top_5pct': float(direct_top5),
        'ndcg_100': float(direct_ndcg),
        'spearman': float(direct_spearman)
    }
    
    # Two-Stage baseline (p×m, but here we use Stage2 alone for comparison)
    log_message("-" * 40)
    log_message("Baseline: Stage2-only (full ranking)")
    s2_top1 = compute_top_k_capture(y_true_raw, stage2_pred_raw, 0.01)
    s2_top5 = compute_top_k_capture(y_true_raw, stage2_pred_raw, 0.05)
    s2_ndcg = compute_ndcg(y_true_raw, stage2_pred_raw, 100)
    s2_spearman = compute_spearman(y_true_raw, stage2_pred_raw)
    
    log_message(f"  Top-1%: {s2_top1*100:.2f}%")
    log_message(f"  Top-5%: {s2_top5*100:.2f}%")
    log_message(f"  NDCG@100: {s2_ndcg:.4f}")
    log_message(f"  Spearman: {s2_spearman:.4f}")
    
    results['baseline_stage2'] = {
        'top_1pct': float(s2_top1),
        'top_5pct': float(s2_top5),
        'ndcg_100': float(s2_ndcg),
        'spearman': float(s2_spearman)
    }
    
    # Recall-Rerank with different Top-M values
    log_message("=" * 60)
    log_message("Recall-Rerank Pipeline")
    log_message("=" * 60)
    
    for top_m in top_m_list:
        log_message("-" * 40)
        log_message(f"Top-M = {top_m:,}")
        
        # Step 1: Recall using Direct model
        # Get Top-M indices based on Direct predictions
        direct_order = np.argsort(-direct_pred_raw)
        recall_indices = direct_order[:top_m]
        
        # Step 2: Rerank using Stage2 model
        # Get Stage2 predictions for recalled candidates
        stage2_scores = stage2_pred_raw[recall_indices]
        
        # Sort recalled candidates by Stage2 score
        rerank_order = np.argsort(-stage2_scores)
        final_indices = recall_indices[rerank_order]
        
        # Step 3: Create final prediction scores
        # Assign scores based on final ranking position
        final_pred = np.zeros(n_test)
        for rank, idx in enumerate(final_indices):
            final_pred[idx] = n_test - rank  # Higher score = better rank
        
        # For samples not in Top-M, use Direct scores (scaled down)
        non_recall_mask = np.ones(n_test, dtype=bool)
        non_recall_mask[recall_indices] = False
        final_pred[non_recall_mask] = direct_pred_raw[non_recall_mask] * 1e-6  # Much lower score
        
        # Compute metrics
        rr_top1 = compute_top_k_capture(y_true_raw, final_pred, 0.01)
        rr_top5 = compute_top_k_capture(y_true_raw, final_pred, 0.05)
        rr_ndcg = compute_ndcg(y_true_raw, final_pred, 100)
        rr_spearman = compute_spearman(y_true_raw, final_pred)
        
        # Recall coverage
        recall_coverage_1pct = compute_recall_coverage(y_true_raw, direct_pred_raw, top_m, 0.01)
        recall_coverage_5pct = compute_recall_coverage(y_true_raw, direct_pred_raw, top_m, 0.05)
        
        log_message(f"  Recall Coverage (Top-1%): {recall_coverage_1pct*100:.2f}%")
        log_message(f"  Recall Coverage (Top-5%): {recall_coverage_5pct*100:.2f}%")
        log_message(f"  Top-1% Capture: {rr_top1*100:.2f}%")
        log_message(f"  Top-5% Capture: {rr_top5*100:.2f}%")
        log_message(f"  NDCG@100: {rr_ndcg:.4f}")
        log_message(f"  Spearman: {rr_spearman:.4f}")
        
        results['metrics'].append({
            'top_m': top_m,
            'recall_coverage_1pct': float(recall_coverage_1pct),
            'recall_coverage_5pct': float(recall_coverage_5pct),
            'top_1pct': float(rr_top1),
            'top_5pct': float(rr_top5),
            'ndcg_100': float(rr_ndcg),
            'spearman': float(rr_spearman)
        })
    
    # Store predictions for plotting
    results['predictions'] = {
        'direct_pred': direct_pred_raw,
        'stage2_pred': stage2_pred_raw,
        'y_true': y_true_raw
    }
    
    return results


# ============ Plotting Functions ============

def plot_fig1_topk_vs_topm(results, save_path):
    """Fig1: Top-M vs Top-1% Capture (line plot)"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    top_m_list = [m['top_m'] for m in results['metrics']]
    top1_capture = [m['top_1pct'] * 100 for m in results['metrics']]
    
    ax.plot(top_m_list, top1_capture, 'o-', color='coral', lw=2, markersize=8, label='Recall-Rerank')
    
    # Baseline: Direct-only
    direct_top1 = results['baseline_direct']['top_1pct'] * 100
    ax.axhline(y=direct_top1, color='steelblue', linestyle='--', lw=2, label=f'Direct-only ({direct_top1:.1f}%)')
    
    # Target line
    ax.axhline(y=56, color='green', linestyle=':', lw=1.5, alpha=0.7, label='Target (56%)')
    
    ax.set_xlabel('Top-M (Recall Candidates)', fontsize=12)
    ax.set_ylabel('Top-1% Capture (%)', fontsize=12)
    ax.set_title('Recall-Rerank: Top-1% Capture vs Recall Size', fontsize=13)
    ax.legend(loc='lower right')
    ax.set_xscale('log')
    ax.set_xticks(top_m_list)
    ax.set_xticklabels([str(m) for m in top_m_list])
    ax.grid(True, alpha=0.3)
    
    # Annotate points
    for m, t in zip(top_m_list, top1_capture):
        ax.annotate(f'{t:.1f}%', xy=(m, t), textcoords="offset points", 
                   xytext=(0, 8), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}", "SUCCESS")


def plot_fig2_method_comparison(results, save_path):
    """Fig2: Method comparison (grouped bar)"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Find best Top-M (highest Top-1%)
    best_idx = np.argmax([m['top_1pct'] for m in results['metrics']])
    best_m = results['metrics'][best_idx]
    
    methods = ['Direct-only', 'Stage2-only', f'Recall-Rerank\n(M={best_m["top_m"]})']
    
    top1_values = [
        results['baseline_direct']['top_1pct'] * 100,
        results['baseline_stage2']['top_1pct'] * 100,
        best_m['top_1pct'] * 100
    ]
    top5_values = [
        results['baseline_direct']['top_5pct'] * 100,
        results['baseline_stage2']['top_5pct'] * 100,
        best_m['top_5pct'] * 100
    ]
    ndcg_values = [
        results['baseline_direct']['ndcg_100'] * 100,
        results['baseline_stage2']['ndcg_100'] * 100,
        best_m['ndcg_100'] * 100
    ]
    
    x = np.arange(len(methods))
    width = 0.25
    
    bars1 = ax.bar(x - width, top1_values, width, label='Top-1%', color='coral', alpha=0.85)
    bars2 = ax.bar(x, top5_values, width, label='Top-5%', color='steelblue', alpha=0.85)
    bars3 = ax.bar(x + width, ndcg_values, width, label='NDCG@100', color='green', alpha=0.85)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Method Comparison: Top-K% Capture & NDCG@100', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 2), textcoords="offset points", ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}", "SUCCESS")


def plot_fig3_tradeoff(results, save_path):
    """Fig3: Top-M vs Top-1% Capture / Recall Coverage (dual y-axis)"""
    fig, ax1 = plt.subplots(figsize=(6, 5))
    
    top_m_list = [m['top_m'] for m in results['metrics']]
    top1_capture = [m['top_1pct'] * 100 for m in results['metrics']]
    recall_coverage = [m['recall_coverage_1pct'] * 100 for m in results['metrics']]
    
    color1 = 'coral'
    ax1.set_xlabel('Top-M (Recall Candidates)', fontsize=12)
    ax1.set_ylabel('Top-1% Capture (%)', color=color1, fontsize=12)
    line1 = ax1.plot(top_m_list, top1_capture, 'o-', color=color1, lw=2, markersize=8, label='Top-1% Capture')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log')
    ax1.set_xticks(top_m_list)
    ax1.set_xticklabels([str(m) for m in top_m_list])
    
    ax2 = ax1.twinx()
    color2 = 'steelblue'
    ax2.set_ylabel('Recall Coverage (%)', color=color2, fontsize=12)
    line2 = ax2.plot(top_m_list, recall_coverage, 's--', color=color2, lw=2, markersize=8, label='Recall Coverage')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Target line for recall coverage (80%)
    ax2.axhline(y=80, color='gray', linestyle=':', lw=1.5, alpha=0.7)
    ax2.annotate('80%', xy=(top_m_list[-1], 80), xytext=(5, 0), textcoords="offset points",
                fontsize=9, color='gray')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right')
    
    ax1.set_title('Trade-off: Recall Size vs Performance', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}", "SUCCESS")


def plot_fig4_score_correlation(results, save_path, n_samples=5000):
    """Fig4: Direct Score Rank vs Stage2 Score Rank (scatter)"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    direct_pred = results['predictions']['direct_pred']
    stage2_pred = results['predictions']['stage2_pred']
    y_true = results['predictions']['y_true']
    
    n = len(direct_pred)
    
    # Sample for visualization
    if n > n_samples:
        np.random.seed(SEED)
        sample_idx = np.random.choice(n, n_samples, replace=False)
    else:
        sample_idx = np.arange(n)
    
    # Convert to rank (lower rank = higher score)
    direct_rank = np.argsort(np.argsort(-direct_pred))
    stage2_rank = np.argsort(np.argsort(-stage2_pred))
    
    # Color by ground truth (gift amount)
    y_sampled = y_true[sample_idx]
    colors = np.log1p(y_sampled)
    
    scatter = ax.scatter(direct_rank[sample_idx], stage2_rank[sample_idx], 
                         c=colors, cmap='YlOrRd', alpha=0.5, s=5)
    
    # Add perfect correlation line
    max_rank = max(direct_rank.max(), stage2_rank.max())
    ax.plot([0, max_rank], [0, max_rank], 'k--', lw=1, alpha=0.5, label='Perfect Correlation')
    
    ax.set_xlabel('Direct Model Rank', fontsize=12)
    ax.set_ylabel('Stage2 Model Rank', fontsize=12)
    ax.set_title('Score Correlation: Direct vs Stage2', fontsize=13)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log(1+gift)', fontsize=10)
    
    # Spearman correlation annotation
    spearman = compute_spearman(direct_pred, stage2_pred)
    ax.annotate(f'Spearman = {spearman:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
               fontsize=11, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='lower right')
    ax.set_xlim(0, max_rank * 1.05)
    ax.set_ylim(0, max_rank * 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}", "SUCCESS")


# ============ Main ============

def main():
    log_message("=" * 70)
    log_message("MVP-5.1: Recall-Rerank Pipeline Experiment")
    log_message("=" * 70)
    
    start_time = time.time()
    
    # Load data
    log_message("Loading data...")
    gift = pd.read_csv(DATA_DIR / "gift.csv")
    user = pd.read_csv(DATA_DIR / "user.csv")
    streamer = pd.read_csv(DATA_DIR / "streamer.csv")
    room = pd.read_csv(DATA_DIR / "room.csv")
    click = pd.read_csv(DATA_DIR / "click.csv")
    
    log_message(f"Gift: {len(gift):,} | User: {len(user):,} | Streamer: {len(streamer):,}")
    log_message(f"Room: {len(room):,} | Click: {len(click):,}")
    
    # Prepare features
    df = prepare_features(gift, user, streamer, room, click)
    train, val, test = temporal_split(df)
    feature_cols = get_feature_columns(df)
    log_message(f"Features: {len(feature_cols)}")
    
    # Load pre-trained models
    log_message("Loading pre-trained models...")
    
    direct_model_path = MODELS_DIR / "fair_direct_reg_20260108.pkl"
    stage2_model_path = MODELS_DIR / "fair_two_stage_reg_20260108.pkl"
    
    if not direct_model_path.exists() or not stage2_model_path.exists():
        log_message("Pre-trained models not found! Please run train_fair_comparison.py first.", "ERROR")
        return None
    
    with open(direct_model_path, 'rb') as f:
        direct_model = pickle.load(f)
    with open(stage2_model_path, 'rb') as f:
        stage2_model = pickle.load(f)
    
    log_message("Models loaded successfully", "SUCCESS")
    
    # Define Top-M sweep values
    top_m_list = [50, 100, 200, 500, 1000]
    
    # Run Recall-Rerank evaluation
    results = recall_rerank_evaluate(direct_model, stage2_model, test, feature_cols, top_m_list)
    
    # Generate figures
    log_message("=" * 60)
    log_message("Generating Figures")
    log_message("=" * 60)
    
    plot_fig1_topk_vs_topm(results, IMG_DIR / "mvp51_recall_vs_topk.png")
    plot_fig2_method_comparison(results, IMG_DIR / "mvp51_method_comparison.png")
    plot_fig3_tradeoff(results, IMG_DIR / "mvp51_tradeoff.png")
    plot_fig4_score_correlation(results, IMG_DIR / "mvp51_score_correlation.png")
    
    # Prepare results for JSON (remove numpy arrays)
    json_results = {
        'experiment_id': 'EXP-20260109-gift-allocation-51',
        'mvp': 'MVP-5.1',
        'timestamp': datetime.now().isoformat(),
        'test_size': len(test),
        'n_gift': int(test['is_gift'].sum()),
        'top_m_list': top_m_list,
        'baseline_direct': results['baseline_direct'],
        'baseline_stage2': results['baseline_stage2'],
        'recall_rerank_metrics': results['metrics']
    }
    
    # Find best configuration
    best_idx = np.argmax([m['top_1pct'] for m in results['metrics']])
    best_config = results['metrics'][best_idx]
    
    json_results['best_config'] = {
        'top_m': best_config['top_m'],
        'top_1pct': best_config['top_1pct'],
        'top_5pct': best_config['top_5pct'],
        'ndcg_100': best_config['ndcg_100'],
        'recall_coverage_1pct': best_config['recall_coverage_1pct']
    }
    
    # Gate-5A evaluation
    direct_top1 = results['baseline_direct']['top_1pct']
    direct_ndcg = results['baseline_direct']['ndcg_100']
    best_top1 = best_config['top_1pct']
    best_ndcg = best_config['ndcg_100']
    
    gate_pass = best_top1 >= 0.56 and best_ndcg > direct_ndcg
    
    json_results['gate_5a'] = {
        'condition_1': f'Top-1% >= 56%: {best_top1*100:.1f}% ({"PASS" if best_top1 >= 0.56 else "FAIL"})',
        'condition_2': f'NDCG@100 > baseline: {best_ndcg:.4f} > {direct_ndcg:.4f} ({"PASS" if best_ndcg > direct_ndcg else "FAIL"})',
        'overall': 'PASS' if gate_pass else 'FAIL'
    }
    
    # Save results
    results_path = RESULTS_DIR / "recall_rerank_20260109.json"
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    log_message(f"Results saved: {results_path}", "SUCCESS")
    
    # Print Summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("📊 EXPERIMENT SUMMARY - MVP-5.1 Recall-Rerank")
    print("=" * 70)
    print(f"Test samples: {len(test):,} ({json_results['n_gift']:,} gifts)")
    print(f"Total time: {total_time:.1f}s")
    print("-" * 70)
    print("BASELINE METRICS:")
    print(f"  Direct-only:   Top-1%={direct_top1*100:.1f}%  NDCG@100={direct_ndcg:.4f}")
    print(f"  Stage2-only:   Top-1%={results['baseline_stage2']['top_1pct']*100:.1f}%  NDCG@100={results['baseline_stage2']['ndcg_100']:.4f}")
    print("-" * 70)
    print("RECALL-RERANK RESULTS:")
    for m in results['metrics']:
        print(f"  Top-M={m['top_m']:4d}: Top-1%={m['top_1pct']*100:5.1f}%  "
              f"NDCG={m['ndcg_100']:.4f}  Coverage={m['recall_coverage_1pct']*100:.1f}%")
    print("-" * 70)
    print("BEST CONFIGURATION:")
    print(f"  Top-M: {best_config['top_m']}")
    print(f"  Top-1% Capture: {best_config['top_1pct']*100:.1f}%")
    print(f"  Top-5% Capture: {best_config['top_5pct']*100:.1f}%")
    print(f"  NDCG@100: {best_config['ndcg_100']:.4f}")
    print(f"  Recall Coverage (Top-1%): {best_config['recall_coverage_1pct']*100:.1f}%")
    print("-" * 70)
    print("🎯 GATE-5A EVALUATION:")
    print(f"  {json_results['gate_5a']['condition_1']}")
    print(f"  {json_results['gate_5a']['condition_2']}")
    print(f"  Overall: {json_results['gate_5a']['overall']}")
    print("=" * 70)
    
    log_message("Experiment completed!", "SUCCESS")
    
    return json_results


if __name__ == "__main__":
    results = main()
