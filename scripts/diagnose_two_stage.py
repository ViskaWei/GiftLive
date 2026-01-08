#!/usr/bin/env python3
"""
Two-Stage Diagnostic Decomposition
===================================

Experiment: EXP-20260108-gift-allocation-11 (MVP-1.4)

Goal: Diagnose why Two-Stage (35.7%) underperforms Direct Regression (54.5%)
by decomposing the error into Stage1, Stage2, and combination components.

Hypotheses:
- H1: Stage2 data insufficient (34k vs 1.87M)
- H2: p×m multiplication amplifies errors
- H3: Stage2 OOD prediction problem
- H4: Potential feature/methodology issues

Experiments:
1. Stage1-only ranking: Use p(x) alone for Top-K
2. Stage2 gift subset: Evaluate m(x) on Y>0 samples only
3. Oracle p: Use true 1(Y>0) × m(x) for upper bound
4. Oracle m: Use p(x) × true log(1+Y) for upper bound

Author: Viska Wei
Date: 2026-01-08
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

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150


def log_message(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "📝", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "")
    print(f"[{timestamp}] {prefix} {msg}")


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


def load_models_and_data():
    """Load trained models and prepare test data."""
    log_message("Loading models and data...")
    
    # Load models
    with open(MODELS_DIR / "fair_direct_reg_20260108.pkl", 'rb') as f:
        direct_model = pickle.load(f)
    with open(MODELS_DIR / "fair_two_stage_clf_20260108.pkl", 'rb') as f:
        clf_model = pickle.load(f)
    with open(MODELS_DIR / "fair_two_stage_reg_20260108.pkl", 'rb') as f:
        reg_model = pickle.load(f)
    
    log_message("Models loaded successfully", "SUCCESS")
    
    # Load data (rerun feature prep from fair_comparison)
    gift = pd.read_csv(DATA_DIR / "gift.csv")
    user = pd.read_csv(DATA_DIR / "user.csv")
    streamer = pd.read_csv(DATA_DIR / "streamer.csv")
    room = pd.read_csv(DATA_DIR / "room.csv")
    click = pd.read_csv(DATA_DIR / "click.csv")
    
    log_message(f"Click records: {len(click):,}")
    
    return direct_model, clf_model, reg_model, gift, user, streamer, room, click


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
    user_streamer_click.columns = [
        'user_id', 'streamer_id', 
        'pair_watch_count', 'pair_watch_time_sum', 'pair_watch_time_mean'
    ]
    
    user_streamer_gift = gift.groupby(['user_id', 'streamer_id']).agg({
        'gift_price': ['count', 'sum', 'mean']
    }).reset_index()
    user_streamer_gift.columns = [
        'user_id', 'streamer_id',
        'pair_gift_count', 'pair_gift_sum', 'pair_gift_mean'
    ]
    
    interaction = user_streamer_click.merge(user_streamer_gift, on=['user_id', 'streamer_id'], how='outer')
    interaction = interaction.fillna(0)
    
    return interaction


def encode_categorical(df, cat_columns):
    df = df.copy()
    for col in cat_columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].fillna('unknown')
            df[col] = df[col].astype('category').cat.codes
    return df


def prepare_test_data(gift, user, streamer, room, click):
    """Prepare test data with same feature engineering as training."""
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
    
    cat_columns = [
        'age', 'gender', 'device_brand', 'device_price', 'fans_num', 'follow_num',
        'accu_watch_live_cnt', 'accu_watch_live_duration',
        'streamer_gender', 'streamer_age', 'streamer_device_brand', 'streamer_device_price',
        'live_operation_tag', 'fans_user_num', 'fans_group_fans_num', 'follow_user_num',
        'accu_live_cnt', 'accu_live_duration', 'accu_play_cnt', 'accu_play_duration',
        'live_type', 'live_content_category'
    ]
    df = encode_categorical(df, cat_columns)
    
    df['target'] = np.log1p(df['gift_price'])
    df['is_gift'] = (df['gift_price'] > 0).astype(int)
    
    # Temporal split - get test set
    df = df.sort_values('timestamp').reset_index(drop=True)
    max_date = df['date'].max()
    test_start = max_date - pd.Timedelta(days=6)
    test = df[df['date'] >= test_start].copy()
    
    log_message(f"Test set: {len(test):,} samples, {test['is_gift'].sum():,} gifts ({test['is_gift'].mean()*100:.2f}%)")
    
    return test


def get_feature_columns(df):
    exclude_cols = [
        'user_id', 'live_id', 'streamer_id', 'timestamp', 'timestamp_dt',
        'date', 'gift_price', 'total_gift_price', 'target', 'is_gift',
        'watch_live_time'
    ]
    return [c for c in df.columns if c not in exclude_cols]


def run_diagnosis(direct_model, clf_model, reg_model, test, feature_cols):
    """Run diagnostic decomposition experiments."""
    log_message("=" * 60)
    log_message("DIAGNOSTIC DECOMPOSITION")
    log_message("=" * 60)
    
    X_test = test[feature_cols]
    y_true_raw = test['gift_price'].values
    y_true_log = test['target'].values
    y_true_binary = test['is_gift'].values
    
    # Get predictions
    direct_pred_log = direct_model.predict(X_test)
    direct_pred_raw = np.expm1(np.maximum(direct_pred_log, 0))
    
    p_x = clf_model.predict(X_test)  # Stage1: P(gift)
    m_x_log = reg_model.predict(X_test)  # Stage2: E[log(1+Y)|Y>0]
    m_x = np.expm1(np.maximum(m_x_log, 0))
    
    v_x = p_x * m_x  # Two-Stage combined
    
    results = {}
    
    # Reference: Direct Regression
    log_message("-" * 40)
    log_message("Reference: Direct Regression")
    direct_top1 = compute_top_k_capture(y_true_raw, direct_pred_raw, 0.01)
    direct_top5 = compute_top_k_capture(y_true_raw, direct_pred_raw, 0.05)
    direct_ndcg = compute_ndcg(y_true_raw, direct_pred_raw, 100)
    direct_spearman, _ = stats.spearmanr(y_true_raw, direct_pred_raw)
    
    log_message(f"  Top-1%: {direct_top1*100:.1f}%")
    log_message(f"  Top-5%: {direct_top5*100:.1f}%")
    log_message(f"  NDCG@100: {direct_ndcg:.4f}")
    log_message(f"  Spearman: {direct_spearman:.4f}")
    
    results['direct_reg'] = {
        'top_1pct': float(direct_top1),
        'top_5pct': float(direct_top5),
        'ndcg_100': float(direct_ndcg),
        'spearman': float(direct_spearman)
    }
    
    # Reference: Two-Stage (p×m)
    log_message("-" * 40)
    log_message("Reference: Two-Stage (p×m)")
    ts_top1 = compute_top_k_capture(y_true_raw, v_x, 0.01)
    ts_top5 = compute_top_k_capture(y_true_raw, v_x, 0.05)
    ts_ndcg = compute_ndcg(y_true_raw, v_x, 100)
    ts_spearman, _ = stats.spearmanr(y_true_raw, v_x)
    
    log_message(f"  Top-1%: {ts_top1*100:.1f}%")
    log_message(f"  Top-5%: {ts_top5*100:.1f}%")
    log_message(f"  NDCG@100: {ts_ndcg:.4f}")
    log_message(f"  Spearman: {ts_spearman:.4f}")
    
    results['two_stage'] = {
        'top_1pct': float(ts_top1),
        'top_5pct': float(ts_top5),
        'ndcg_100': float(ts_ndcg),
        'spearman': float(ts_spearman)
    }
    
    # Exp1: Stage1-only (use p(x) for ranking)
    log_message("-" * 40)
    log_message("Exp1: Stage1-only ranking (p(x))")
    s1_top1 = compute_top_k_capture(y_true_raw, p_x, 0.01)
    s1_top5 = compute_top_k_capture(y_true_raw, p_x, 0.05)
    s1_ndcg = compute_ndcg(y_true_raw, p_x, 100)
    s1_spearman, _ = stats.spearmanr(y_true_raw, p_x)
    
    log_message(f"  Top-1%: {s1_top1*100:.1f}%")
    log_message(f"  Top-5%: {s1_top5*100:.1f}%")
    log_message(f"  NDCG@100: {s1_ndcg:.4f}")
    log_message(f"  Spearman: {s1_spearman:.4f}")
    
    results['exp1_stage1_only'] = {
        'top_1pct': float(s1_top1),
        'top_5pct': float(s1_top5),
        'ndcg_100': float(s1_ndcg),
        'spearman': float(s1_spearman),
        'interpretation': 'Stage1 classifier ranking ability without m(x)'
    }
    
    # Exp2: Stage2 on gift subset
    log_message("-" * 40)
    log_message("Exp2: Stage2 on gift subset (Y>0)")
    gift_mask = y_true_raw > 0
    n_gift = gift_mask.sum()
    log_message(f"  Gift samples: {n_gift:,}")
    
    m_x_gift = m_x[gift_mask]
    y_gift = y_true_raw[gift_mask]
    direct_gift = direct_pred_raw[gift_mask]
    
    s2_spearman, _ = stats.spearmanr(y_gift, m_x_gift)
    direct_gift_spearman, _ = stats.spearmanr(y_gift, direct_gift)
    
    log_message(f"  Stage2 Spearman (gift): {s2_spearman:.4f}")
    log_message(f"  Direct Spearman (gift): {direct_gift_spearman:.4f}")
    log_message(f"  Stage2 better: {s2_spearman > direct_gift_spearman}")
    
    results['exp2_stage2_gift_subset'] = {
        'n_gift_samples': int(n_gift),
        'stage2_spearman': float(s2_spearman),
        'direct_spearman': float(direct_gift_spearman),
        'stage2_better': bool(s2_spearman > direct_gift_spearman),
        'interpretation': 'Stage2 ranking ability within gift samples'
    }
    
    # Exp3: Oracle p (use true 1(Y>0))
    log_message("-" * 40)
    log_message("Exp3: Oracle p (true 1(Y>0) × m(x))")
    oracle_p = y_true_binary.astype(float)
    oracle_p_score = oracle_p * m_x
    
    op_top1 = compute_top_k_capture(y_true_raw, oracle_p_score, 0.01)
    op_top5 = compute_top_k_capture(y_true_raw, oracle_p_score, 0.05)
    op_ndcg = compute_ndcg(y_true_raw, oracle_p_score, 100)
    
    log_message(f"  Top-1%: {op_top1*100:.1f}%")
    log_message(f"  Top-5%: {op_top5*100:.1f}%")
    log_message(f"  NDCG@100: {op_ndcg:.4f}")
    
    results['exp3_oracle_p'] = {
        'top_1pct': float(op_top1),
        'top_5pct': float(op_top5),
        'ndcg_100': float(op_ndcg),
        'description': 'Perfect classification × actual m(x)',
        'interpretation': 'Upper bound if Stage1 were perfect'
    }
    
    # Exp4: Oracle m (use true log(1+Y) for gift, keep m(x) for non-gift)
    log_message("-" * 40)
    log_message("Exp4: Oracle m (p(x) × true log(1+Y))")
    oracle_m = np.where(gift_mask, y_true_raw, m_x)
    oracle_m_score = p_x * oracle_m
    
    om_top1 = compute_top_k_capture(y_true_raw, oracle_m_score, 0.01)
    om_top5 = compute_top_k_capture(y_true_raw, oracle_m_score, 0.05)
    om_ndcg = compute_ndcg(y_true_raw, oracle_m_score, 100)
    
    log_message(f"  Top-1%: {om_top1*100:.1f}%")
    log_message(f"  Top-5%: {om_top5*100:.1f}%")
    log_message(f"  NDCG@100: {om_ndcg:.4f}")
    
    results['exp4_oracle_m'] = {
        'top_1pct': float(om_top1),
        'top_5pct': float(om_top5),
        'ndcg_100': float(om_ndcg),
        'description': 'Actual p(x) × perfect amount',
        'interpretation': 'Upper bound if Stage2 were perfect'
    }
    
    # Binary classification upper bound
    log_message("-" * 40)
    log_message("Reference: Binary classification upper bound")
    binary_top1 = compute_top_k_capture(y_true_raw, y_true_binary.astype(float), 0.01)
    log_message(f"  Top-1% (if predict binary only): {binary_top1*100:.1f}%")
    
    results['binary_upper_bound'] = {
        'top_1pct': float(binary_top1),
        'interpretation': 'Best possible with perfect binary classification'
    }
    
    return results


def diagnose_primary_cause(results):
    """Analyze results to determine primary cause of Two-Stage underperformance."""
    log_message("=" * 60)
    log_message("DIAGNOSIS ANALYSIS")
    log_message("=" * 60)
    
    direct_top1 = results['direct_reg']['top_1pct']
    ts_top1 = results['two_stage']['top_1pct']
    s1_top1 = results['exp1_stage1_only']['top_1pct']
    oracle_p_top1 = results['exp3_oracle_p']['top_1pct']
    oracle_m_top1 = results['exp4_oracle_m']['top_1pct']
    s2_better = results['exp2_stage2_gift_subset']['stage2_better']
    
    diagnosis = {
        'gap_ts_vs_direct': float(ts_top1 - direct_top1),
        'gap_s1_vs_direct': float(s1_top1 - direct_top1),
        'gain_oracle_p': float(oracle_p_top1 - ts_top1),
        'gain_oracle_m': float(oracle_m_top1 - ts_top1),
        's2_better_in_gift_subset': s2_better
    }
    
    # Determine primary cause
    if abs(s1_top1 - direct_top1) < 0.05:  # Stage1-only is close to Direct
        if oracle_p_top1 > oracle_m_top1:
            primary_cause = "stage2_data_insufficient"
            evidence = f"Stage1-only ({s1_top1*100:.1f}%) close to Direct ({direct_top1*100:.1f}%), Oracle_p gains more"
        else:
            primary_cause = "multiplication_noise"
            evidence = f"Stage1-only close to Direct, but multiplication hurts"
    elif oracle_p_top1 - ts_top1 > oracle_m_top1 - ts_top1:
        primary_cause = "stage1_classification"
        evidence = f"Oracle_p gain ({(oracle_p_top1-ts_top1)*100:.1f}pp) > Oracle_m gain ({(oracle_m_top1-ts_top1)*100:.1f}pp)"
    else:
        primary_cause = "stage2_data_insufficient"
        evidence = f"Oracle_m gain ({(oracle_m_top1-ts_top1)*100:.1f}pp) >= Oracle_p gain ({(oracle_p_top1-ts_top1)*100:.1f}pp)"
    
    if s2_better:
        recommendation = "Consider recall-rank pipeline: Direct for recall, Two-Stage for rerank"
    else:
        recommendation = "Close Two-Stage direction entirely"
    
    diagnosis['primary_cause'] = primary_cause
    diagnosis['evidence'] = evidence
    diagnosis['recommendation'] = recommendation
    
    log_message(f"Primary cause: {primary_cause}")
    log_message(f"Evidence: {evidence}")
    log_message(f"Recommendation: {recommendation}")
    
    return diagnosis


def plot_diagnosis_results(results):
    """Create visualization of diagnosis results."""
    log_message("Creating diagnosis plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Top-1% comparison
    ax = axes[0]
    methods = ['Direct', 'Two-Stage', 'Stage1-only', 'Oracle p', 'Oracle m']
    values = [
        results['direct_reg']['top_1pct'] * 100,
        results['two_stage']['top_1pct'] * 100,
        results['exp1_stage1_only']['top_1pct'] * 100,
        results['exp3_oracle_p']['top_1pct'] * 100,
        results['exp4_oracle_m']['top_1pct'] * 100
    ]
    colors = ['steelblue', 'coral', 'green', 'purple', 'orange']
    
    bars = ax.bar(methods, values, color=colors, alpha=0.8)
    ax.axhline(y=results['binary_upper_bound']['top_1pct'] * 100, color='red', 
               linestyle='--', label='Binary upper bound')
    ax.set_ylabel('Top-1% Capture (%)')
    ax.set_title('Top-1% Capture: Diagnostic Decomposition')
    ax.legend()
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Oracle gains
    ax = axes[1]
    ts_top1 = results['two_stage']['top_1pct'] * 100
    oracle_p_gain = (results['exp3_oracle_p']['top_1pct'] - results['two_stage']['top_1pct']) * 100
    oracle_m_gain = (results['exp4_oracle_m']['top_1pct'] - results['two_stage']['top_1pct']) * 100
    
    gains = ['Oracle p gain', 'Oracle m gain']
    gain_values = [oracle_p_gain, oracle_m_gain]
    
    bars = ax.bar(gains, gain_values, color=['purple', 'orange'], alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Gain over Two-Stage (pp)')
    ax.set_title('Oracle Decomposition: Potential Improvement')
    
    for bar, val in zip(bars, gain_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'+{val:.1f}pp', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Gift subset comparison
    ax = axes[2]
    methods = ['Stage2 (m(x))', 'Direct']
    spearman_vals = [
        results['exp2_stage2_gift_subset']['stage2_spearman'],
        results['exp2_stage2_gift_subset']['direct_spearman']
    ]
    
    bars = ax.bar(methods, spearman_vals, color=['coral', 'steelblue'], alpha=0.8)
    ax.set_ylabel('Spearman Correlation')
    ax.set_title(f"Gift Subset Ranking (n={results['exp2_stage2_gift_subset']['n_gift_samples']:,})")
    
    for bar, val in zip(bars, spearman_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(IMG_DIR / "two_stage_diagnosis.png", bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {IMG_DIR}/two_stage_diagnosis.png", "SUCCESS")


def main():
    log_message("=" * 60)
    log_message("MVP-1.4: Two-Stage Diagnostic Decomposition")
    log_message("=" * 60)
    
    start_time = time.time()
    
    # Load models and data
    direct_model, clf_model, reg_model, gift, user, streamer, room, click = load_models_and_data()
    
    # Prepare test data
    test = prepare_test_data(gift, user, streamer, room, click)
    feature_cols = get_feature_columns(test)
    log_message(f"Features: {len(feature_cols)}")
    
    # Run diagnostic experiments
    results = run_diagnosis(direct_model, clf_model, reg_model, test, feature_cols)
    
    # Analyze and diagnose
    diagnosis = diagnose_primary_cause(results)
    results['diagnosis'] = diagnosis
    
    # Create plots
    plot_diagnosis_results(results)
    
    # Save results
    results['experiment_id'] = 'EXP-20260108-gift-allocation-11'
    results['mvp'] = 'MVP-1.4'
    results['timestamp'] = datetime.now().isoformat()
    results['test_size'] = len(test)
    results['n_gift'] = int(test['is_gift'].sum())
    
    results_path = RESULTS_DIR / "two_stage_diagnosis_20260108.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved: {results_path}", "SUCCESS")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 DIAGNOSIS SUMMARY - MVP-1.4")
    print("=" * 60)
    print(f"Test samples: {len(test):,} ({results['n_gift']:,} gifts)")
    print("-" * 60)
    print("TOP-1% CAPTURE COMPARISON:")
    print(f"  Direct Regression:  {results['direct_reg']['top_1pct']*100:.1f}%")
    print(f"  Two-Stage (p×m):    {results['two_stage']['top_1pct']*100:.1f}%")
    print(f"  Stage1-only (p):    {results['exp1_stage1_only']['top_1pct']*100:.1f}%")
    print(f"  Oracle p (×m):      {results['exp3_oracle_p']['top_1pct']*100:.1f}%")
    print(f"  Oracle m (p×):      {results['exp4_oracle_m']['top_1pct']*100:.1f}%")
    print("-" * 60)
    print("GIFT SUBSET (Y>0) RANKING:")
    print(f"  Stage2 Spearman:    {results['exp2_stage2_gift_subset']['stage2_spearman']:.4f}")
    print(f"  Direct Spearman:    {results['exp2_stage2_gift_subset']['direct_spearman']:.4f}")
    print(f"  Stage2 better:      {results['exp2_stage2_gift_subset']['stage2_better']}")
    print("-" * 60)
    print("🔍 DIAGNOSIS:")
    print(f"  Primary cause:      {diagnosis['primary_cause']}")
    print(f"  Evidence:           {diagnosis['evidence']}")
    print(f"  Recommendation:     {diagnosis['recommendation']}")
    print("=" * 60)
    
    total_time = time.time() - start_time
    log_message(f"Total time: {total_time:.1f}s", "SUCCESS")
    
    return results


if __name__ == "__main__":
    results = main()
