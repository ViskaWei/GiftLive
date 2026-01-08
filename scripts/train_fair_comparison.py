#!/usr/bin/env python3
"""
Fair Comparison V2: 修复 Two-Stage 训练不充分的问题
==============================================================

修复:
1. Stage 1 使用 AUC 优化 (而非 logloss)
2. 增加 min_data_in_leaf 防止过早收敛
3. 设置 min_child_samples 确保充分训练

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
from sklearn.metrics import precision_recall_curve, auc, log_loss, roc_auc_score
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path("/home/swei20/GiftLive")
DATA_DIR = BASE_DIR / "data" / "KuaiLive"
OUTPUT_DIR = BASE_DIR / "gift_allocation"
IMG_DIR = OUTPUT_DIR / "img"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"

for d in [IMG_DIR, RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def log_message(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "📝", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "")
    print(f"[{timestamp}] {prefix} {msg}")


def load_data():
    log_message("Loading data files...")
    gift = pd.read_csv(DATA_DIR / "gift.csv")
    user = pd.read_csv(DATA_DIR / "user.csv")
    streamer = pd.read_csv(DATA_DIR / "streamer.csv")
    room = pd.read_csv(DATA_DIR / "room.csv")
    click = pd.read_csv(DATA_DIR / "click.csv")
    return gift, user, streamer, room, click


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
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].fillna('unknown')
            df[col] = df[col].astype('category').cat.codes
    return df


def prepare_features(gift, user, streamer, room, click):
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


def temporal_split(df):
    log_message("Performing temporal split...")
    df = df.sort_values('timestamp').reset_index(drop=True)
    min_date = df['date'].min()
    max_date = df['date'].max()
    test_start = max_date - pd.Timedelta(days=6)
    val_start = test_start - pd.Timedelta(days=7)
    
    train = df[df['date'] < val_start].copy()
    val = df[(df['date'] >= val_start) & (df['date'] < test_start)].copy()
    test = df[df['date'] >= test_start].copy()
    
    log_message(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test


def get_feature_columns(df):
    exclude_cols = ['user_id', 'live_id', 'streamer_id', 'timestamp', 'timestamp_dt',
                    'date', 'gift_price', 'total_gift_price', 'target', 'is_gift', 'watch_live_time']
    return [c for c in df.columns if c not in exclude_cols]


# ============ 改进版 Two-Stage ============

def train_stage1_classifier_v2(train, val, feature_cols):
    """改进版 Stage 1: 使用 AUC 优化 + 更多训练轮次"""
    log_message("=" * 50)
    log_message("Training Stage 1 V2: AUC-optimized Classification")
    log_message("=" * 50)
    
    X_train = train[feature_cols]
    y_train = train['is_gift']
    X_val = val[feature_cols]
    y_val = val['is_gift']
    
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = min(n_neg / n_pos, 50.0) if n_pos > 0 else 1.0
    
    log_message(f"Pos: {n_pos:,} ({n_pos/len(y_train)*100:.2f}%)")
    log_message(f"scale_pos_weight: {scale_pos_weight:.2f}")
    
    # 关键改进：使用 AUC 优化
    params = {
        'objective': 'binary',
        'metric': 'auc',  # 改用 AUC
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight,
        'min_data_in_leaf': 100,  # 增加防止过拟合
        'verbose': -1,
        'seed': SEED,
        'n_jobs': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]
    
    start_time = time.time()
    model = lgb.train(params, train_data, num_boost_round=500,
                      valid_sets=[train_data, val_data], valid_names=['train', 'val'],
                      callbacks=callbacks)
    training_time = time.time() - start_time
    
    log_message(f"Stage 1 V2 completed in {training_time:.1f}s, iter={model.best_iteration}", "SUCCESS")
    return model, training_time


def train_stage2_regressor_v2(train, val, feature_cols):
    """Stage 2: Regression on gift samples only"""
    log_message("=" * 50)
    log_message("Training Stage 2: Conditional Amount Regression")
    log_message("=" * 50)
    
    train_gift = train[train['is_gift'] == 1].copy()
    val_gift = val[val['is_gift'] == 1].copy()
    
    log_message(f"Training on {len(train_gift):,} gift samples")
    
    X_train = train_gift[feature_cols]
    y_train = train_gift['target']
    X_val = val_gift[feature_cols]
    y_val = val_gift['target']
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'seed': SEED,
        'n_jobs': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
    
    start_time = time.time()
    model = lgb.train(params, train_data, num_boost_round=500,
                      valid_sets=[train_data, val_data], valid_names=['train', 'val'],
                      callbacks=callbacks)
    training_time = time.time() - start_time
    
    log_message(f"Stage 2 completed in {training_time:.1f}s, iter={model.best_iteration}", "SUCCESS")
    return model, training_time


def train_direct_regression(train, val, feature_cols):
    """Direct Regression baseline"""
    log_message("=" * 50)
    log_message("Training Direct Regression on log(1+Y)")
    log_message("=" * 50)
    
    X_train = train[feature_cols]
    y_train = train['target']
    X_val = val[feature_cols]
    y_val = val['target']
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': SEED,
        'n_jobs': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
    
    start_time = time.time()
    model = lgb.train(params, train_data, num_boost_round=500,
                      valid_sets=[train_data, val_data], valid_names=['train', 'val'],
                      callbacks=callbacks)
    training_time = time.time() - start_time
    
    log_message(f"Direct Reg completed in {training_time:.1f}s, iter={model.best_iteration}", "SUCCESS")
    return model, training_time


# ============ 评估 ============

def compute_metrics(y_true_raw, y_pred_raw):
    """统一使用原始金额计算指标"""
    n = len(y_true_raw)
    
    # Spearman
    spearman_corr, _ = stats.spearmanr(y_true_raw, y_pred_raw)
    
    # Top-K% Capture
    y_true_rank = np.argsort(np.argsort(-y_true_raw))
    y_pred_rank = np.argsort(np.argsort(-y_pred_raw))
    
    def capture_rate(top_pct):
        k = max(1, int(n * top_pct))
        true_topk = set(np.where(y_true_rank < k)[0])
        pred_topk = set(np.where(y_pred_rank < k)[0])
        return len(true_topk & pred_topk) / len(true_topk) if len(true_topk) > 0 else 0
    
    # MAE in log space
    mae_log = np.mean(np.abs(np.log1p(y_true_raw) - np.log1p(np.maximum(y_pred_raw, 0))))
    
    # NDCG@100
    def ndcg_at_k(y_true, y_pred, k=100):
        pred_order = np.argsort(-y_pred)[:k]
        dcg = sum(y_true[idx] / np.log2(i + 2) for i, idx in enumerate(pred_order))
        sorted_true = np.sort(y_true)[::-1][:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(sorted_true))
        return dcg / idcg if idcg > 0 else 0
    
    return {
        'spearman': spearman_corr,
        'top_1pct': capture_rate(0.01),
        'top_5pct': capture_rate(0.05),
        'top_10pct': capture_rate(0.10),
        'mae_log': mae_log,
        'ndcg_100': ndcg_at_k(y_true_raw, y_pred_raw, k=100),
    }


def evaluate_models(direct_model, clf_model, reg_model, test, feature_cols):
    """评估并对比两个模型"""
    X = test[feature_cols]
    y_true_raw = test['gift_price'].values
    y_true_binary = test['is_gift'].values
    
    # Direct Regression: 预测 log(1+Y)，转回原始金额
    direct_pred_log = direct_model.predict(X)
    direct_pred_raw = np.expm1(np.maximum(direct_pred_log, 0))
    
    # Two-Stage: v(x) = p(x) * m(x)
    p_x = clf_model.predict(X)  # 概率
    m_x_log = reg_model.predict(X)  # 条件金额 (log space)
    m_x = np.expm1(np.maximum(m_x_log, 0))  # 转回原始
    v_x = p_x * m_x  # EV = P(gift) * E[amount|gift]
    
    # 评估
    log_message("=" * 50)
    log_message("EVALUATION RESULTS")
    log_message("=" * 50)
    
    direct_metrics = compute_metrics(y_true_raw, direct_pred_raw)
    twostage_metrics = compute_metrics(y_true_raw, v_x)
    
    # Stage 1 metrics
    pr_auc = auc(*precision_recall_curve(y_true_binary, p_x)[1::-1])
    roc = roc_auc_score(y_true_binary, p_x)
    
    print("\n📊 Direct Regression:")
    for k, v in direct_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print("\n📊 Two-Stage (p×m):")
    for k, v in twostage_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  Stage1 PR-AUC: {pr_auc:.4f}")
    print(f"  Stage1 ROC-AUC: {roc:.4f}")
    
    print("\n📊 Comparison (Two-Stage - Direct):")
    for k in direct_metrics:
        delta = twostage_metrics[k] - direct_metrics[k]
        better = "✅" if delta > 0 else "❌"
        print(f"  Δ {k}: {delta:+.4f} {better}")
    
    # 诊断：分析预测分布
    print("\n📊 Prediction Distribution Analysis:")
    print(f"  Direct pred: mean={direct_pred_raw.mean():.2f}, std={direct_pred_raw.std():.2f}, max={direct_pred_raw.max():.2f}")
    print(f"  Two-Stage v(x): mean={v_x.mean():.2f}, std={v_x.std():.2f}, max={v_x.max():.2f}")
    print(f"  p(x): mean={p_x.mean():.4f}, std={p_x.std():.4f}, max={p_x.max():.4f}")
    print(f"  m(x): mean={m_x.mean():.2f}, std={m_x.std():.2f}, max={m_x.max():.2f}")
    print(f"  y_true: mean={y_true_raw.mean():.2f}, std={y_true_raw.std():.2f}, max={y_true_raw.max():.2f}")
    
    return direct_metrics, twostage_metrics, {
        'direct_pred': direct_pred_raw,
        'two_stage_pred': v_x,
        'p_x': p_x,
        'm_x': m_x,
        'y_true': y_true_raw
    }


def main():
    log_message("=" * 70)
    log_message("Fair Comparison V2: Fixed Two-Stage Training")
    log_message("=" * 70)
    
    gift, user, streamer, room, click = load_data()
    df = prepare_features(gift, user, streamer, room, click)
    train, val, test = temporal_split(df)
    feature_cols = get_feature_columns(df)
    
    log_message(f"Features: {len(feature_cols)}")
    
    # Train models
    direct_model, direct_time = train_direct_regression(train, val, feature_cols)
    clf_model, clf_time = train_stage1_classifier_v2(train, val, feature_cols)
    reg_model, reg_time = train_stage2_regressor_v2(train, val, feature_cols)
    
    print("\n" + "=" * 70)
    print("Training Summary:")
    print(f"  Direct Reg: {direct_time:.1f}s, iter={direct_model.best_iteration}")
    print(f"  Stage 1 V2: {clf_time:.1f}s, iter={clf_model.best_iteration}")
    print(f"  Stage 2:    {reg_time:.1f}s, iter={reg_model.best_iteration}")
    print("=" * 70)
    
    # Evaluate
    direct_metrics, twostage_metrics, preds = evaluate_models(
        direct_model, clf_model, reg_model, test, feature_cols
    )
    
    # 保存结果
    results = {
        "experiment": "fair_comparison_v2",
        "timestamp": datetime.now().isoformat(),
        "train_size": len(train),
        "test_size": len(test),
        "direct_reg": direct_metrics,
        "two_stage": twostage_metrics,
        "training": {
            "direct_iter": direct_model.best_iteration,
            "stage1_iter": clf_model.best_iteration,
            "stage2_iter": reg_model.best_iteration,
        }
    }
    
    with open(RESULTS_DIR / "fair_comparison_v2_20260108.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    log_message("Results saved!", "SUCCESS")
    return results


if __name__ == "__main__":
    results = main()
