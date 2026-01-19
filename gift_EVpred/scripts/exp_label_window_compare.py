#!/usr/bin/env python3
"""
Label Window Comparison Experiment
===================================
对比不同标签窗口设置对 Ridge Regression 的影响

窗口设置：
1. 固定 1h
2. 固定 30min
3. 固定 10min
4. min(1h, live_end) - 直播结束截断

Author: Viska Wei
Date: 2026-01-19
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR = Path("/home/swei20/GiftLive")
DATA_DIR = BASE_DIR / "data" / "KuaiLive"
OUTPUT_DIR = BASE_DIR / "gift_EVpred"

# =============================================================================
# 工具函数
# =============================================================================
def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "📝", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "")
    print(f"[{timestamp}] {prefix} {msg}")

def compute_revcap(y_true, y_pred, top_pct=0.01):
    """计算 RevCap@k%"""
    n = len(y_true)
    k = max(1, int(n * top_pct))
    top_idx = np.argsort(y_pred)[-k:]
    return y_true[top_idx].sum() / y_true.sum() if y_true.sum() > 0 else 0

# =============================================================================
# 数据加载
# =============================================================================
def load_data():
    log("Loading raw data...")
    gift = pd.read_csv(DATA_DIR / "gift.csv")
    click = pd.read_csv(DATA_DIR / "click.csv")
    room = pd.read_csv(DATA_DIR / "room.csv")
    user = pd.read_csv(DATA_DIR / "user.csv")
    streamer = pd.read_csv(DATA_DIR / "streamer.csv")

    log(f"Loaded: gift={len(gift):,}, click={len(click):,}, room={len(room):,}")
    return gift, click, room, user, streamer

# =============================================================================
# 标签构建（支持多种窗口）
# =============================================================================
def build_labels(click, gift, room, window_minutes, use_live_end=False):
    """
    构建标签

    Args:
        click: click DataFrame
        gift: gift DataFrame
        room: room DataFrame (含 end_timestamp)
        window_minutes: 窗口分钟数
        use_live_end: 是否用 min(window, live_end) 截断
    """
    click = click.copy()
    gift = gift.copy()

    # 删除 watch_live_time（泄漏特征）
    if 'watch_live_time' in click.columns:
        click = click.drop(columns=['watch_live_time'])

    # 转换时间
    click['timestamp_dt'] = pd.to_datetime(click['timestamp'], unit='ms')
    gift['timestamp_dt'] = pd.to_datetime(gift['timestamp'], unit='ms')

    # 计算窗口结束时间
    click['label_end_dt'] = click['timestamp_dt'] + pd.Timedelta(minutes=window_minutes)

    # 如果使用 live_end 截断
    if use_live_end:
        room_times = room[['live_id', 'end_timestamp']].drop_duplicates('live_id')
        room_times['live_end_dt'] = pd.to_datetime(room_times['end_timestamp'], unit='ms')
        click = click.merge(room_times[['live_id', 'live_end_dt']], on='live_id', how='left')

        # 截断：min(固定窗口, live_end)
        click['label_end_dt'] = click[['label_end_dt', 'live_end_dt']].min(axis=1)
        click = click.drop(columns=['live_end_dt'])

    # Merge and filter
    merged = click[['user_id', 'streamer_id', 'live_id', 'timestamp', 'timestamp_dt', 'label_end_dt']].merge(
        gift[['user_id', 'streamer_id', 'live_id', 'timestamp_dt', 'gift_price']],
        on=['user_id', 'streamer_id', 'live_id'],
        how='left',
        suffixes=('_click', '_gift')
    )

    # Filter: gift within window
    merged = merged[
        (merged['timestamp_dt_gift'] >= merged['timestamp_dt_click']) &
        (merged['timestamp_dt_gift'] <= merged['label_end_dt'])
    ]

    # Aggregate
    gift_agg = merged.groupby(
        ['user_id', 'streamer_id', 'live_id', 'timestamp']
    )['gift_price'].sum().reset_index().rename(columns={'gift_price': 'gift_price_label'})

    # Merge back
    click = click.merge(gift_agg, on=['user_id', 'streamer_id', 'live_id', 'timestamp'], how='left')
    click['gift_price_label'] = click['gift_price_label'].fillna(0)

    # 清理
    click = click.drop(columns=['label_end_dt', 'timestamp_dt'], errors='ignore')

    return click

# =============================================================================
# 7-7-7 划分
# =============================================================================
def split_by_days(df, train_days=7, val_days=7, test_days=7):
    from datetime import timedelta

    df = df.copy()
    df['_datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['_date'] = df['_datetime'].dt.date

    dates = sorted(df['_date'].unique())
    min_date = dates[0]

    train_end = min_date + timedelta(days=train_days - 1)
    val_start = train_end + timedelta(days=1)
    val_end = val_start + timedelta(days=val_days - 1)
    test_start = val_end + timedelta(days=1)
    test_end = test_start + timedelta(days=test_days - 1)

    train_df = df[df['_date'] <= train_end].copy()
    val_df = df[(df['_date'] >= val_start) & (df['_date'] <= val_end)].copy()
    test_df = df[(df['_date'] >= test_start) & (df['_date'] <= test_end)].copy()

    for d in [train_df, val_df, test_df]:
        d.drop(columns=['_datetime', '_date'], inplace=True, errors='ignore')

    return train_df, val_df, test_df

# =============================================================================
# Day-Frozen 特征（简化版）
# =============================================================================
def add_day_frozen_features(train_df, val_df, test_df, gift):
    """添加 Day-Frozen 历史特征（简化版，只用 pair/user/streamer count/sum）"""
    gift = gift.copy()
    gift['day'] = pd.to_datetime(gift['timestamp'], unit='ms').dt.normalize()

    # 获取 train 结束时间
    train_max_ts = train_df['timestamp'].max()
    train_end_day = pd.to_datetime(train_max_ts, unit='ms').normalize()

    # 只用 train 期间的 gift 构建 lookup
    gift_train = gift[gift['day'] <= train_end_day]

    # Pair-level
    pair_stats = gift_train.groupby(['user_id', 'streamer_id'])['gift_price'].agg(
        pair_gift_cnt='count',
        pair_gift_sum='sum'
    ).reset_index()

    # User-level
    user_stats = gift_train.groupby('user_id')['gift_price'].agg(
        user_gift_cnt='count',
        user_gift_sum='sum'
    ).reset_index()

    # Streamer-level
    str_stats = gift_train.groupby('streamer_id')['gift_price'].agg(
        str_gift_cnt='count',
        str_gift_sum='sum'
    ).reset_index()

    # Apply to all sets
    result = []
    for df in [train_df, val_df, test_df]:
        df = df.merge(pair_stats, on=['user_id', 'streamer_id'], how='left')
        df = df.merge(user_stats, on='user_id', how='left')
        df = df.merge(str_stats, on='streamer_id', how='left')

        # Fill NaN
        for col in ['pair_gift_cnt', 'pair_gift_sum', 'user_gift_cnt', 'user_gift_sum', 'str_gift_cnt', 'str_gift_sum']:
            df[col] = df[col].fillna(0)

        result.append(df)

    return result[0], result[1], result[2]

# =============================================================================
# 添加静态特征
# =============================================================================
def add_static_features(df, user, streamer):
    # User features (only numeric)
    user_cols = ['user_id', 'fans_num', 'follow_num']
    user_cols = [c for c in user_cols if c in user.columns]
    df = df.merge(user[user_cols], on='user_id', how='left')

    # Streamer features
    str_cols = ['streamer_id', 'fans_user_num', 'accu_live_cnt']
    str_cols = [c for c in str_cols if c in streamer.columns]
    df = df.merge(streamer[str_cols], on='streamer_id', how='left')

    # Time features
    ts_dt = pd.to_datetime(df['timestamp'], unit='ms')
    df['hour'] = ts_dt.dt.hour
    df['day_of_week'] = ts_dt.dt.dayofweek

    return df

# =============================================================================
# 训练和评估
# =============================================================================
def train_and_evaluate(train_df, test_df, feature_cols):
    """训练 Ridge 并评估"""
    X_train = train_df[feature_cols].fillna(0).values
    y_train = np.log1p(train_df['gift_price_label'].values)

    X_test = test_df[feature_cols].fillna(0).values
    y_test_raw = test_df['gift_price_label'].values

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练 Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = model.predict(X_test_scaled)

    # 评估
    revcap_1 = compute_revcap(y_test_raw, y_pred, 0.01)
    revcap_5 = compute_revcap(y_test_raw, y_pred, 0.05)

    # 正样本率
    gift_rate = (y_test_raw > 0).mean()
    total_revenue = y_test_raw.sum()

    return {
        'revcap_1': revcap_1,
        'revcap_5': revcap_5,
        'gift_rate': gift_rate,
        'total_revenue': total_revenue,
        'n_samples': len(test_df)
    }

# =============================================================================
# 主实验
# =============================================================================
def run_experiment():
    log("=" * 60)
    log("Label Window Comparison Experiment")
    log("=" * 60)

    # 加载数据
    gift, click, room, user, streamer = load_data()

    # 实验配置
    configs = [
        {'name': '1h_fixed', 'window_minutes': 60, 'use_live_end': False},
        {'name': '30min_fixed', 'window_minutes': 30, 'use_live_end': False},
        {'name': '10min_fixed', 'window_minutes': 10, 'use_live_end': False},
        {'name': '1h_live_end', 'window_minutes': 60, 'use_live_end': True},
        {'name': '30min_live_end', 'window_minutes': 30, 'use_live_end': True},
    ]

    results = []

    for cfg in configs:
        log(f"\n{'='*40}")
        log(f"Running: {cfg['name']}")
        log(f"  Window: {cfg['window_minutes']}min, Live-end truncation: {cfg['use_live_end']}")

        # 构建标签
        click_with_labels = build_labels(
            click, gift, room,
            window_minutes=cfg['window_minutes'],
            use_live_end=cfg['use_live_end']
        )

        log(f"  Gift rate: {(click_with_labels['gift_price_label'] > 0).mean()*100:.2f}%")
        log(f"  Total revenue: {click_with_labels['gift_price_label'].sum():,.0f}")

        # 划分
        train_df, val_df, test_df = split_by_days(click_with_labels)
        log(f"  Split: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

        # 添加特征
        train_df, val_df, test_df = add_day_frozen_features(train_df, val_df, test_df, gift)
        train_df = add_static_features(train_df, user, streamer)
        test_df = add_static_features(test_df, user, streamer)

        # 特征列 (only numeric - 历史特征 + 时间特征)
        feature_cols = [
            'pair_gift_cnt', 'pair_gift_sum',
            'user_gift_cnt', 'user_gift_sum',
            'str_gift_cnt', 'str_gift_sum',
            'hour', 'day_of_week'
        ]
        feature_cols = [c for c in feature_cols if c in train_df.columns]
        log(f"  Features: {feature_cols}")

        # 训练评估
        metrics = train_and_evaluate(train_df, test_df, feature_cols)
        metrics['config'] = cfg['name']
        metrics['window_minutes'] = cfg['window_minutes']
        metrics['use_live_end'] = cfg['use_live_end']

        log(f"  RevCap@1%: {metrics['revcap_1']*100:.2f}%")
        log(f"  RevCap@5%: {metrics['revcap_5']*100:.2f}%")

        results.append(metrics)

    # 汇总结果
    log("\n" + "=" * 60)
    log("Summary Results")
    log("=" * 60)

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    # 保存结果
    output_path = OUTPUT_DIR / "results" / "label_window_compare_20260119.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_json(output_path, orient='records', indent=2)
    log(f"\nResults saved to: {output_path}", "SUCCESS")

    # 分析
    log("\n" + "=" * 60)
    log("Analysis")
    log("=" * 60)

    baseline = results_df[results_df['config'] == '1h_fixed'].iloc[0]
    for _, row in results_df.iterrows():
        if row['config'] != '1h_fixed':
            rev_diff = (row['total_revenue'] - baseline['total_revenue']) / baseline['total_revenue'] * 100
            revcap_diff = (row['revcap_1'] - baseline['revcap_1']) * 100
            log(f"{row['config']} vs 1h_fixed:")
            log(f"  Revenue diff: {rev_diff:+.2f}%")
            log(f"  RevCap@1% diff: {revcap_diff:+.2f}pp")

    return results_df

if __name__ == '__main__':
    results = run_experiment()
