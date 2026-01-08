#!/usr/bin/env python3
"""
验证评估指标是否正确
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

BASE_DIR = Path("/home/swei20/GiftLive")
MODELS_DIR = BASE_DIR / "gift_allocation" / "models"
DATA_DIR = BASE_DIR / "data" / "KuaiLive"

print("=" * 70)
print("🔍 验证评估指标")
print("=" * 70)

# 加载数据（简化版，直接用测试数据）
print("\n加载数据...")
gift = pd.read_csv(DATA_DIR / "gift.csv")
click = pd.read_csv(DATA_DIR / "click.csv")

print(f"Click 总数: {len(click):,}")
print(f"Gift 总数: {len(gift):,}")

# 构建测试数据的 y_true
# 聚合每个 click 的打赏金额
gift_agg = gift.groupby(['user_id', 'streamer_id', 'live_id']).agg({
    'gift_price': 'sum'
}).reset_index()

click_with_gift = click.merge(
    gift_agg, 
    on=['user_id', 'streamer_id', 'live_id'], 
    how='left'
)
click_with_gift['gift_price'] = click_with_gift['gift_price'].fillna(0)

y_true = click_with_gift['gift_price'].values

print(f"\n📊 y_true 分布分析:")
print(f"  总样本数: {len(y_true):,}")
print(f"  Y=0 样本数: {(y_true == 0).sum():,} ({(y_true == 0).mean()*100:.2f}%)")
print(f"  Y>0 样本数: {(y_true > 0).sum():,} ({(y_true > 0).mean()*100:.2f}%)")
print(f"  Mean: {y_true.mean():.2f}")
print(f"  Max: {y_true.max():.2f}")

# Top-1% 对应多少样本？
n = len(y_true)
k_1pct = int(n * 0.01)
print(f"\n  Top-1% = {k_1pct:,} 样本")

# Top-1% 的 y_true 阈值是多少？
y_sorted = np.sort(y_true)[::-1]
threshold_1pct = y_sorted[k_1pct-1] if k_1pct > 0 else 0
print(f"  Top-1% 阈值: Y >= {threshold_1pct:.2f}")

# 有多少样本的 Y >= 这个阈值？
n_above_threshold = (y_true >= threshold_1pct).sum()
print(f"  Y >= {threshold_1pct:.2f} 的样本数: {n_above_threshold:,}")

print("\n" + "=" * 70)
print("💡 关键发现")
print("=" * 70)

# 计算真正的 Top-1% 样本
y_rank = np.argsort(np.argsort(-y_true))
true_top1pct = set(np.where(y_rank < k_1pct)[0])

# 这些 Top-1% 样本的特征
top1pct_values = y_true[list(true_top1pct)]
print(f"\n真正 Top-1% 样本的 y_true 分布:")
print(f"  样本数: {len(true_top1pct):,}")
print(f"  Min: {top1pct_values.min():.2f}")
print(f"  Mean: {top1pct_values.mean():.2f}")
print(f"  Max: {top1pct_values.max():.2f}")

# 有多少 Y=0 在 Top-1% 里？
zero_in_top1pct = (top1pct_values == 0).sum()
print(f"  Y=0 的数量: {zero_in_top1pct:,} ({zero_in_top1pct/len(true_top1pct)*100:.2f}%)")

print("\n" + "=" * 70)
print("🧪 模拟不同预测器的 Top-1% Capture")
print("=" * 70)

def compute_top1pct_capture(y_true, y_pred):
    n = len(y_true)
    k = int(n * 0.01)
    y_true_rank = np.argsort(np.argsort(-y_true))
    y_pred_rank = np.argsort(np.argsort(-y_pred))
    true_topk = set(np.where(y_true_rank < k)[0])
    pred_topk = set(np.where(y_pred_rank < k)[0])
    return len(true_topk & pred_topk) / len(true_topk)

# 模拟器1: 随机预测
np.random.seed(42)
y_random = np.random.randn(len(y_true))
capture_random = compute_top1pct_capture(y_true, y_random)
print(f"\n1. 随机预测: Top-1% Capture = {capture_random*100:.2f}%")
print(f"   (理论值应该接近 1%)")

# 模拟器2: 完美预测
capture_perfect = compute_top1pct_capture(y_true, y_true)
print(f"\n2. 完美预测: Top-1% Capture = {capture_perfect*100:.2f}%")
print(f"   (应该是 100%)")

# 模拟器3: 预测 Y > 0 (二分类)
y_binary = (y_true > 0).astype(float)
capture_binary = compute_top1pct_capture(y_true, y_binary)
print(f"\n3. 预测 Y>0 (0/1): Top-1% Capture = {capture_binary*100:.2f}%")

# 模拟器4: 预测 log(1+Y)
y_log = np.log1p(y_true)
capture_log = compute_top1pct_capture(y_true, y_log)
print(f"\n4. 预测 log(1+Y): Top-1% Capture = {capture_log*100:.2f}%")
print(f"   (应该是 100%，因为 log 是单调的)")

print("\n" + "=" * 70)
print("🔍 检查实验中的预测值")
print("=" * 70)

# 加载模型
with open(MODELS_DIR / "fair_direct_reg_20260108.pkl", 'rb') as f:
    direct_model = pickle.load(f)

print("\nDirect Regression 模型预测分析:")
print(f"  模型训练在 log(1+Y) 目标上")
print(f"  预测值 y_pred_log = model.predict(X)")
print(f"  转换回原始: y_pred_raw = expm1(max(y_pred_log, 0))")

print("\nTwo-Stage 模型预测分析:")
print(f"  Stage 1: p(x) = P(Y>0|x)")
print(f"  Stage 2: m(x) = E[log(1+Y)|Y>0,x]")
print(f"  组合: v(x) = p(x) * expm1(m(x))")

print("\n⚠️ 潜在问题:")
print("""
1. 【尺度不一致】
   - Direct Reg 预测 log(1+Y)，转换后是 E[Y]
   - Two-Stage 预测 p(x) * m(x)，其中 m(x) 是 expm1(log_pred)
   - 问题：Stage 2 预测的是 E[log(1+Y)|gift]，不是 E[Y|gift]！
   
2. 【正确的 Two-Stage 应该是】
   v(x) = P(Y>0|x) * E[Y|Y>0,x]
   
   但我们实现的是：
   v(x) = P(Y>0|x) * expm1(E[log(1+Y)|Y>0,x])
   
   这两者不等价！因为 E[expm1(log_pred)] ≠ expm1(E[log_pred])
   
3. 【Jensen 不等式】
   对于凸函数 f (expm1 是凸的)：
   E[f(X)] >= f(E[X])
   
   所以 Two-Stage 的预测值可能系统性偏高！
""")
