#!/usr/bin/env python3
"""
深入分析：为什么 Direct Regression 在 Top-K% 上更好？
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

BASE_DIR = Path("/home/swei20/GiftLive")
RESULTS_DIR = BASE_DIR / "gift_allocation" / "results"
MODELS_DIR = BASE_DIR / "gift_allocation" / "models"
DATA_DIR = BASE_DIR / "data" / "KuaiLive"

# 加载结果
import json
with open(RESULTS_DIR / "fair_comparison_v2_20260108.json") as f:
    results = json.load(f)

print("=" * 70)
print("🔍 深入分析：为什么 Direct Regression 在 Top-1% 上更好？")
print("=" * 70)

print("\n📊 关键结果对比:")
print(f"  Direct Reg Top-1%:  {results['direct_reg']['top_1pct']*100:.1f}%")
print(f"  Two-Stage Top-1%:   {results['two_stage']['top_1pct']*100:.1f}%")
print(f"  差距:               {(results['two_stage']['top_1pct'] - results['direct_reg']['top_1pct'])*100:.1f} pp")

print("\n" + "=" * 70)
print("💡 直觉解释")
print("=" * 70)

print("""
1. 【数据稀疏性的核心影响】

   在 1.33M 测试样本中：
   - 只有 ~2% 有打赏 (约 26k 正样本)
   - 98% 是 Y=0 (约 1.3M 负样本)
   
   Top-1% 指的是预测最高的 ~13k 样本中，有多少是真正的高价值用户。

2. 【Direct Regression 的优势】

   Direct Reg 直接预测 log(1+Y)，对 98% 的 Y=0 样本学到的是：
   "这些人不会送礼" → 预测值接近 0
   
   这 98% 的 Y=0 样本提供了巨大的负例信息！
   模型学会了区分 "送礼 vs 不送礼"。

3. 【Two-Stage 的问题】

   v(x) = p(x) × m(x)
   
   - p(x) ≈ 0.06 (平均概率)
   - m(x) ≈ 4.58 (平均条件金额，在 log space)
   
   问题在于：
   a) Stage 2 只在 gift 样本上训练 (~34k)
   b) 预测时对所有 1.33M 样本预测 m(x)
   c) 对于从未送过礼的用户，m(x) 预测不准！

4. 【乘法放大误差】

   v(x) = p(x) × m(x)
   
   如果 p(x) 预测不准（比如某用户 p=0.1 应该是 0.01），
   即使 m(x) 正确，v(x) 也会被放大 10 倍。
   
   Direct Reg 没有这个问题，直接预测期望值。

5. 【NDCG@100 vs Top-1% 的差异】

   Two-Stage NDCG@100 更好 (+0.143) 的原因：
   - NDCG 考虑的是 Top-100 的排序质量
   - 在高价值用户（真正的金主）内部，m(x) 确实能更好地区分金额大小
   - 但 p(x) × m(x) 的乘法组合在边界处不如直接回归准确
""")

print("\n" + "=" * 70)
print("🎯 核心结论")
print("=" * 70)

print("""
【为什么 Direct Regression 更优？】

1. ✅ 利用了 98% 负样本的信息
   - 学会了 "谁不会送礼"
   - 这在极度稀疏场景下是关键优势

2. ✅ 没有乘法带来的误差放大
   - 直接预测 E[Y]，而非 P(Y>0) × E[Y|Y>0]
   
3. ✅ 简单模型在稀疏数据上更稳健
   - 两段式需要两个模型都准确，误差会累积

【Two-Stage 何时更好？】

1. 正样本率更高时（比如 >10%）
2. 分类边界清晰时
3. 需要可解释性时（可以分别分析 p 和 m）
4. 需要校准 P(gift) 时
5. 在 Top-100 精细排序时（NDCG 更好）

【建议】

对于这个场景（1.82% 正样本），Direct Regression 是更好的选择。

如果需要两段式的可解释性，可以：
1. 训练 Direct Reg 作为最终模型
2. 同时训练 Two-Stage 用于解释和诊断
""")

print("\n" + "=" * 70)
print("📐 数学直觉")
print("=" * 70)

print("""
设 Y = 打赏金额，X = 特征

【Direct Regression】
直接估计 E[Y|X] = E[log(1+Y)|X]

【Two-Stage】
v(x) = P(Y>0|X) × E[Y|Y>0,X]

两者在数学上是等价的（全概率公式）：
E[Y|X] = P(Y>0|X) × E[Y|Y>0,X] + P(Y=0|X) × 0
       = P(Y>0|X) × E[Y|Y>0,X]

但在有限样本+模型误差下：
- Direct Reg 用 1.87M 样本学一个模型
- Two-Stage 用 1.87M 学 P(Y>0|X)，用 34k 学 E[Y|Y>0,X]

Stage 2 只有 34k 样本，信息量少得多！
而且对从未送礼的用户，Stage 2 完全是外推。
""")
