#!/usr/bin/env python3
"""
诊断脚本：分析 Two-Stage vs Direct Regression 的差异
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/home/swei20/GiftLive")
MODELS_DIR = BASE_DIR / "gift_allocation" / "models"
DATA_DIR = BASE_DIR / "data" / "KuaiLive"

# 加载模型
with open(MODELS_DIR / "fair_direct_reg_20260108.pkl", 'rb') as f:
    direct_model = pickle.load(f)
with open(MODELS_DIR / "fair_two_stage_clf_20260108.pkl", 'rb') as f:
    clf_model = pickle.load(f)
with open(MODELS_DIR / "fair_two_stage_reg_20260108.pkl", 'rb') as f:
    reg_model = pickle.load(f)

print("=" * 60)
print("模型训练信息")
print("=" * 60)
print(f"Direct Reg best iteration: {direct_model.best_iteration}")
print(f"Stage 1 (clf) best iteration: {clf_model.best_iteration}")
print(f"Stage 2 (reg) best iteration: {reg_model.best_iteration}")

print("\n问题1: Stage 1 只训练了 9 轮！")
print("这说明分类器很快就收敛了，可能是因为：")
print("  - 类别极度不平衡 (1.82% 正样本)")
print("  - early_stopping 在 binary_logloss 上太激进")
print("  - 需要更多轮次才能学到细粒度模式")

# 读取一些测试数据来分析预测分布
print("\n" + "=" * 60)
print("加载数据...")
print("=" * 60)

# 重新加载数据（简化版）
gift = pd.read_csv(DATA_DIR / "gift.csv")
click = pd.read_csv(DATA_DIR / "click.csv")
user = pd.read_csv(DATA_DIR / "user.csv")
streamer = pd.read_csv(DATA_DIR / "streamer.csv")
room = pd.read_csv(DATA_DIR / "room.csv")

# 简化的特征准备（只取一小部分）
print(f"Click samples: {len(click):,}")
print(f"Gift samples: {len(gift):,}")
print(f"Gift rate: {len(gift)/len(click)*100:.2f}%")

# 分析模型预测
print("\n" + "=" * 60)
print("模型特征重要性对比")
print("=" * 60)

# 获取特征列表
feature_names = direct_model.feature_name()
print(f"\n特征数量: {len(feature_names)}")

# Direct Reg top 10 特征
direct_imp = direct_model.feature_importance(importance_type='gain')
direct_top = sorted(zip(feature_names, direct_imp), key=lambda x: -x[1])[:10]
print("\n[Direct Regression] Top 10 特征:")
for name, imp in direct_top:
    print(f"  {name}: {imp:.1f}")

# Stage 1 top 10 特征
clf_imp = clf_model.feature_importance(importance_type='gain')
clf_top = sorted(zip(feature_names, clf_imp), key=lambda x: -x[1])[:10]
print("\n[Stage 1 Classification] Top 10 特征:")
for name, imp in clf_top:
    print(f"  {name}: {imp:.1f}")

# Stage 2 top 10 特征
reg_imp = reg_model.feature_importance(importance_type='gain')
reg_top = sorted(zip(feature_names, reg_imp), key=lambda x: -x[1])[:10]
print("\n[Stage 2 Regression] Top 10 特征:")
for name, imp in reg_top:
    print(f"  {name}: {imp:.1f}")

print("\n" + "=" * 60)
print("关键洞察")
print("=" * 60)
print("""
问题分析:

1. Stage 1 训练不充分 (9 轮 vs Direct 的 162 轮)
   - binary_logloss 收敛太快，但模型可能还没学到细粒度排序能力
   - 建议：换成 AUC 优化，或增加 min_data_in_leaf

2. Two-Stage 的 p(x) * m(x) 计算可能有问题：
   - Stage 2 只在 gift 样本上训练，但预测时对所有样本预测
   - 对非 gift 样本的 m(x) 预测可能不准
   
3. 评估指标的尺度问题：
   - Direct Reg 预测 log(1+Y)
   - Two-Stage 预测 p(x) * m(x)，其中 m(x) = expm1(log_pred)
   - 这两者不在同一尺度，可能影响 Top-K% Capture 计算

建议修复：
1. Stage 1 使用 AUC 优化而非 logloss
2. Stage 1 增加 min_child_samples 或 min_data_in_leaf
3. 比较时统一尺度：都用原始金额或都用 log 金额
4. 或者：Two-Stage 预测 E[Y] = p(x) * E[Y|gift]，然后取 log
""")
