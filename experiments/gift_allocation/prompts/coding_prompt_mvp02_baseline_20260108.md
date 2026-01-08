# 🤖 Coding Prompt: Baseline (直接回归)

> **Experiment ID:** `EXP-20260108-gift-allocation-02`  
> **MVP:** MVP-0.2  
> **Date:** 2026-01-08  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：建立直接回归预测打赏金额的 Baseline 模型，为后续两段式建模提供对比基准。

**验证假设**：H0 - 直接回归 log(1+Y) 可作为合理的 baseline

**预期结果**：
- 产出 Baseline 指标：MAE(log), Top-1%/5% 捕获率, Spearman 相关
- 作为 MVP-1.1 (两段式建模) 的对比基准
- 若 Top-1% 捕获率 < 30% → 说明有较大提升空间

---

## 2. 🧪 实验设定

### 2.1 数据

```yaml
data:
  source: "KuaiLive"
  path: "data/KuaiLive/"
  split:
    method: "temporal"  # 按时间切分，避免数据泄漏
    train: "前 N-14 天"
    val: "第 N-13 到 N-7 天"
    test: "最后 7 天"
  target:
    raw: "gift_amount"
    transform: "log(1+Y)"
  features:
    user: ["user_id", "历史打赏次数", "历史打赏金额", ...]  # 根据 EDA 确定
    streamer: ["streamer_id", "历史收到打赏", "粉丝数", ...]
    context: ["hour", "weekday", "room_id", ...]
    interaction: ["历史互动次数", "观看时长", ...]
```

### 2.2 模型

```yaml
model:
  name: "LightGBM"
  task: "regression"
  params:
    objective: "regression"
    metric: "mae"
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 500
    early_stopping_rounds: 50
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 5
    verbose: -1
```

### 2.3 训练

```yaml
training:
  seed: 42
  cv_folds: null  # 使用固定 train/val/test 切分
  early_stopping: true
  save_model: true
  model_path: "experiments/gift_allocation/models/baseline_lgb_20260108.pkl"
```

### 2.4 评估指标

```yaml
metrics:
  regression:
    - MAE(log)          # log(1+Y) 空间的 MAE
    - RMSE(log)
    - Spearman          # 排序能力
  
  ranking:
    - top_1pct_capture  # Top-1% 真实金主中，模型排名前 1% 能捕获多少
    - top_5pct_capture
    - top_10pct_capture
    - NDCG@100
  
  calibration:
    - bucket_mean_pred_vs_actual  # 分桶校准
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | scatter | Predicted log(1+Y) | Actual log(1+Y) | `experiments/gift_allocation/img/baseline_pred_vs_actual.png` |
| Fig2 | bar | Feature | Importance | `experiments/gift_allocation/img/baseline_feature_importance.png` |
| Fig3 | line | Top-K% Threshold | Capture Rate | `experiments/gift_allocation/img/baseline_topk_capture.png` |
| Fig4 | line | Training Iteration | MAE (train/val) | `experiments/gift_allocation/img/baseline_learning_curve.png` |
| Fig5 | bar | Bucket (pred decile) | Mean Actual vs Mean Pred | `experiments/gift_allocation/img/baseline_calibration.png` |

**图表要求**：
- 所有文字必须英文
- 包含 legend、title、axis labels
- 分辨率 ≥ 300 dpi
- Scatter plot 需含对角线参考线

---

## 4. 📁 参考代码

> ⚠️ **不要在这里写代码！只写路径，让 Agent 自己读取**

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `_backend/scripts/training/driver.py` | 训练流程框架 | 适配 LightGBM |
| `_backend/scripts/training/post_process.py` | 结果后处理 | 添加排名指标 |

**依赖 MVP-0.1 的输出**：
- 数据 schema 理解
- 特征工程思路（基于 EDA 发现）
- 关键数字基准

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `experiments/gift_allocation/exp_baseline_20260108.md`
- **模板**: `_backend/template/exp.md`
- **必须包含**:
  - ⚡ 核心结论速览（MAE, Top-1% 捕获率, Spearman）
  - 📊 实验图表（所有图 + 观察）
  - 📝 结论（baseline 性能分析 + 改进方向）

### 5.2 图表文件
- **路径**: `experiments/gift_allocation/img/`

### 5.3 数值结果
- **格式**: JSON
- **路径**: `experiments/gift_allocation/results/baseline_results_20260108.json`
- **内容**:
```json
{
  "model": "LightGBM",
  "target": "log(1+Y)",
  "train_size": N,
  "val_size": N,
  "test_size": N,
  "metrics": {
    "mae_log": X.XXX,
    "rmse_log": X.XXX,
    "spearman": X.XXX,
    "top_1pct_capture": X.XXX,
    "top_5pct_capture": X.XXX,
    "top_10pct_capture": X.XXX
  },
  "top_features": [
    {"name": "feature1", "importance": X.XXX},
    {"name": "feature2", "importance": X.XXX}
  ],
  "training_time_seconds": X
}
```

### 5.4 模型文件
- **路径**: `experiments/gift_allocation/models/baseline_lgb_20260108.pkl`

---

## 6. 📤 报告抄送

完成后需同步更新：

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_allocation_roadmap.md` | MVP-0.2 状态 ✅ + 结论快照 | §2.1, §4.3 |
| `gift_allocation_hub.md` | Baseline 指标（权威数字） | §0 共识表, 权威数字 |

---

## 7. ⚠️ 注意事项

- [ ] 依赖 MVP-0.1 完成（需要数据理解）
- [ ] 代码中添加 seed=42 固定随机性
- [ ] 图表文字全英文
- [ ] 按时间切分数据，不要随机切分
- [ ] 保存完整日志到 `logs/`
- [ ] 长时间任务使用 nohup 后台运行

---

## 8. 📋 分析要点

完成 Baseline 后需回答：

1. **预测误差主要来自哪里？** → 高金额样本 vs 低金额样本
2. **哪些特征最重要？** → 指导后续特征工程
3. **Top-1% 捕获率如何？** → 核心业务指标
4. **预测是否校准？** → 是否需要校准层
5. **与 MVP-1.1 的预期差异？** → 两段式的提升空间在哪

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先确认 MVP-0.1 已完成（需要数据理解）
3. ✅ 参考 EDA 结果设计特征
4. ✅ 按模板输出 exp.md 报告
5. ✅ 所有图表保存到 img/ 目录
6. ✅ 数值结果保存为 JSON
7. ✅ 保存训练好的模型
-->
