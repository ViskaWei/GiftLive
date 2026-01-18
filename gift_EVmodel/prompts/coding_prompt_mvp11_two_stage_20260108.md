# 🤖 Coding Prompt: 两段式建模对比实验

> **Experiment ID:** `EXP-20260108-gift-allocation-03`  
> **MVP:** MVP-1.1  
> **Date:** 2026-01-08  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：验证两段式建模（是否打赏 × 条件金额）是否优于直接回归 Baseline

**验证假设**：DG1 - 两段式 vs 直接回归的增益有多大？

**决策规则**：
- 若 PR-AUC 提升 > 5% 且 Top-1% Capture 有改善 → 确认两段式路线
- 若提升不显著 → 考虑端到端或更简单方案

**预期结果**：
- 若 两段式 Top-1% > 56.2% → 验证两段式有效，进入 Gate-1 下一阶段
- 若 两段式 Top-1% ≤ 56.2% → 直接回归已足够好，需重新评估复杂度收益

---

## 2. 🧪 实验设定

### 2.1 数据

```yaml
data:
  source: "KuaiLive"
  path: "data/KuaiLive/"
  train_size: ~26,000
  val_size: ~24,000
  test_size: ~23,000
  features: 68
  split: "temporal (same as MVP-0.2)"
```

### 2.2 模型架构

**两段式模型**：

$$
v(x) = p(x) \cdot m(x)
$$

其中：
- $p(x) = \Pr(Y > 0 | x)$：是否打赏的二分类概率
- $m(x) = \mathbb{E}[Y | Y > 0, x]$：给定打赏的条件金额

```yaml
model:
  stage1_classification:
    name: "LightGBM"
    objective: "binary"
    metric: "binary_logloss"
    label: "is_gift (binary: 1 if gift_price > 0)"
    loss_variant: "focal_loss (optional)"  # 处理极端不均衡
    params:
      num_leaves: 31
      learning_rate: 0.05
      feature_fraction: 0.8
      bagging_fraction: 0.8
      scale_pos_weight: "auto (= n_neg / n_pos)"

  stage2_regression:
    name: "LightGBM"
    objective: "regression"
    metric: "mae"
    label: "log(1 + gift_price) | gift_price > 0"
    train_filter: "only samples with gift_price > 0"
    params:
      num_leaves: 31
      learning_rate: 0.05
      feature_fraction: 0.8
      bagging_fraction: 0.8
```

### 2.3 训练

```yaml
training:
  stage1:
    num_boost_round: 500
    early_stopping: 50
    seed: 42
  stage2:
    num_boost_round: 500
    early_stopping: 50
    seed: 42
```

### 2.4 关键变量

```yaml
variants:
  - name: "two_stage_vanilla"
    description: "标准两段式 (no focal loss)"
  - name: "two_stage_focal"
    description: "两段式 + focal loss (Stage 1)"
    focal_gamma: 2.0
    focal_alpha: 0.25

baseline:
  name: "direct_regression"
  reference: "MVP-0.2"
  metrics:
    top_1pct_capture: 0.562
    spearman: 0.891
    mae_log: 0.263
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | bar | Model (Baseline / Two-Stage / Two-Stage+Focal) | Top-1% Capture | `../img/two_stage_comparison_topk.png` |
| Fig2 | line | Top-K% Threshold (1-50%) | Capture Rate | `../img/two_stage_topk_curve.png` |
| Fig3 | bar | Model | PR-AUC (Stage 1) | `../img/two_stage_prauc.png` |
| Fig4 | scatter | Predicted p(x) | Actual Gift Rate (binned) | `../img/two_stage_calibration_stage1.png` |
| Fig5 | scatter | Predicted v(x) = p(x)·m(x) | Actual Y | `../img/two_stage_pred_vs_actual.png` |
| Fig6 | bar (horizontal) | Feature | Importance | `../img/two_stage_feature_importance.png` |

**图表要求**：
- 所有文字必须英文
- 包含 legend、title、axis labels
- 分辨率 ≥ 300 dpi
- 对比图使用统一配色方案

---

## 4. 📁 参考代码

> ⚠️ **不要在这里写代码！只写路径，让 Agent 自己读取**

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/train_baseline_lgb.py` | `load_data()`, `create_*_features()`, `temporal_split()`, `encode_categorical()` | 添加两段式训练逻辑 |
| `scripts/train_baseline_lgb.py` | `evaluate_model()`, `plot_*()` | 适配两段式评估 |
| `scripts/train_baseline_lgb.py` | `save_results()` | 添加 Stage1/Stage2 分别的指标 |

**重要参考**：
- 特征工程：完全复用 MVP-0.2 的特征
- 数据切分：复用 `temporal_split()` 保证可比性
- 评估指标：复用 `evaluate_model()` 中的 Top-K% capture 逻辑

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `exp/exp_two_stage_20260108.md`
- **模板**: `_backend/template/exp.md`
- **必须包含**:
  - ⚡ 核心结论速览（一句话 + 假设验证 + 关键数字）
  - 📊 实验图表（6 张图 + 每张的关键观察）
  - 📝 结论（两段式是否优于 Baseline）

### 5.2 图表文件
- **路径**: `../img/two_stage_*.png`
- **命名**: 
  - `two_stage_comparison_topk.png`
  - `two_stage_topk_curve.png`
  - `two_stage_prauc.png`
  - `two_stage_calibration_stage1.png`
  - `two_stage_pred_vs_actual.png`
  - `two_stage_feature_importance.png`

### 5.3 数值结果
- **格式**: JSON
- **路径**: `../results/two_stage_results_20260108.json`
- **结构**:
```json
{
  "experiment_id": "EXP-20260108-gift-allocation-03",
  "models": {
    "baseline": { "top_1pct_capture": 0.562, ... },
    "two_stage_vanilla": { ... },
    "two_stage_focal": { ... }
  },
  "comparison": {
    "delta_top_1pct": ...,
    "delta_prauc": ...,
    "conclusion": "accept/reject"
  }
}
```

### 5.4 模型文件
- **路径**: `gift_allocation/models/`
- **文件**:
  - `two_stage_clf_20260108.pkl` (Stage 1 分类器)
  - `two_stage_reg_20260108.pkl` (Stage 2 回归器)

---

## 6. 📤 报告抄送

完成后需同步更新：

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_EVmodel_roadmap.md` | MVP-1.1 状态 → ✅，结论快照 | §2.1 总览, §4.3 结论快照 |
| `gift_EVmodel_hub.md` | Q1.1 验证结果，DG1 关闭状态 | §1 假设树, §5 Decision Gaps |

---

## 7. ⚠️ 注意事项

- [ ] 代码中固定 seed=42 保证可复现
- [ ] Stage 2 只在打赏样本上训练（gift_price > 0）
- [ ] 最终预测使用 v(x) = p(x) · m(x)
- [ ] 图表文字全英文
- [ ] 保存完整日志到 `logs/two_stage_20260108.log`
- [ ] 长时间任务使用 nohup 后台运行

---

## 8. 📐 评估指标定义

### 8.1 Stage 1 分类评估

| 指标 | 定义 | 目的 |
|------|------|------|
| PR-AUC | Precision-Recall AUC | 处理极不均衡，比 ROC-AUC 更合适 |
| ECE | Expected Calibration Error | 概率校准程度 |
| Precision@K | 预测 Top-K 中真正打赏的比例 | 排序能力 |

### 8.2 联合评估

| 指标 | 定义 | 目的 |
|------|------|------|
| Top-K% Capture | 预测 Top-K% 与真实 Top-K% 的重叠率 | 金主捕获能力 |
| Spearman | 排名相关系数 | 排序一致性 |
| MAE(log) | Mean Absolute Error on log(1+Y) | 预测准确度 |

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先读取"参考代码"中列出的文件 (scripts/train_baseline_lgb.py)
3. ✅ 理解现有代码逻辑后再修改
4. ✅ 复用已有函数，不要重复造轮子
5. ✅ 按模板输出 exp.md 报告
6. ✅ 保持数据切分一致性（temporal_split）
7. ✅ 对比必须公平（相同特征、相同切分）
-->
