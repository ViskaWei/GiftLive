# 🤖 Coding Prompt: Two-Stage vs Direct Regression 公平对比

> **Experiment ID:** `EXP-20260108-gift-allocation-04`  
> **MVP:** MVP-1.1-fair  
> **Date:** 2026-01-08  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：在相同的 click 全量数据上公平对比两段式建模 (p×m) 与直接回归，关闭 DG1

**验证问题**：DG1 - 两段式 vs 直接回归的增益有多大？

**预期结果**：
- 若 Top-1% Capture 提升 ≥5% → 确认两段式路线，进入 Gate-1
- 若 Top-1% Capture 提升 <5% → 保留 Baseline 架构，考虑端到端

**背景**：
- MVP-1.1 实验揭示：原 Baseline 在 gift-only (72k) 训练，Two-Stage 在 click (4.9M) 训练
- 两者不可直接对比，需在同一数据集上重新评估

---

## 2. 🧪 实验设定

### 2.1 数据

```yaml
data:
  source: "KuaiLive"
  path: "data/KuaiLive/"
  base: "click 全量（含打赏+非打赏）"
  train_size: ~1.87M
  val_size: ~1.70M
  test_size: ~1.34M
  positive_rate: 1.93%  # 打赏样本占比
  split: "temporal (按天切分)"
```

**⚠️ 关键要求**：
1. 两模型使用完全相同的 train/val/test 切分
2. 目标变量统一为 `log(1+Y)`，其中 Y=0 表示无打赏
3. 评估在同一 test set 上进行

### 2.2 模型

```yaml
model_a:  # Direct Regression on Click (新增)
  name: "LightGBM Regression"
  data: "click 全量 (含 Y=0)"
  target: "log(1+Y)"
  objective: "regression"
  metric: "mae"
  
model_b:  # Two-Stage (已有，复用 MVP-1.1 代码)
  name: "Two-Stage LightGBM"
  stage1: "binary classification (is_gift)"
  stage2: "regression (amount | gift)"
  combined: "v(x) = p(x) × m(x)"
```

### 2.3 训练配置

```yaml
training:
  # 共同超参（与 MVP-1.1 保持一致）
  num_leaves: 31
  learning_rate: 0.05
  n_estimators: 500
  early_stopping: 50
  feature_fraction: 0.8
  bagging_fraction: 0.8
  seed: 42
  
  # Direct Reg 特有
  direct_reg:
    objective: "regression"
    metric: "mae"
  
  # Two-Stage 特有
  two_stage:
    stage1_objective: "binary"
    stage1_scale_pos_weight: "auto (n_neg/n_pos)"
    stage2_objective: "regression"
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | grouped bar | Model | Top-K% Capture | `gift_allocation/img/fair_comparison_topk.png` |
| Fig2 | line | Top-K% | Capture Rate | `gift_allocation/img/fair_comparison_topk_curve.png` |
| Fig3 | scatter | Predicted | Actual | `gift_allocation/img/fair_comparison_scatter.png` |
| Fig4 | 2×1 subplot | Feature | Importance | `gift_allocation/img/fair_comparison_feature_importance.png` |
| Fig5 | bar | Metric | Value | `gift_allocation/img/fair_comparison_all_metrics.png` |

**图表要求**：
- 所有文字必须英文
- 包含 legend、title、axis labels
- 分辨率 ≥ 300 dpi
- **figsize 规则**：
  - 单张图：`figsize=(6, 5)` 或 `(8, 6)`
  - 多张图（subplot）：按 6:5 比例扩增

---

## 4. 📁 参考代码

> ⚠️ **不要在这里写代码！先读取以下文件，理解后再修改**

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/train_two_stage.py` | `load_data()`, `prepare_features()`, `temporal_split()`, `get_feature_columns()`, `train_stage1_classifier()`, `train_stage2_regressor()`, `evaluate_two_stage()`, 所有 plot 函数 | 输出路径改为 `fair_*`; 添加 Direct Reg 模型 |
| `scripts/train_baseline_lgb.py` | `train_lightgbm()`, `evaluate_model()` | 适配 click 全量数据（含 Y=0 样本） |

### 关键修改点

1. **新增 Direct Regression on Click**:
   - 复用 `train_baseline_lgb.py` 的 `train_lightgbm()`
   - 但数据用 `train_two_stage.py` 的 `prepare_features()` 生成的 click 全量
   - 目标变量：`log(1+Y)`，含大量 Y=0 样本

2. **Two-Stage 保持不变**:
   - 直接复用 `train_two_stage.py` 的代码
   - 确保使用相同的 train/val/test 切分

3. **评估对齐**:
   - 两模型在同一 test set 上评估
   - 统一使用 `evaluate_two_stage()` 中的 Top-K% Capture、Spearman、NDCG 计算

4. **输出路径**:
   - 模型：`gift_allocation/models/fair_direct_reg_20260108.pkl`
   - 结果：`gift_allocation/results/fair_comparison_20260108.json`
   - 图表：`gift_allocation/img/fair_*.png`

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `gift_allocation/exp/exp_fair_comparison_20260108.md`
- **模板**: `_backend/template/exp.md`
- **必须包含**:
  - ⚡ 核心结论速览（一句话 + 关键数字对比表）
  - 📊 实验图表（所有 5 张图 + 观察）
  - 📝 结论（DG1 验证结果 + 设计决策）

### 5.2 图表文件
- **路径**: `gift_allocation/img/`
- **命名**: `fair_comparison_*.png`

### 5.3 数值结果
- **格式**: JSON
- **路径**: `gift_allocation/results/fair_comparison_20260108.json`
- **内容**:
```yaml
required_fields:
  - experiment_id
  - timestamp
  - train_size, val_size, test_size
  - models:
      direct_reg: {top_1pct, top_5pct, spearman, mae_log, ndcg_100}
      two_stage: {top_1pct, top_5pct, spearman, mae_log, ndcg_100, pr_auc, ece}
  - comparison:
      delta_top_1pct
      delta_spearman
      conclusion: "accept/reject"
      decision_rule
```

---

## 6. 📤 报告抄送

完成后需同步更新：

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_allocation/gift_allocation_roadmap.md` | MVP-1.1-fair 状态 → ✅; 结论快照 | §2.1, §4.3 |
| `gift_allocation/gift_allocation_hub.md` | DG1 关闭状态; 更新共识表 | §5 DG1, §L1 K1 |

---

## 7. ⚠️ 注意事项

- [ ] 代码中固定 `seed=42`
- [ ] 确保两模型使用相同的 train/val/test 切分（时间戳排序后切分）
- [ ] 图表文字全英文
- [ ] 保存完整日志到 `logs/fair_comparison_20260108.log`
- [ ] 使用 nohup 后台运行：`nohup python scripts/train_fair_comparison.py > logs/fair_comparison_20260108.log 2>&1 &`

---

## 8. 📐 验收标准

| 指标 | 门槛 | 决策 |
|------|------|------|
| Top-1% Capture 提升 | ≥ 5% | → 确认两段式，关闭 DG1 |
| Top-1% Capture 提升 | < 5% | → 保留 Baseline，考虑端到端 |
| PR-AUC (Stage 1) | > 0.60 | → Stage 1 分类有效 |

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先读取"参考代码"中列出的文件
3. ✅ 理解现有代码逻辑后再修改
4. ✅ 复用已有函数，不要重复造轮子
5. ✅ 按模板输出 exp.md 报告
6. ✅ 确保两模型在相同数据上训练和评估
-->
