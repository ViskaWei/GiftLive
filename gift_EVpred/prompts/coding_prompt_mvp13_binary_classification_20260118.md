# 🤖 Coding Prompt: Binary Classification - 任务降级验证

> **Experiment ID:** `EXP-20260118-gift_EVpred-05`
> **MVP:** MVP-1.3
> **Date:** 2026-01-18
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：验证二分类任务（P(gift>0)）是否比 EV 回归更可行，作为召回阶段或 Two-Stage 改进的基础

**验证假设**：

| 假设 | 验证方法 | 通过标准 |
|------|----------|----------|
| H3.1: 二分类任务可行 | 训练分类器，计算 AUC | AUC > 0.70 |
| H3.2: 分类比回归简单 | 对比 AUC 排名 vs Spearman 排名 | AUC > Frozen Two-Stage Stage1 (0.62) |
| H3.3: 二分类可用于召回 | 计算 Precision@K, Recall@K | Precision@1% > 5% |
| H3.4: 二分类+回归更优 | 优化后的 Two-Stage | Top-1% > 纯回归 11.6% |

**预期结果**：
- 若 AUC > 0.7 → 二分类可用于召回/筛选，继续探索 Two-Stage 改进
- 若 AUC < 0.65 → 当前特征对"是否送礼"也缺乏预测力，需更多特征工程

---

## 2. 🧪 实验设定

### 2.1 数据

```yaml
data:
  source: "KuaiLive"
  path: "data/KuaiLive/"
  sample_unit: "click-level (含 Y=0)"
  positive_rate: "~1.5% (gift > 0)"
  negative_rate: "~98.5% (gift = 0)"
  imbalance_ratio: "~66:1"
  train_size: "70% (按时间)"
  val_size: "15%"
  test_size: "15%"
  feature_version: "frozen (严格无泄漏)"
```

### 2.2 模型

```yaml
models:
  # Exp A: 默认 LightGBM
  exp_a:
    name: "LightGBM Default"
    objective: binary
    params:
      num_leaves: 31
      learning_rate: 0.05
      feature_fraction: 0.8
      bagging_fraction: 0.8
      early_stopping: 50

  # Exp B: 加权采样
  exp_b:
    name: "LightGBM + scale_pos_weight"
    objective: binary
    params:
      num_leaves: 31
      learning_rate: 0.05
      scale_pos_weight: "neg_count / pos_count (~66)"
      min_data_in_leaf: 100

  # Exp C: 欠采样
  exp_c:
    name: "LightGBM + Undersampling"
    objective: binary
    sampling:
      method: "random undersample negative"
      target_ratio: "1:10"

  # Exp D: Focal Loss (XGBoost)
  exp_d:
    name: "XGBoost + Focal Loss"
    objective: "binary:logistic"
    custom_loss: "focal_loss"
    params:
      gamma: 2.0
      alpha: 0.25

  # Exp E: Two-Stage 改进版
  exp_e:
    name: "Two-Stage Improved"
    stage1: "最佳二分类模型 (从 A-D 选)"
    stage2: "LightGBM regression on Y|Y>0"
    combine: "p(gift>0) × E[gift|gift>0]"
```

### 2.3 训练

```yaml
training:
  seed: 42
  n_estimators: 500
  early_stopping_rounds: 50
  split: "temporal (时间切分)"
  validation_metric: "auc"
```

### 2.4 评估指标

```yaml
metrics:
  classification:
    - AUC (ROC)
    - PR-AUC (更适合不平衡)
    - Precision@K (K=1%, 5%, 10%)
    - Recall@K (K=1%, 5%, 10%)
    - F1@optimal_threshold

  ev_prediction:
    - Top-1% Capture
    - Revenue Capture@1%
    - Spearman
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | line (多曲线) | FPR | TPR | `gift_EVpred/img/binary_roc_curves.png` |
| Fig2 | line (多曲线) | Recall | Precision | `gift_EVpred/img/binary_pr_curves.png` |
| Fig3 | line | Top-K% | Precision / Recall | `gift_EVpred/img/binary_precision_recall_at_k.png` |
| Fig4 | bar | Model | AUC / PR-AUC | `gift_EVpred/img/binary_model_comparison.png` |
| Fig5 | bar | Model (Two-Stage variants) | Top-1% / RevCap@1% | `gift_EVpred/img/binary_twostage_improvement.png` |

**图表要求**：
- 所有文字必须英文
- 包含 legend、title、axis labels
- 分辨率 ≥ 300 dpi
- **figsize 规则（必须遵守）**：
  - 单张图：`figsize=(6, 5)` 锁死
  - 多张图（subplot）：按 6:5 比例扩增，如 `(12, 5)` for 1×2, `(12, 10)` for 2×2

---

## 4. 📁 参考代码

> ⚠️ **不要在这里写代码！只写路径，让 Agent 自己读取**

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/train_leakage_free_baseline.py` | `load_data()`, `prepare_click_level_data()`, `create_past_only_features_frozen()`, `apply_frozen_features()`, `temporal_split()` | 添加二分类训练逻辑 |
| `scripts/train_leakage_free_baseline.py` | `train_two_stage_models()` Stage1 部分 | 提取为独立函数，添加不同方法 |
| `scripts/train_leakage_free_baseline.py` | `compute_top_k_capture()`, `compute_revenue_capture_at_k()` | 添加 Precision@K, Recall@K |
| `gift_EVpred/exp/exp_leakage_free_baseline_20260118.md` | 实验设计参考 | 理解 frozen 特征实现 |
| `gift_EVpred/exp/exp_binary_classification_20260118.md` | 实验计划 | 填写结果 |

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `gift_EVpred/exp/exp_binary_classification_20260118.md`
- **模板**: `_backend/template/exp.md`
- **必须包含**:
  - ⚡ 核心结论速览（一句话 + 关键数字）
  - 📊 实验图表（所有 5 张图 + 观察）
  - 📝 结论（假设验证 + 设计启示 + 下一步）

### 5.2 图表文件
- **路径**: `gift_EVpred/img/`
- **命名**:
  - `binary_roc_curves.png`
  - `binary_pr_curves.png`
  - `binary_precision_recall_at_k.png`
  - `binary_model_comparison.png`
  - `binary_twostage_improvement.png`

### 5.3 数值结果
- **格式**: JSON
- **路径**: `gift_EVpred/results/binary_classification_eval_20260118.json`
- **内容**:
  ```yaml
  exp_a_default:
    auc: float
    pr_auc: float
    precision_1pct: float
    recall_1pct: float
  exp_b_weighted:
    ...
  exp_c_undersample:
    ...
  exp_d_focal:
    ...
  exp_e_twostage_improved:
    top_1pct_capture: float
    revenue_capture_1pct: float
    spearman: float
  ```

---

## 6. 📤 报告抄送

完成后需同步更新：

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_EVpred_roadmap.md` | MVP-1.3 状态 + 结论快照 | §2.1, §4.3 |
| `gift_EVpred_hub.md` | H3 假设验证状态 + 洞见 | §1, §4 |

---

## 7. ⚠️ 注意事项

- [ ] 代码中添加 seed 固定随机性 (`seed=42`)
- [ ] 图表文字全英文
- [ ] **必须使用 Frozen 特征版本**（已验证无泄漏）
- [ ] 类不平衡处理：记录每种方法的 threshold 选择策略
- [ ] Focal Loss 实现：XGBoost 需自定义损失函数
- [ ] 保存完整日志到 `logs/binary_classification_20260118.log`
- [ ] 长时间任务使用 nohup 后台运行

---

## 8. 📋 实验矩阵详解

| 实验 | 方法 | 关键参数 | 预期效果 |
|------|------|---------|---------|
| **Exp A** | LightGBM Default | - | Baseline，可能 AUC 低 |
| **Exp B** | scale_pos_weight | weight=66 | 提升 Recall，可能降低 Precision |
| **Exp C** | Undersampling 1:10 | 负样本采样 10% | 加速训练，风险是丢失信息 |
| **Exp D** | Focal Loss | gamma=2, alpha=0.25 | 聚焦难样本，可能提升整体 |
| **Exp E** | Two-Stage 改进 | 最佳 Stage1 + Stage2 | 验证分工策略 |

---

## 9. 🔑 关键检查清单

| 检查项 | 通过标准 | 说明 |
|--------|----------|------|
| 数据构造正确 | Y=0 占比 ~98.5% | 与 MVP-1.0 一致 |
| 使用 Frozen 特征 | 特征重要性比 < 2x | 无泄漏 |
| AUC 计算正确 | sklearn.metrics.roc_auc_score | 二分类标签 |
| Precision/Recall@K | 手动验证 Top-K 样本 | 防止 off-by-one |
| Two-Stage 改进 | Stage1 用最佳模型 | 不是 Exp A |

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先读取"参考代码"中列出的文件
3. ✅ 理解现有代码逻辑后再修改
4. ✅ 复用已有函数，不要重复造轮子
5. ✅ 按模板输出 exp.md 报告
6. ✅ Frozen 特征版本是唯一正确的版本（Rolling 有泄漏问题）
-->
