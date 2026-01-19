# 🤖 Coding Prompt: 预测目标扩展

> **Experiment ID:** `EXP-20260119-EVpred-05`
> **MVP:** MVP-4.0
> **Date:** 2026-01-19
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：从"预测 gift"扩展到"预测行动后果"，构建用户留存标签并验证可预测性

**验证假设**：H2.4 - 多目标预测框架可以保持短期 RevCap 不退化的同时提供留存预测信号

**预期结果**：
- 若 留存 AUC > 0.6 → 留存可预测，可构建多目标框架
- 若 留存 AUC < 0.55 → 留存难预测，需重新定义标签或特征

---

## 2. 🧪 实验设定

### 2.1 数据（强制使用 data_utils）

```yaml
data:
  source: "KuaiLive"
  split: "7-7-7 by days"
  train_size: ~1.6M
  val_size: ~1.7M
  test_size: ~1.4M

  # 现有标签
  existing_labels:
    - gift_price_label  # 短期收益（已有）
    - target_raw        # = gift_price_label
    - is_gift           # = (gift_price_label > 0)

  # 新增标签（本实验构建）
  new_labels:
    - return_1d         # 用户次日是否回访
    - return_7d         # 用户 7 天内是否回访
```

### 2.2 新标签构建逻辑

#### 2.2.1 用户留存标签

| 标签 | 定义 | 类型 | 构建逻辑 |
|------|------|------|---------|
| `return_1d` | 次日是否有 click | 二分类 | `exists click in [day+1, day+2)` |
| `return_7d` | 7 天内是否有 click | 二分类 | `exists click in [day+1, day+8)` |

**构建步骤**：
1. 从 `click.csv` 提取 `user_id`, `timestamp`
2. 计算每个 click 的 `click_day`
3. 对每个 click，检查该 user 在 `click_day + 1` 是否有 click → `return_1d`
4. 对每个 click，检查该 user 在 `[click_day + 1, click_day + 8)` 是否有 click → `return_7d`

**注意**：
- Test 集最后几天的样本无法计算留存（超出数据范围）→ 需 mask 掉
- 7-7-7 划分后，Test 只有前 1-2 天可计算 `return_7d`

#### 2.2.2 生态基线指标

| 指标 | 定义 | 粒度 | 计算方式 |
|------|------|------|---------|
| `Gini(exposure)` | 曝光分布基尼系数 | 按天 | 每天各主播 click 数的 Gini |
| `Gini(revenue)` | 收入分布基尼系数 | 按天 | 每天各主播 gift 金额的 Gini |
| `tail_coverage` | 长尾主播覆盖率 | 按天 | 收到曝光的 streamer / 总 streamer |

### 2.3 模型（方案 A: 独立模型）

```yaml
# 短期收益模型（复用 baseline）
revenue_model:
  name: "Ridge"
  params:
    alpha: 1.0
  target: "target_raw"  # gift_price_label
  metric: "RevCap@1%"

# 留存预测模型（新增）
retention_model:
  name: "LogisticRegression"
  params:
    C: 1.0
    max_iter: 1000
  target: "return_1d"  # 或 return_7d
  metric: "AUC"
```

### 2.4 评估指标

| 目标 | 主指标 | 阈值 | 辅助指标 |
|------|--------|------|---------|
| 短期收益 | RevCap@1% | ≥ 51.4% | RevCap 曲线 |
| 用户留存 | AUC | > 0.6 | Precision@K, Recall@K |
| 生态健康 | Gini | 仅观察 | tail_coverage |

### 2.5 实验步骤

| Step | 任务 | 输入 | 输出 |
|------|------|------|------|
| **1** | 构建留存标签 | `click.csv` | `return_1d`, `return_7d` 列 |
| **2** | 留存率 EDA | 标签 | 留存率分布、可预测性初判 |
| **3** | 训练留存模型 | 现有特征 + 留存标签 | LR 模型 |
| **4** | 评估留存 AUC | 模型预测 | AUC 报告 |
| **5** | 计算生态基线 | `click.csv`, `gift.csv` | Gini, tail_coverage |
| **6** | 对比分析 | 短期 vs 留存 | 相关性分析 |

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | Bar | Day | Retention Rate (%) | `gift_EVpred/img/retention_rate_by_day.png` |
| Fig2 | ROC Curve | FPR | TPR | `gift_EVpred/img/retention_roc_curve.png` |
| Fig3 | Bar | Day | Gini(exposure) | `gift_EVpred/img/gini_exposure_by_day.png` |
| Fig4 | Scatter | RevCap@1% | Retention AUC | `gift_EVpred/img/revcap_vs_retention.png` |

**图表要求**：
- 所有文字英文
- figsize: 单张 (6, 5)，多张按 6:5 扩增
- 分辨率 >= 300 dpi

---

## 4. 📁 参考代码

> ⚠️ **不要在这里写代码！只写路径，让 Agent 自己读取**

| 参考脚本 | 可复用 | 需修改/扩展 |
|---------|--------|------------|
| **`gift_EVpred/data_utils.py`** | `prepare_dataset()`, `split_by_days()` | 添加 `add_retention_labels()` 函数 |
| **`gift_EVpred/metrics.py`** | `revenue_capture_at_k()` | 添加 `gini_coefficient()`, `retention_metrics()` |
| `gift_EVpred/exp/exp_baseline_ridge_20260119.md` | 实验报告模板 | - |
| `gift_EVpred/results/baseline_ridge_final_20260119.json` | baseline 结果格式 | - |

### 关键复用点

1. **数据加载**：复用 `prepare_dataset()`，在返回前添加留存标签
2. **特征列**：复用 `get_feature_columns()`，短期/留存共用同一特征集
3. **短期评估**：复用 `evaluate_model()` 验证 RevCap 不退化
4. **结果保存**：复用 JSON 格式

### 新增函数设计

```yaml
# data_utils.py 新增
add_retention_labels:
  input: df (with timestamp, user_id), click_df (full)
  output: df with return_1d, return_7d columns
  logic:
    - 按 user_id + day 分组
    - 检查 day+1 / day+1:day+8 是否有 click

# metrics.py 新增
gini_coefficient:
  input: array of values
  output: float (0~1)

retention_metrics:
  input: y_true, y_pred (binary)
  output: dict with AUC, precision@k, recall@k
```

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `gift_EVpred/exp/exp_prediction_target_expansion_20260119.md`
- **模板**: 已创建，需补充实验结果部分
- **必须包含**:
  - ⚡ 核心结论速览（留存 AUC + RevCap 对比）
  - 📊 所有图表 + 观察
  - 📝 结论（Q2.4 验证结果）

### 5.2 图表文件
- **路径**: `gift_EVpred/img/`
- **命名**:
  - `retention_rate_by_day.png`
  - `retention_roc_curve.png`
  - `gini_exposure_by_day.png`
  - `revcap_vs_retention.png`

### 5.3 数值结果
- **路径**: `gift_EVpred/results/prediction_target_expansion_20260119.json`
- **格式**:
```yaml
config:
  experiment_id: "EXP-20260119-EVpred-05"
  retention_target: "return_1d"
  feature_cols: [...]

results:
  # 留存预测
  retention_auc: 0.xxx
  retention_rate_train: 0.xxx
  retention_rate_test: 0.xxx

  # 短期收益（对照）
  revcap_1pct: 0.514

  # 生态指标
  gini_exposure_mean: 0.xxx
  gini_revenue_mean: 0.xxx
  tail_coverage_mean: 0.xxx
```

---

## 6. ⚠️ 检查清单

### 数据处理检查
- [ ] 使用 `prepare_dataset()` 加载数据
- [ ] 新增 `add_retention_labels()` 函数到 data_utils.py
- [ ] 留存标签构建逻辑正确（无未来泄漏）
- [ ] Test 集末尾样本已正确 mask

### 指标评估检查
- [ ] 留存用 AUC（sklearn.metrics.roc_auc_score）
- [ ] 短期用 RevCap@1%（复用 metrics.py）
- [ ] 生态用 Gini（新增函数）
- [ ] 结果保存到 JSON

### 代码检查
- [ ] seed=42 固定随机性
- [ ] 图表文字全英文
- [ ] 保存日志

---

## 7. 📤 报告抄送

完成后需同步更新：

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_EVpred_roadmap.md` | MVP-4.0 状态 + 结论 | §2.1, §4.1 |
| `gift_EVpred_hub.md` | Q2.4 验证状态 + 洞见 | §2 问题树, §5 洞见 |

---

## 8. 🎯 成功标准

| 指标 | 成功阈值 | 失败阈值 |
|------|---------|---------|
| 留存 AUC (return_1d) | > 0.60 | < 0.55 |
| 留存 AUC (return_7d) | > 0.65 | < 0.58 |
| 短期 RevCap@1% | ≥ 51.4% (不退化) | < 46% |

**决策规则**：
- 若 留存 AUC > 0.6 → ✅ 继续多目标联合建模
- 若 留存 AUC 在 0.55-0.6 → 🟡 优化特征后重试
- 若 留存 AUC < 0.55 → ❌ 放弃留存作为预测目标，改为约束条件

---

<!--
📌 Agent 执行规则：

1. ⚠️ 必须使用 gift_EVpred/data_utils.py 加载数据
2. ⚠️ 必须使用 gift_EVpred/metrics.py 评估短期收益
3. ✅ 新增函数应遵循现有代码风格
4. ✅ 留存标签构建需验证无未来泄漏
5. ✅ 先 EDA 留存率分布，再训练模型
6. ✅ 按模板更新 exp.md 报告
-->
