# 🤖 Coding Prompt: 指标体系落地

> **Experiment ID:** `EXP-20260119-EVpred-01`  
> **MVP:** MVP-3.1  
> **Date:** 2026-01-19  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：将三层指标体系从设计文档落实到可执行的统一评估流程，包括 RevCap 曲线、Tail Calibration 和稳定性评估

**验证假设**：
- 统一报告模板是否覆盖识别层/估值层/分配层三层指标？
- RevCap 曲线（K ∈ {0.01%, 0.1%, 0.5%, 1%, 2%, 5%, 10%}）是否可解释业务价值？
- 按天 RevCap@1% 稳定性是否可接受（标准差 < 5%）？

**预期结果**：
- 若 统一模板可落地 → 所有后续实验使用
- 若 RevCap 曲线稳定（CV < 10%）→ 作为标准评估工具
- 若 稳定性不达标 → 需要进一步分析原因

---

## 2. 🧪 实验设定

### 2.1 数据

```yaml
data:
  source: "KuaiLive (Day-Frozen)"
  path: "data/KuaiLive/"
  train_size: 1629415
  val_size: 1717199
  test_size: 1409533
  split: "7-7-7 by days"
  features: 31
  label_window: "1 hour"
  protocol: "Day-Frozen (day < click_day)"
```

**⚠️ 强制要求**：
- 必须使用 `gift_EVpred/data_utils.py` 的 `prepare_dataset()` 加载数据
- 禁止自行实现数据处理逻辑
- 禁止使用 `watch_live_time` 特征

### 2.2 模型

```yaml
model:
  name: "Ridge Regression"
  params:
    alpha: 1.0
    solver: "auto"
    random_state: 42
  target: "raw Y (target_raw)"
  scaling: "StandardScaler (fit on train, transform val/test)"
```

**统一口径**：
- 所有实验使用相同的 Ridge Regression 模型
- 预测目标统一为 `target_raw`（原始金额，非 log(1+Y)）
- 特征标准化：用 train 的均值和标准差 transform val/test

### 2.3 训练

```yaml
training:
  model: "Ridge(alpha=1.0)"
  target: "target_raw"
  scaling: "StandardScaler"
  seed: 42
  validation: "Use val set for hyperparameter selection (if needed)"
  test: "Only for final report, no hyperparameter tuning"
```

### 2.4 扫描参数

```yaml
sweep:
  N/A: "No hyperparameter sweep, use fixed alpha=1.0"
  fixed:
    model: "Ridge"
    alpha: 1.0
    target: "target_raw"
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | line | K (0.01%, 0.1%, 0.5%, 1%, 2%, 5%, 10%) | RevCap@K | `gift_EVpred/img/revcap_curve.png` |
| Fig2 | bar | Day (1-7) | RevCap@1% | `gift_EVpred/img/revcap_stability.png` |
| Fig3 | heatmap | Bucket (Top 0.1%, 0.5%, 1%, 5%) | Calibration Ratio | `gift_EVpred/img/tail_calibration.png` |

**图表要求**：
- 所有文字必须英文
- 包含 legend、title、axis labels
- 分辨率 ≥ 300 dpi
- **figsize 规则（必须遵守）**：
  - 单张图：`figsize=(6, 5)` 锁死
  - 多张图（subplot）：按 6:5 比例扩增，如 `(12, 5)` for 1×2, `(12, 10)` for 2×2

**Fig1: RevCap 曲线**
- 对比线：Model (Ridge + raw Y)、Oracle（按真实 y 排序）、Random
- X 轴：K ∈ {0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10}
- Y 轴：RevCap@K
- Legend：Model, Oracle, Random

**Fig2: 按天稳定性**
- X 轴：Test 的 7 天（day 1-7）
- Y 轴：RevCap@1% for each day
- 添加：均值线、±1 标准差区间
- 标题：显示均值、标准差、CV

**Fig3: Tail Calibration 热力图**
- X 轴：Bucket (Top 0.1%, 0.5%, 1%, 5%)
- Y 轴：Calibration Type (Sum, Mean)
- 颜色：Calibration Ratio (Sum(pred)/Sum(actual), Mean(pred)/Mean(actual))
- 目标：接近 1.0（绿色），偏离 1.0 用红色

---

## 4. 📁 参考代码

> ⚠️ **不要在这里写代码！只写路径，让 Agent 自己读取**

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `gift_EVpred/data_utils.py` | `prepare_dataset()`, `get_feature_columns()` | 无需修改 |
| `gift_EVpred/exp/exp_raw_vs_log_20260118.md` § 7.2 | Ridge 训练代码示例 | 参考训练流程 |
| `gift_EVpred/exp/exp_lightgbm_raw_y_20260118.md` § 7.2 | `revenue_capture_at_k()` 函数 | 参考 RevCap 计算 |
| `gift_EVpred/exp/exp_metrics_framework_20260118.md` | 三层指标定义 | 参考指标计算逻辑 |

**关键函数参考**：
- RevCap 计算：参考 `exp_lightgbm_raw_y_20260118.md` 中的 `revenue_capture_at_k()` 函数
- 数据加载：使用 `data_utils.py` 的 `prepare_dataset()` 和 `get_feature_columns()`
- 模型训练：Ridge + StandardScaler，参考 `exp_raw_vs_log_20260118.md` 的训练代码

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `gift_EVpred/exp/exp_metrics_landing_20260119.md`
- **模板**: `_backend/template/exp.md`
- **必须包含**:
  - ⚡ 核心结论速览（一句话 + 关键数字）
  - 📊 实验图表（Fig1-3 + 观察）
  - 📝 结论（假设验证 + 设计启示）
  - 📊 统一报告模板（Markdown 表格格式）

### 5.2 图表文件
- **路径**: `gift_EVpred/img/`
- **命名**: 
  - `revcap_curve.png`
  - `revcap_stability.png`
  - `tail_calibration.png`

### 5.3 数值结果
- **格式**: JSON
- **路径**: `gift_EVpred/results/metrics_landing_20260119.json`
- **必须包含**:
  - RevCap@K for K ∈ {0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10}
  - RevCap@1% by day (7 days)
  - Tail Calibration (Sum/Mean) for top buckets
  - Normalized RevCap@K (RevCap@K / Oracle@K)

### 5.4 统一报告模板
- **路径**: `gift_EVpred/exp/exp_metrics_landing_20260119.md` § 统一报告模板
- **格式**: Markdown 表格
- **内容**: 识别层/估值层/稳定性三层指标表格

---

## 6. 📤 报告抄送

完成后需同步更新：

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_EVpred_roadmap.md` | MVP-3.1 状态 + 结论快照 | §2.1, §4.1 |
| `gift_EVpred_hub.md` | 指标体系落地状态 | §10 |

---

## 7. ⚠️ 注意事项

- [ ] 代码中添加 seed=42 固定随机性
- [ ] 图表文字全英文
- [ ] 保存完整日志到 `logs/`
- [ ] 长时间任务使用 nohup 后台运行
- [ ] **必须使用 `data_utils.py` 加载数据，禁止自行实现**
- [ ] **禁止使用 `watch_live_time` 特征**
- [ ] **统一口径：Ridge Regression + raw Y + StandardScaler**
- [ ] **Test 只做一次最终报告，不参与超参选择**

---

## 8. 🔧 评估指标计算

### 8.1 RevCap@K

```python
# 参考实现（不要直接复制，先读取 exp_lightgbm_raw_y_20260118.md）
def revenue_capture_at_k(y_true, y_pred, k=0.01):
    n_top = int(len(y_true) * k)
    top_indices = np.argsort(y_pred)[-n_top:]
    return y_true[top_indices].sum() / y_true.sum()
```

**计算范围**：K ∈ {0.001, 0.005, 0.01, 0.02, 0.05, 0.10}

### 8.2 Normalized RevCap@K

```python
# Oracle = 按真实 y 排序的理论上限
oracle_revcap_k = revenue_capture_at_k(y_true, y_true, k)
normalized_revcap_k = revcap_k / oracle_revcap_k
```

### 8.3 Tail Calibration

```python
# 按预测分数分桶
buckets = [0.001, 0.005, 0.01, 0.05]
for k in buckets:
    top_k_idx = np.argsort(y_pred)[-int(len(y_pred)*k):]
    sum_calibration = y_pred[top_k_idx].sum() / y_true[top_k_idx].sum()
    mean_calibration = y_pred[top_k_idx].mean() / y_true[top_k_idx].mean()
```

### 8.4 稳定性评估

```python
# 按天计算 RevCap@1%
test_df['day'] = pd.to_datetime(test_df['timestamp']).dt.date
revcap_by_day = []
for day in test_df['day'].unique():
    day_mask = test_df['day'] == day
    revcap_day = revenue_capture_at_k(
        y_true[day_mask], y_pred[day_mask], k=0.01
    )
    revcap_by_day.append(revcap_day)

mean_revcap = np.mean(revcap_by_day)
std_revcap = np.std(revcap_by_day)
cv = std_revcap / mean_revcap
```

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先读取"参考代码"中列出的文件
3. ✅ 理解现有代码逻辑后再修改
4. ✅ 复用已有函数，不要重复造轮子
5. ✅ 按模板输出 exp.md 报告
6. ✅ **必须使用 data_utils.py 加载数据**
7. ✅ **统一口径：Ridge Regression + raw Y + StandardScaler**
-->
