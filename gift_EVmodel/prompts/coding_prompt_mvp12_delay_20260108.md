# 🤖 Coding Prompt: Delayed Feedback Modeling

> **Experiment ID:** `EXP-20260108-gift-allocation-05`  
> **MVP:** MVP-1.2  
> **Date:** 2026-01-08  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：验证延迟反馈建模（Chapelle 软标签/生存分析）是否能改善打赏预测的校准性能

**验证问题**：DG2 - 延迟校正的增益有多大？

**预期结果**：
- 若 ECE 改善 ≥ 0.02 + 近期样本 PR-AUC 提升 ≥ 0.03 → 确认延迟建模，集成到主模型
- 若 ECE 改善 < 0.02 → 简单窗口截断足够，不引入复杂性

**背景**：
- 打赏行为存在延迟反馈——用户进入直播间后一段时间才打赏
- 近期样本若简单当负例，会系统性低估打赏概率
- 需要通过生存分析或软标签加权校正这种偏差

---

## 2. 🧪 实验设定

### 2.1 数据

```yaml
data:
  source: "KuaiLive"
  path: "data/KuaiLive/"
  files:
    - click.csv: user_id, live_id, streamer_id, timestamp, watch_live_time
    - gift.csv: user_id, live_id, streamer_id, timestamp, gift_price
  join_key: [user_id, live_id]
  
  # 延迟相关字段
  click_time: "click.timestamp"
  gift_time: "gift.timestamp"
  watch_duration: "click.watch_live_time"  # 单位: 毫秒
  
  # 计算延迟
  delay: "gift_time - click_time"  # 单位: 毫秒
  censored: "delay > watch_duration OR no gift"
```

**⚠️ 时间字段说明**：
- `timestamp` 是 Unix epoch 毫秒
- `watch_live_time` 是用户观看时长（毫秒）
- 延迟 = gift_time - click_time（仅对有打赏样本）
- 删失 = 用户离开时仍未打赏

### 2.2 延迟分布建模

```yaml
delay_distribution:
  # Step 1: 从历史打赏样本估计延迟分布
  training_data: "only positive samples (gift > 0)"
  target: "delay = gift_time - click_time"
  
  # 候选分布
  candidates:
    - name: "Weibull"
      params: [shape, scale]
    - name: "LogNormal"
      params: [mu, sigma]
    - name: "Exponential"
      params: [rate]
  
  # 选择标准: AIC/BIC 最优
  selection: "AIC"
```

### 2.3 模型

```yaml
models:
  baseline:
    name: "LightGBM Binary (No Correction)"
    description: "直接用观察标签，无延迟校正"
    label: "1 if gift else 0"
    
  chapelle:
    name: "LightGBM Weighted (Chapelle Method)"
    description: "Chapelle 2014 软标签加权"
    label: "soft label with delay correction"
    weight_formula: |
      w_i = 1 if observed_gift
      w_i = 1 - F(H - t_i) if no gift
    # F = delay CDF, H = 观察窗口, t_i = 距窗口结束时间
    
  survival:
    name: "Weibull AFT"
    description: "参数化生存模型"
    method: "lifelines.WeibullAFTFitter"
    target: "time to gift (censored if no gift)"
```

### 2.4 训练配置

```yaml
training:
  # 数据切分
  split:
    method: "temporal"
    train: "前70%时间"
    val: "70%-85%时间"
    test: "85%-100%时间"
  
  # 时间窗口分析
  recency_windows:
    - name: "0-6h"
      hours: [0, 6]
    - name: "6-24h"
      hours: [6, 24]
    - name: "24-48h"
      hours: [24, 48]
    - name: ">48h"
      hours: [48, null]
  
  # LightGBM 参数 (与 MVP-1.1-fair 一致)
  lgb_params:
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 500
    early_stopping: 50
    feature_fraction: 0.8
    bagging_fraction: 0.8
    seed: 42
    objective: "binary"
    metric: "auc"
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | histogram + fitted curve | Delay (minutes) | Density | `../img/delay_distribution.png` |
| Fig2 | grouped bar | Time Window | ECE | `../img/delay_ece_by_window.png` |
| Fig3 | line (3 models) | Predicted Prob | Actual Freq | `../img/delay_calibration_comparison.png` |
| Fig4 | grouped bar | Recency Window | PR-AUC | `../img/delay_prauc_by_recency.png` |
| Fig5 | bar | Model | ECE / Brier | `../img/delay_model_comparison.png` |

**图表要求**：
- 所有文字必须英文
- 包含 legend、title、axis labels
- 分辨率 ≥ 300 dpi
- **figsize 规则**：
  - 单张图：`figsize=(6, 5)`
  - 多张图（subplot）：按 6:5 比例扩增

---

## 4. 📁 参考代码

> ⚠️ **不要在这里写代码！先读取以下文件，理解后再修改**

| 参考脚本 | 可复用 | 需修改/新增 |
|---------|--------|------------|
| `scripts/train_fair_comparison.py` | `load_data()`, `prepare_features()`, `temporal_split()`, `get_feature_columns()`, `calculate_ece()`, plot 函数, 评估逻辑 | 新增延迟计算, 软标签逻辑, 生存模型 |
| `scripts/train_two_stage.py` | 特征工程逻辑, LightGBM 训练函数 | 适配加权训练 |
| `scripts/eda_kuailive.py` | 数据加载和探索逻辑 | 参考时间字段处理 |

### 关键实现步骤

1. **延迟分布估计**：
   - 从 gift 样本中计算 delay = gift_time - click_time
   - 用 scipy.stats 拟合 Weibull/LogNormal，选 AIC 最优
   - 可视化延迟分布 (Fig1)

2. **Chapelle 软标签计算**：
   ```
   # 伪代码
   for each sample i:
       if has_gift:
           w_i = 1.0
       else:
           time_since_click = observation_end - click_time
           w_i = 1 - F_delay(time_since_click)  # F = fitted CDF
   ```

3. **生存模型**：
   - 使用 `lifelines` 库
   - target: delay (if gift) or watch_duration (if no gift, censored)
   - event: 1 if gift else 0

4. **分窗口评估**：
   - 按样本距观察窗口结束时间分组
   - 分别计算 ECE, PR-AUC
   - 检验近期样本是否被系统性低估

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `exp/exp_delay_modeling_20260108.md`
- **模板**: 已存在草稿，补全 §4-§6
- **必须包含**:
  - ⚡ 核心结论速览（一句话 + 关键数字对比表）
  - 📊 实验图表（所有 5 张图 + 观察）
  - 📝 结论（DG2 验证结果 + 设计决策）

### 5.2 图表文件
- **路径**: `../img/`
- **命名**: `delay_*.png`

### 5.3 数值结果
- **格式**: JSON
- **路径**: `../results/delay_modeling_20260108.json`
- **内容**:
```yaml
required_fields:
  - experiment_id: "EXP-20260108-gift-allocation-05"
  - timestamp
  - delay_distribution:
      type: "Weibull/LogNormal"
      params: {shape, scale} or {mu, sigma}
      aic: float
  - models:
      baseline: {ece, brier, pr_auc, pr_auc_by_window}
      chapelle: {ece, brier, pr_auc, pr_auc_by_window}
      survival: {ece, brier, pr_auc, pr_auc_by_window}
  - comparison:
      ece_improvement: float  # chapelle - baseline
      prauc_recent_improvement: float  # 近期样本
      conclusion: "accept/reject"
      decision: "adopt delay correction / simple truncation"
```

---

## 6. 📤 报告抄送

完成后需同步更新：

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `../gift_EVmodel_roadmap.md` | MVP-1.2 状态 → ✅; 结论快照 | §2.1, §4.3 |
| `../gift_EVmodel_hub.md` | DG2 验证状态; 更新 P4 原则 | §5 DG2, §6 |

---

## 7. ⚠️ 注意事项

- [ ] 代码中固定 `seed=42`
- [ ] 时间戳单位统一（毫秒 → 分钟/秒，按需转换）
- [ ] 安装依赖：`pip install lifelines` (生存分析库)
- [ ] 图表文字全英文
- [ ] 保存完整日志到 `logs/delay_modeling_20260108.log`
- [ ] 使用 nohup 后台运行：`nohup python scripts/train_delay_modeling.py > logs/delay_modeling_20260108.log 2>&1 &`

---

## 8. 📐 验收标准

| 指标 | 门槛 | 决策 |
|------|------|------|
| ECE 改善 | ≥ 0.02 | → 采用延迟校正 |
| 近期样本 PR-AUC 提升 | ≥ 0.03 | → 确认近期偏差存在 |
| 整体 PR-AUC | ≥ baseline | → 校正不损害整体性能 |

**决策树**：
```
IF ECE改善≥0.02 AND 近期PR-AUC提升≥0.03:
    → 采用 Chapelle/Survival 中更优者
    → 集成到主模型 (MVP-1.3)
ELSE:
    → 简单窗口截断足够
    → 关闭延迟校正方向
```

---

## 9. 🔧 依赖安装

```bash
# 确保环境初始化
source init.sh

# 安装生存分析库（如果没有）
pip install lifelines
```

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先读取"参考代码"中列出的文件
3. ✅ 理解现有代码逻辑后再修改
4. ✅ 复用已有函数，不要重复造轮子
5. ✅ 按模板输出 exp.md 报告
6. ✅ 注意时间单位转换（毫秒 → 分钟）
7. ✅ 生存分析使用 lifelines 库
-->
