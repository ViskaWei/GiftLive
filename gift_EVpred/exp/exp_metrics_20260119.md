<!--
📝 Agent 书写规范（不出现在正文）:
- Header 全英文
- 正文中文
- 图表文字全英文（中文会乱码）
- 公式用 LaTeX: $inline$ 或 $$block$$

🎯 本模板的目标：
- 前 30 行：读者不仅知道"结论是什么"，还要知道"这是在做什么 & 怎么跑出来的"。
- 正文：读者可以按『实验流程伪代码』复现/重跑。
-->

# 🍃 Metrics v2.2 指标体系完整手册
> **Name:** Metrics v2.2 Complete Reference (口径修订版)  \
> **ID:** `EVpred-20260119-metrics-v22`  \
> **Topic:** `gift_EVpred` | **MVP:** MVP-3.2 | **Project:** `VIT`  \
> **Author:** Viska Wei | **Date:** 2026-01-19 | **Status:** ✅ Completed
>
> 🎯 **Target:** 定义 gift_EVpred 的统一评估指标体系，为所有实验提供标准化评估  \
> 🚀 **Decision / Next:** 所有实验必须使用 `metrics.py` v2.2 进行评估

---

## ⚡ 核心结论速览（供 main 提取；≤30行；必含 I/O + Pipeline TL;DR）

> **一句话**: 7 层指标体系，**共 38 个指标**，从选模型到上线监控全覆盖；使用 `evaluate_model()` 一键评估

### 0.1 这实验到底在做什么？（X := 算法/机制 → 目标 | Why+How | I/O | Trade-off）

$$
X := \underbrace{\text{Metrics Framework v2.2}}_{\text{是什么}}\ \xrightarrow[\text{基于}]{\ \text{7 层分级 + 多粒度}\ }\ \underbrace{\text{标准化模型评估}}_{\text{用于}}\ \big|\ \underbrace{\text{Why: 需统一口径}}_{\text{痛点}} + \underbrace{\text{How: 多层覆盖}}_{\text{难点}}
$$

- **🐻 What (是什么)**: 7 层 38 个指标的完整评估框架，覆盖模型选择到上线监控
- **🍎 核心机制**: L1 主指标 → L2 诊断 → L3 稳定性 → L4 校准 → L5 切片 → L6 生态 → L7 决策核心
- **⭐ 目标**: 为 gift_EVpred 所有实验提供统一评估标准
- **🩸 Why（痛点）**: 之前实验指标不统一，口径混乱，难以对比
- **💧 How（难点）**: 需覆盖 Sample/User/Streamer 多粒度，金额/比例双版本

$$
\underbrace{\text{I/O 🫐}}_{\text{输入→输出}}\ =\ \underbrace{\Delta^+: \text{一键评估 38+ 指标}}_{\text{优势}}\ -\ \underbrace{\Delta^-: \text{部分指标需额外输入}}_{\text{约束}}
$$

**I/O（必须写清楚，读者靠这一段理解实验"在干嘛"）**

| 类型 | 符号 | 说明 | 示例 |
|------|------|------|------|
| 🫐 输入 | $y_{true}$ | 真实标签（打赏金额） | ndarray (N,) |
| 🫐 输入 | $y_{pred}$ | 模型预测值 | ndarray (N,) |
| 🫐 输入 | $\mathcal{D}_{test}$ | 测试集 DataFrame | 含 user_id, streamer_id 等列 |
| 🫐 输出 | $M$ | MetricsResult 对象 | 38+ 指标, .summary(), .to_json() |
| 📊 指标 | $L1$-$L7$ | 7 层指标体系 | RevCap, Whale, PSI, EffN... |
| 🍁 基线 | $v2.1$ | 旧版指标体系 | 缺金额版 Regret, nAUC 不规范 |

### 0.2 Pipeline TL;DR（5-10 行极简伪代码，一眼看懂在跑什么）

```
1. 准备输入：
   - y_true = [0, 0, 52.0, 0, 520, ...] (N=1.4M, 打赏金额)
   - y_pred = [0.1, 0.2, 48.3, 0.0, 480.5, ...] (N=1.4M, 预测值)
   - test_df = DataFrame(user_id, streamer_id, ...)
2. 调用评估：
   result = evaluate_model(y_true, y_pred, test_df, compute_decision_metrics=True)
   → 内部循环: 对每个 K ∈ [0.1%, 0.5%, 1%, 2%, 5%, 10%] 计算 RevCap/Whale/...
3. 评估输出: result = {
       'RevCap@1%': 0.514, 'WRecLift@1%': 34.6×, 'PSI': 0.155,
       'slices': {'cold_start': {...}, 'whale_true': {...}},
       ... (共 38 个指标)
   }
4. 落盘：result.to_json('results/metrics_v21_baseline.json')
```

> ⚠️ **复现命令**（repo/entry/config/seed）→ 见 §7.2 附录
> 📖 **详细伪代码**（对齐真实代码）→ 见 §2.4.2

### 0.3 对假设/验证问题的回答

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| Q: 指标体系是否覆盖全场景？ | ✅ 7 层 38 指标 | 覆盖选模型→上线→监控 |
| Q: 是否支持多粒度？ | ✅ Sample/User/Streamer | 三粒度全覆盖 |
| Q: 口径是否统一？ | ✅ v2.2 规范 | 金额/比例双版本 |

### 0.4 关键数字（只放最重要的 3-5 个）

| Metric | Value | vs Baseline | Notes |
|--------|-------|------------|------|
| 指标总数 | **38** | v2.1: ~30 | +决策核心指标 |
| 覆盖层级 | **7 层** | L1-L7 | 全链路覆盖 |
| 评估时间 | **~30s** | 1.4M samples | CPU 单核 |
| 覆盖率要求 | **≥90%** | 32/35 必须 | 否则不得合入 |

### 0.5 Links

| Type | Link |
|------|------|
| 📊 Code | `gift_EVpred/metrics.py` v2.2 |
| 🧠 Hub | `gift_EVpred/gift_EVpred_hub.md` |
| 📖 Usage | `gift_EVpred/examples/metrics_v2_usage.py` |

---

# 1. 🎯 目标

**核心问题**: 如何为 gift_EVpred 建立统一、全面、可复用的评估指标体系？

**对应 main / roadmap**:
- 验证问题：Q10.5 指标体系是否完整？
- 子假设：H10.5.1 需要 7 层分级覆盖
- Gate（如有）：Gate-3.2 指标体系闭环

## 1.1 成功标准（验收 / stop rule）

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| ✅ 通过 | 覆盖率 ≥ 90%，口径统一 | 可作为标准 |
| ❌ 否决 | 关键指标缺失或口径冲突 | 需重新设计 |
| ⚠️ 异常 | 评估时间 > 5min | 需优化性能 |

---

# 2. 🦾 方法（算法 + I/O + 实验流程）

> 📌 **本章至少要填 2.2 I/O 与 2.4 实验流程**；否则读者无法知道实验怎么做的。

## 2.1 算法（7 层指标体系）

### L1 主指标（选模型/调参）

**RevCap@K (Revenue Capture)**：

$$
\text{RevCap@K} = \frac{\sum_{i \in \text{Top}_K(\hat{y})} y_i}{\sum_{i=1}^{N} y_i}
$$

### L7 决策核心指标

**Misallocation Cost（Regret）**：

$$
\text{Regret@K} = \text{OracleRev@K} - \text{AchievedRev@K} \quad \text{（元）}
$$

$$
\text{nRevCap@K} = \frac{\text{AchievedRev@K}}{\text{OracleRev@K}} \in [0, 1]
$$

**Capture AUC（推荐使用 nAUC）**：

$$
\text{nAUC@K} = \frac{\text{CapAUC@K}}{\text{OracleAUC@K}} \in [0, 1]
$$

**PSI (Population Stability Index)**：

$$
\text{PSI} = \sum_{i=1}^{B} (p_{\text{test},i} - p_{\text{train},i}) \cdot \ln\left(\frac{p_{\text{test},i}}{p_{\text{train},i}}\right)
$$

**Diversity (Effective Number)**：

$$
\text{EffN@K} = e^{H}, \quad H = -\sum_{g} p_g \ln(p_g)
$$

## 2.2 输入 / 输出（必填：比 0.1 更细一点）

> 目标：读者不看代码，也能知道"喂进去什么 → 吐出来什么"。

### I/O Schema

| Component | Type/Shape | Example | Notes |
|----------|------------|---------|------|
| y_true | ndarray (N,) | 原始打赏金额 | 必须 |
| y_pred | ndarray (N,) | 预测分数/金额 | 必须 |
| test_df | DataFrame | 含 user_id, streamer_id | 必须（切片/生态用） |
| y_pred_train | ndarray | 训练集预测 | 可选（PSI 用） |
| whale_threshold | float | 100 | 默认 P90(y\|y>0) |
| Output: MetricsResult | object | 35+ 指标 | .summary(), .to_json() |

### Assumptions & Constraints

| Assumption/Constraint | Why it matters | How handled |
|----------------------|----------------|------------|
| y_pred 可不校准 | 只用于排序 | 金额校准指标需 EV 刻度 |
| Whale 阈值 | 影响 L2 指标 | 支持固定/分位双阈值 |
| 需 test_df 列 | 切片/生态分析 | 明确列名映射 |

## 2.3 实现要点（读者能对照代码定位）

| What | Where (file:function) | Key detail |
|------|------------------------|-----------|
| 主入口 | `metrics.py:evaluate_model` | 一键评估 |
| 主指标 | `metrics.py:revenue_capture_at_k` | L1 |
| Whale 七件套 | `metrics.py:whale_recall_at_k` 等 | L2 |
| 稳定性 | `metrics.py:compute_stability_summary` | L3 |
| 切片 | `metrics.py:compute_slice_metrics` | L5 |
| 生态 | `metrics.py:compute_ecosystem_metrics` | L6 |
| 决策核心 | `metrics.py:compute_misallocation_cost` 等 | L7 |

## 2.4 实验流程（必填：模块拆解 + 核心循环展开 + Code Pointer）

### 2.4.1 实验流程树状图（完整可视化）

> ⚠️ **每一步都带 I/O 数值例子**，让读者秒懂每层指标输出什么

```
evaluate_model() 内部流程
│
├── 输入
│   ├── y_true = [0, 0, 52.0, 0, 520, ...] (N=1.4M)
│   ├── y_pred = [0.1, 0.2, 48.3, 0.0, 480.5, ...] (N=1.4M)
│   └── test_df = DataFrame(user_id, streamer_id, date, ...)
│
├── L1. 主指标 ⭐（必须）
│   ├── RevCap@K：Top K% 捕获收入比例 → RevCap@1% = 0.514
│   ├── Oracle@K：理论上限 → Oracle@1% = 0.996
│   └── nRevCap@K：归一化版本 → nRevCap@1% = 0.516
│
├── L2. 诊断（必须）
│   ├── Whale 七件套 → {WRec: 0.12, WRecLift: 34.6×, WPrec: 0.73, ...}
│   └── 阈值：T=100 (P90)
│
├── L3. 稳定性（必须）
│   ├── CV：按天变异系数 → CV = 0.08
│   └── Bootstrap CI → [0.48, 0.55] (95% CI)
│
├── L4. 校准（可选）
│   ├── ECE：期望校准误差 → ECE = 0.12
│   └── Tail Ratio → TailRatio = 0.85
│
├── L5. 切片（需 compute_slices=True）
│   ├── Cold Start Pair → {RevCap: 0.32, Lift: 12×}
│   ├── Whale True → {RevCap: 0.68, Lift: 42×}
│   └── Tier 分层 (6 个) → tier_1: {...}, tier_2: {...}, ...
│
├── L6. 生态（需 compute_ecosystem=True）
│   ├── Gini → Gini = 0.89
│   ├── Coverage → Coverage = 0.65
│   └── Overload → Overload = 0.12
│
└── L7. 决策核心 ⭐（需 compute_decision_metrics=True）
    ├── Regret：错分代价 → Regret@1% = 45,230 元
    ├── nAUC：归一化 Capture AUC → nAUC@10% = 0.612
    ├── PSI：分布漂移 → PSI = 0.155
    └── EffN：有效分组数 → EffN@1% = 234

输出: MetricsResult 对象
    → result.summary() = {'RevCap@1%': 0.514, 'WRecLift@1%': 34.6, 'PSI': 0.155, ...}
    → result.to_json('path') 保存 38+ 指标
```

### 2.4.2 模块拆解（7 层对应 7 个模块，带 Code Pointer）

| Module | Responsibility | Input → Output | Code Pointer |
|--------|----------------|----------------|--------------|
| L1: primary | 主指标 RevCap/Oracle | y_true, y_pred → revcap dict | `revenue_capture_at_k()` |
| L2: diagnostic | Whale 七件套 | y_true, y_pred, threshold → whale dict | `whale_recall_at_k()` 等 |
| L3: stability | CV, Bootstrap CI | y_true, y_pred, test_df → stability dict | `compute_stability_summary()` |
| L4: calibration | ECE, Tail Ratio | y_true, y_pred → calibration dict | `tail_calibration()` |
| L5: slices | 10 个切片 | y_true, y_pred, test_df → slice dict | `compute_slice_metrics()` |
| L6: ecosystem | Gini, Coverage, Overload | y_true, y_pred, test_df → eco dict | `compute_ecosystem_metrics()` |
| L7: decision | Regret, nAUC, PSI, EffN | y_true, y_pred, y_pred_train → decision dict | `compute_misallocation_cost()` 等 |

### 2.4.3 核心循环展开（evaluate_model 内部）

```python
# === evaluate_model 内部流程 ===

def evaluate_model(y_true, y_pred, test_df, **kwargs):
    result = MetricsResult()
    
    # L1: 主指标（必须）
    for k in K_VALUES:
        result.revcap[k] = revenue_capture_at_k(y_true, y_pred, k)
        result.oracle[k] = revenue_capture_at_k(y_true, y_true, k)
    
    # L2: 诊断（必须）
    whale_mask = y_true >= whale_threshold
    result.whale = compute_whale_metrics(y_true, y_pred, k, whale_mask)
    
    # L3: 稳定性（必须）
    if 'date' in test_df.columns:
        result.stability = compute_stability_by_day(...)
    result.bootstrap_ci = bootstrap_confidence_interval(...)
    
    # L4: 校准（如果 y_pred 是 EV 刻度）
    result.tail_calibration = tail_calibration(y_true, y_pred, k)
    
    # L5: 切片（需 compute_slices=True）
    if compute_slices:
        for slice_name, slice_mask in get_slices(test_df, slice_config):
            result.slices[slice_name] = compute_metrics(y_true[mask], y_pred[mask])
    
    # L6: 生态（需 compute_ecosystem=True）
    if compute_ecosystem:
        result.ecosystem = compute_ecosystem_metrics(test_df, y_pred, y_true)
    
    # L7: 决策核心（需 compute_decision_metrics=True）
    if compute_decision_metrics:
        result.misallocation_cost = compute_misallocation_cost(y_true, y_pred, k)
        result.capture_auc = compute_capture_auc(y_true, y_pred)
        if y_pred_train is not None:
            result.drift = compute_drift(y_pred_train, y_pred)
        result.diversity = compute_diversity(test_df, y_pred, k, y_true)
    
    return result
```

### 2.4.4 复现清单

- [x] 指标函数单元测试：`tests/test_metrics_upgrade.py`
- [x] 示例用法：`gift_EVpred/examples/metrics_v2_usage.py`
- [x] 版本号：v2.2（口径修订版）

---

# 3. 🧪 实验设计（指标规格）

## 3.1 指标总数统计

| 层级 | 用途 | 指标数量 | 核心指标 |
|------|------|----------|---------|
| 1. 主指标 | 选模型/调参 | **3** | RevCap@K, Oracle@K, nRevCap@K |
| 2. 诊断指标 | 理解模型行为 | **6** | WRec/WRecLift/WRevCap/WPrec, Avg Revenue, Gift Rate |
| 3. 稳定性指标 | 评估鲁棒性 | **5** | CV, Bootstrap CI, Min/Max/Mean |
| 4. 校准指标 | 预测准确度 | **3** | ECE (概率), Sum Ratio, Mean Ratio (金额) |
| 5. 切片指标 | 公平性分析 | **10** | Cold Start×2, Whale×2, Tier×6 |
| 6. 生态指标 | 分配层护栏 | **7** | Gini, Coverage×3, Overload×2, Top10 Share |
| 7. 决策核心 | 业务决策支持 | **4类/15+** | Regret, nAUC, PSI, EffN |
| **合计** | — | **38+** | — |

## 3.2 v2.2 新增/修订

| 变更 | v2.1 | v2.2 |
|------|------|------|
| Regret 命名 | opportunity_cost | regret（工业界通用） |
| 金额/比例 | 只有比例版 | 同时提供金额版（元）和比例版 |
| nAUC 规范 | 原始 CapAUC 范围不清 | 明确 nAUC ∈ [0,1] |
| Overload 拆分 | 单一 overload_rate | U-OverTarget + S-OverLoad |
| 工业界短名 | 无 | 完整短名映射表（§9.2） |

## 3.3 覆盖率要求

| 要求 | 最低标准 | 备注 |
|------|----------|------|
| L1-L4 基础指标 | 必须 100% | 无条件 |
| L5 切片指标 | ≥ 8/10 | 冷启动切片必须 |
| L6 生态指标 | ≥ 4/7 | Gini + Coverage 必须 |
| L7 决策指标 | 必须 4/4 | 全部必须 |
| **总覆盖率** | **≥ 90%** | 约 32/35 |

> **不满足覆盖率的实验报告不得合入 roadmap**

---

# 4. 📊 图表 & 结果

> ⚠️ 图表文字必须全英文！

## 4.1 指标速查表

| 指标名 | 简单表述 | 用途 |
|--------|----------|------|
| RevCap@K | Top K% 预测捕获的真实收入比例 | 主指标 |
| WRecLift@K | WRec@K / K，相对随机提升倍数 | 诊断 ⭐ |
| WRevCap@K | Top K% 捕获的 whale 收入比例 | 诊断 ⭐ |
| nRevCap@K | AchievedRev / OracleRev | 决策核心 |
| Regret@K | OracleRev - AchievedRev（元） | 决策核心 |
| nAUC@K | 归一化 Capture AUC | 决策核心 |
| PSI | 分布漂移程度 | 决策核心 |
| EffN@K | 有效分组数 | 决策核心 |

**What it shows**: 核心指标快速参考

**Key observations**:
- 主指标用于选模型，决策指标用于上线判断
- 诊断指标帮助理解模型行为
- 推荐使用 Lift 版本而非绝对值版本

## 4.2 阈值判断表

| 指标 | 阈值 | 判断 |
|------|------|------|
| RevCap@1% | > 40% | 好 |
| WRecLift@K | > 20× | 强 |
| PSI | < 0.25 | 可上线 |
| CV@1% | < 10% | 稳定 |

---

# 5. 💡 洞见（解释"为什么会这样"）

## 5.1 机制层（Mechanism)
- **7 层分级**：从模型选择（L1）到上线监控（L7），覆盖 ML 全生命周期
- **多粒度**：Sample（事件筛选）、User（人群圈选）、Streamer（生态监控）

## 5.2 实验层（Diagnostics)
- **Lift vs 绝对值**：使用 WRecLift 而非 WRec，消除 K 值影响
- **金额 vs 比例**：金额版用于业务沟通，比例版用于跨实验对比

## 5.3 设计层（So what)
- **强制覆盖率**：≥90% 指标覆盖率，保证评估完整性
- **标准化接口**：`evaluate_model()` 一键评估，降低使用门槛

---

# 6. 📝 结论 & 下一步

## 6.1 核心发现（punch line）
> **7 层 38 指标体系完成，v2.2 口径规范化，可作为 gift_EVpred 评估标准**

- ✅ Q10.5: 指标体系完整
- **Decision**: 所有实验必须使用 v2.2 评估

## 6.2 关键结论（2-5 条）

| # | 结论 | 证据（图/表/数字） | 适用范围 |
|---|------|-------------------|---------|
| 1 | **7 层覆盖** | L1-L7 全链路 | 全部实验 |
| 2 | **38+ 指标** | 见 §9.5 完整清单 | 全部实验 |
| 3 | **一键评估** | evaluate_model() | 使用简单 |
| 4 | **口径统一** | v2.2 规范 | 可比性强 |

## 6.3 Trade-offs（Δ+ vs Δ-）

| Upside (Δ+) | Cost / Constraint (Δ-) | When acceptable |
|-------------|--------------------------|----------------|
| 全面覆盖 | 评估时间 ~30s | 可接受 |
| 标准化 | 需要额外输入（PSI 需 y_pred_train） | 按需开启 |

## 6.4 下一步（可执行任务）

| Priority | Task | Owner | Link |
|----------|------|-------|------|
| 🔴 P0 | 所有新实验使用 v2.2 | All | - |
| 🟡 P1 | 补充 DW-ECE, Fairness Gap | Metrics | v2.3 |
| 🟢 P2 | 性能优化（大数据量） | Infra | - |

---

# 7. 📎 附录（复现/审计用）

## 7.1 数值结果（Baseline 参考值）

| 指标 | Baseline 值 | 条件 |
|------|-------------|------|
| RevCap@1% | 51.4% | Ridge α=1.0, 20 特征 |
| nRevCap@1% | 51.6% | vs Oracle |
| WRecLift@1% | 34.6× | T=100 |
| WRevCap@1% | 55.5% | 金额口径 |
| PSI | 0.155 | Train vs Test |
| EffN@1% | 181 | Top 1% 内 |
| CV@1% | 9.4% | 按天稳定性 |

## 7.2 执行记录（复现命令）

| Item | Value |
|------|-------|
| Repo | `~/GiftLive` |
| Script | `gift_EVpred/examples/metrics_v2_usage.py` |
| Config | v2.2 完整模式 |
| Output | MetricsResult 对象 |

```bash
# (1) setup
cd ~/GiftLive
source init.sh

# (2) 使用示例
python -c "
from gift_EVpred.metrics import evaluate_model

# 完整评估
result = evaluate_model(
    y_true=y_test,
    y_pred=y_pred_test,
    test_df=test_df,
    whale_threshold=100,
    y_pred_train=y_pred_train,
    compute_stability=True,
    compute_slices=True,
    compute_ecosystem=True,
    compute_decision_metrics=True,
)

print(result.summary())
result.to_json('results/metrics_result.json')
"

# (3) 单指标计算
python -c "
from gift_EVpred.metrics import revenue_capture_at_k, compute_misallocation_cost

revcap = revenue_capture_at_k(y_true, y_pred, k=0.01)
cost = compute_misallocation_cost(y_true, y_pred, k=0.01)
print(f'RevCap@1%: {revcap:.1%}')
print(f'Regret: {cost[\"regret\"]:,.0f} 元')
"
```

## 7.3 指标清单（完整版）

| # | 指标名 | 层级 | 函数 | 必需 |
|---|--------|------|------|------|
| 1 | RevCap@K | L1 | `revenue_capture_at_k()` | ✅ |
| 2 | Oracle@K | L1 | `compute_oracle_revcap_curve()` | ✅ |
| 3 | nRevCap@K | L1 | `compute_misallocation_cost()` | ✅ |
| 4-9 | Whale 七件套 | L2 | `whale_recall_at_k()` 等 | ✅ |
| 10-14 | 稳定性 5 项 | L3 | `compute_stability_summary()` | ✅ |
| 15-17 | 校准 3 项 | L4 | `tail_calibration()` | ⚠️ |
| 18-27 | 切片 10 项 | L5 | `compute_slice_metrics()` | ✅ |
| 28-34 | 生态 7 项 | L6 | `compute_ecosystem_metrics()` | ✅ |
| 35-38 | 决策核心 4 类 | L7 | 各 compute 函数 | ✅ |

## 7.4 术语对照表（v2.2 短名）

| 短名 | 全称 | 说明 |
|------|------|------|
| `RevCap@K` | Revenue Capture | 主指标 |
| `nRevCap@K` | Normalized RevCap | 归一化版本 |
| `Regret@K` | Regret (Amount) | 金额版（元） |
| `RegretPct@K` | Regret Percentage | 比例版 |
| `nAUC@K` | Normalized AUC | 推荐使用 |
| `WRecLift@K` | Whale Recall Lift | 推荐 ⭐ |
| `WRevCap@K` | Whale Revenue Capture | 金额口径 ⭐ |
| `EffN@K` | Effective Number | exp(Entropy) |
| `U-OverTarget@W` | User OverTarget | 用户侧过度触达 |
| `S-OverLoad@W` | Streamer OverLoad | 主播侧过载 |

## 7.5 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-01-15 | 基础指标 |
| v2.0 | 2026-01-18 | +切片 +生态 +校准 |
| v2.1 | 2026-01-19 | +决策核心 |
| **v2.2** | **2026-01-19** | 口径修订：金额/比例双版本、Regret 命名、nAUC 规范、Overload 拆分 |

---

> **文档完成时间**: 2026-01-19 (v2.2)
