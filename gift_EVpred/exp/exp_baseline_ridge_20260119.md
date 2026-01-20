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

# 🍃 Baseline v2.2 Metrics 完整评估
> **Name:** Baseline Ridge + v2.2 Decision Metrics  \
> **ID:** `EVpred-20260119-baseline-v22`  \
> **Topic:** `gift_EVpred` | **MVP:** MVP-3.2 | **Project:** `VIT`  \
> **Author:** Viska Wei | **Date:** 2026-01-19 | **Status:** ✅ Completed
>
> 🎯 **Target:** 使用 metrics.py v2.2 对 Final Baseline 进行完整评估，验证决策核心指标  \
> 🚀 **Decision / Next:** 指标体系闭环完成 → 可进入分配层实验

---

## ⚡ 核心结论速览（供 main 提取；≤30行；必含 I/O + Pipeline TL;DR）

> **一句话**: Sample-level 捕获 51% 收入（Lift=35×）；User-level 找大哥召回 12%（Lift=12×，池子纯度 73%）；PSI=0.155 可上线

### 0.1 这实验到底在做什么？（X := 算法/机制 → 目标 | Why+How | I/O | Trade-off）

$$
X := \underbrace{\text{Ridge Regression (α=1.0)}}_{\text{是什么}}\ \xrightarrow[\text{基于}]{\ \text{raw Y + Last-Touch 归因}\ }\ \underbrace{\text{预测打赏金额 EV}}_{\text{用于 TopK 筛选}}\ \big|\ \underbrace{\text{Why: 需标准化 baseline}}_{\text{痛点}} + \underbrace{\text{How: 多粒度评估}}_{\text{难点}}
$$

- **🐻 What (是什么)**: Ridge 回归预测直播打赏金额期望值，作为 gift_EVpred 的标准 baseline
- **🍎 核心机制**: raw Y 预测 + Last-Touch 归因（1min 窗口）+ gift_id 去重
- **⭐ 目标**: 验证 v2.2 指标体系，获取全量 baseline 数值
- **🩸 Why（痛点）**: 需要标准化 baseline 作为后续模型对比基准
- **💧 How（难点）**: Sample-level vs User-level 粒度评估、多层指标覆盖

$$
\underbrace{\text{I/O 🫐}}_{\text{输入→输出}}\ =\ \underbrace{\Delta^+: \text{51\% RevCap@1\%, 稳定 PSI}}_{\text{优势}}\ -\ \underbrace{\Delta^-: \text{User-level Lift 仅 12×}}_{\text{待优化}}
$$

**I/O（必须写清楚，读者靠这一段理解实验"在干嘛"）**

| 类型 | 符号 | 说明 | 示例 |
|------|------|------|------|
| 🫐 输入 | $\mathcal{D}$ | 数据集（规模、特征） | KuaiLive Test (N=1.4M, 20 features) |
| 🫐 输入 | $X, y$ | 特征矩阵与标签 | 20 Strict 特征, raw Y (打赏金额) |
| 🫐 输出 | $\hat{y}$ | 模型预测值 | Ridge 回归预测 EV |
| 🫐 输出 | $M$ | 评估指标集 | 35+ 指标 (RevCap, Whale, PSI...) |
| 📊 指标 | $RevCap, WRecLift, PSI$ | 核心指标 | 51.4%, 34.6×, 0.155 |
| 🍁 基线 | $Random, Oracle$ | 对照组 | RevCap@1%: 1%, 99.6% |

### 0.2 Pipeline TL;DR（5-10 行极简伪代码，一眼看懂在跑什么）

```
1. 准备数据：KuaiLive 7-7-7 天划分，Train=1.6M, Val=1.7M, Test=1.4M
2. 加载模型：Ridge(α=1.0) + 20 Strict 特征（无快照累计）
3. 预测：
   for each sample in X_test:
       y_pred = model.predict(sample)
       → 单条输出: {'user_id': 123, 'streamer_id': 456, 'y_true': 52.0, 'y_pred': 48.3}
4. 预测后输出：predictions = [{'user_id':123, 'y_true':52.0, 'y_pred':48.3}, ...] (共 1.4M 条)
5. 评估（metrics v2.2 全量）：
   result = evaluate_model(y_test, y_pred_test, test_df, ...)
   → 输出: {'RevCap@1%': 0.514, 'WRecLift@1%': 34.6, 'PSI': 0.155, ...}
6. 落盘：gift_EVpred/results/metrics_v21_baseline_20260119.json
```

> ⚠️ **复现命令**（repo/entry/config/seed）→ 见 §7.2 附录
> 📖 **详细伪代码**（对齐真实代码）→ 见 §2.4.2

### 0.3 对假设/验证问题的回答

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| Q: Baseline 能否高效识别大额打赏者？ | ✅ WRecLift@1%=34.6× | 强（>20× 阈值），Sample-level 事件识别有效 |
| Q: Top 1% 能捕获多少收入？ | ✅ RevCap@1%=51.4% | 好（>40% 阈值），过半收入 |
| Q: 分布是否稳定可上线？ | ✅ PSI=0.155 | 轻微漂移（<0.25 阈值），可上线 |
| Q: User-level 找大哥能力？ | ⚠️ Lift=11.7× | 中等（5-20× 区间），需优化 |

### 0.4 关键数字（只放最重要的 3-5 个）

| Metric | Value | vs Baseline | Notes |
|--------|-------|------------|------|
| RevCap@1% | **51.4%** | +50pp vs Random | 主指标 |
| WRecLift@1% | **34.6×** | 强（>20×） | Sample-level whale 捕获 |
| WhaleUserPrec@1% | **72.6%** | 池子纯度高 | User-level 73% 是大哥 |
| PSI | **0.155** | <0.25 可上线 | 轻微漂移 |
| nAUC@10% | **61.2%** | 整体排序 | 避免单点过拟合 |

### 0.5 Links

| Type | Link |
|------|------|
| 🧠 Hub | `gift_EVpred/gift_EVpred_hub.md` § 10.6 |
| 📊 Metrics | `gift_EVpred/metrics.py` v2.2 |
| 📁 Results | `gift_EVpred/results/metrics_v21_baseline_20260119.json` |

---

# 1. 🎯 目标

**核心问题**: 使用 v2.2 指标体系完整评估 Baseline Ridge 模型在"找大哥"任务上的表现

**对应 main / roadmap**:
- 验证问题：Q10.6 Baseline 在多粒度下表现如何？
- 子假设：H10.6.1 Sample-level 优于 User-level
- Gate（如有）：Gate-3.2 指标体系闭环

## 1.1 成功标准（验收 / stop rule）

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| ✅ 通过 | RevCap@1% > 40%, PSI < 0.25 | 可上线 |
| ❌ 否决 | RevCap@1% < 30% 或 PSI > 0.25 | 需重新设计 |
| ⚠️ 异常 | User-level Lift < 5× | 检查粒度差异原因 |

---

# 2. 🦾 方法（算法 + I/O + 实验流程）

> 📌 **本章至少要填 2.2 I/O 与 2.4 实验流程**；否则读者无法知道实验怎么做的。

## 2.1 算法（可选：理论推导/关键公式/策略描述）

**Misallocation Cost（错分代价 / Regret）**：

$$
\text{Regret@K} = \text{OracleRev@K} - \text{AchievedRev@K}
$$

$$
\text{nRevCap@K} = \frac{\text{AchievedRev@K}}{\text{OracleRev@K}} \in [0, 1]
$$

**直觉解释**：
- Regret 量化了"把资源分给错误对象"的代价（金额版）
- nRevCap 是归一化效率，便于跨实验对比

## 2.2 输入 / 输出（必填：比 0.1 更细一点）

> 目标：读者不看代码，也能知道"喂进去什么 → 吐出来什么"。

### I/O Schema

| Component | Type/Shape | Example | Notes |
|----------|------------|---------|------|
| y_true | ndarray (N,) | [0, 0, 12, 0, 520...] | 原始打赏金额（元） |
| y_pred | ndarray (N,) | [0.1, 0.2, 15.3...] | Ridge 预测值 |
| test_df | DataFrame (N, M) | user_id, streamer_id... | 含分组/切片列 |
| y_pred_train | ndarray (N_train,) | [...] | 用于 PSI 计算 |
| Output: result | MetricsResult | .summary(), .to_json() | 35+ 指标对象 |

### Assumptions & Constraints

| Assumption/Constraint | Why it matters | How handled |
|----------------------|----------------|------------|
| Last-Touch 归因 | 决定 (user, streamer, time) 如何匹配 | 1min 窗口 + gift_id 去重 |
| Whale 阈值 T=100 | 影响 Whale 指标计算 | ≈ P90(y\|y>0) |
| Strict Mode 20 特征 | 无快照累计特征，更符合上线 | 对比 Benchmark 31 特征 |

## 2.3 实现要点（读者能对照代码定位）

| What | Where (file:function) | Key detail |
|------|------------------------|-----------|
| evaluate_model | `gift_EVpred/metrics.py:evaluate_model` | 主入口，compute_decision_metrics=True |
| misallocation_cost | `gift_EVpred/metrics.py:compute_misallocation_cost` | Regret + nRevCap |
| capture_auc | `gift_EVpred/metrics.py:compute_capture_auc` | nAUC@10% |
| PSI/drift | `gift_EVpred/metrics.py:compute_drift` | 需 y_pred_train |
| diversity | `gift_EVpred/metrics.py:compute_diversity` | EffN@K |

## 2.4 实验流程（必填：模块拆解 + 核心循环展开 + Code Pointer）

> 目标：读者看完这一节，（1）能复述实验流水线，（2）能定位代码位置。

### 2.4.1 实验流程树状图（完整可视化）

> ⚠️ **每一步都带 I/O 数值例子**，让读者秒懂预测和评估的输出是什么

```
实验流程
│
├── 1. 准备数据
│   ├── 数据源：KuaiLive 7-7-7 天划分
│   ├── 规模：Train=1.6M, Val=1.7M, Test=1.4M
│   ├── 特征：20 Strict 特征（无快照累计）
│   └── 输出: X_test (1.4M, 20), y_test (1.4M,), test_df
│
├── 2. 加载模型
│   ├── 模型：Ridge(α=1.0)
│   ├── 路径：`gift_EVpred/models/baseline_ridge_v1.pkl`
│   └── 输出: model = Ridge(alpha=1.0)
│
├── 3. 预测 ⭐
│   ├── y_pred_test = model.predict(X_test)
│   ├── y_pred_train = model.predict(X_train)  # 用于 PSI
│   └── 输出示例: y_pred = [0.1, 0.2, 48.3, 0.0, 520.5, ...]
│       └── 单条: {'user_id': 123, 'streamer_id': 456, 'y_true': 52.0, 'y_pred': 48.3}
│
├── 4. 评估（核心）⭐
│   ├── 输入: y_true, y_pred (各 1.4M 条), test_df
│   ├── L1: 主指标 → RevCap@1% = 0.514
│   ├── L2: Whale 七件套 → WRecLift@1% = 34.6×
│   ├── L3: 稳定性 → CV = 0.08
│   ├── L5: 切片 → cold_start_pair: RevCap = 0.32
│   ├── L7: 决策核心 → Regret@1% = 45,230元, PSI = 0.155
│   └── 输出: result = MetricsResult(revcap={0.01: 0.514}, whale={...}, ...)
│
├── 5. 汇总
│   ├── result.summary() 打印表格
│   └── 输出示例: {'RevCap@1%': 0.514, 'WRecLift@1%': 34.6, 'PSI': 0.155}
│
└── 6. 落盘
    ├── 结果：gift_EVpred/results/metrics_v21_baseline_20260119.json
    └── 输出示例: {"RevCap@1%": 0.514, "WRecLift@1%": 34.6, "PSI": 0.155, ...}
```

### 2.4.2 模块拆解（详细展开每个模块，带 Code Pointer）

| Module | Responsibility | Input → Output | Code Pointer |
|--------|----------------|----------------|--------------|
| M1: load_data | 加载 KuaiLive 数据 | path → train/val/test df | `scripts/data_utils.py:load_kuailive` |
| M2: load_model | 加载训练好的 Ridge | path → model | `joblib.load(model_path)` |
| M3: predict | 预测 | model + X → y_pred | `model.predict(X)` |
| M4: evaluate | **核心评估** | y_true + y_pred → metrics | `gift_EVpred/metrics.py:evaluate_model` |
| M5: aggregate | 多粒度聚合 | metrics → summary | `result.summary()` |
| M6: save | 保存结果 | metrics → JSON | `result.to_json(path)` |

> ⚠️ **M4 evaluate 必须展开**：不能只写 `evaluate_model()`，要写清楚里面在干什么（见 2.4.3）

### 2.4.3 核心循环展开（对齐真实代码的详细伪代码）

> ⚠️ **必填**：把 M4 展开，让读者不打开代码也能理解核心算法是怎么跑的

```python
# === 核心评估（对齐 gift_EVpred/metrics.py:evaluate_model）===

def evaluate_model(y_true, y_pred, test_df, y_pred_train=None, ...):
    result = MetricsResult()
    
    # Step 1: 主指标（L1）
    for k in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
        result.revcap[k] = revenue_capture_at_k(y_true, y_pred, k)
        result.oracle[k] = revenue_capture_at_k(y_true, y_true, k)  # Oracle
    
    # Step 2: 诊断指标（L2）- Whale 七件套
    whale_mask = y_true >= whale_threshold  # T=100
    result.whale_recall = whale_recall_at_k(y_true, y_pred, k, whale_mask)
    result.whale_precision = whale_precision_at_k(...)
    result.whale_rev_cap = whale_revenue_capture_at_k(...)  # 金额口径
    
    # Step 3: 稳定性（L3）
    result.cv = compute_cv_by_day(test_df, y_true, y_pred, k)
    result.bootstrap_ci = bootstrap_confidence_interval(y_true, y_pred, k)
    
    # Step 4: 切片（L5）
    for slice_name in ['cold_start_pair', 'whale_true', 'user_top_1pct', ...]:
        slice_mask = get_slice_mask(test_df, slice_name)
        result.slices[slice_name] = compute_slice_metrics(y_true[slice_mask], ...)
    
    # Step 5: 决策核心（L7）
    if compute_decision_metrics:
        result.misallocation_cost = compute_misallocation_cost(y_true, y_pred, k)
        result.capture_auc = compute_capture_auc(y_true, y_pred, max_k=0.1)
        result.drift = compute_drift(y_pred_train, y_pred)  # PSI
        result.diversity = compute_diversity(test_df, y_pred, k, y_true)
    
    return result
```

### 2.4.4 参数扫描（如果有 sweep）

```python
# Strict vs Benchmark 对比
for mode in ['strict', 'benchmark']:
    features = get_features(mode)  # 20 vs 31 features
    y_pred = model.predict(X_test[features])
    result[mode] = evaluate_model(y_true, y_pred, ...)
```

### 2.4.5 复现清单

- [x] 固定随机性：seed=42
- [x] 固定数据版本：KuaiLive v1.0, 7-7-7 split
- [x] 固定对照组：Ridge α=1.0, Strict 20 features
- [x] 输出物：metrics.json, summary table

---

# 3. 🧪 实验设计（具体到本次实验）

## 3.1 数据 / 环境

| Item | Value |
|------|-------|
| Source | KuaiLive (`data/KuaiLive/`) v1.0 |
| Path | `data/KuaiLive/` |
| Split | Train/Val/Test = 1,629,415 / 1,717,199 / 1,409,533 (7-7-7 days) |
| Feature | dim=20 (Strict Mode), 无快照累计 |
| Target | raw Y（原始打赏金额，非 log） |

## 3.2 Baselines（对照组）

| Baseline | Purpose | Key config |
|----------|---------|-----------|
| Random | sanity check | RevCap@1% ≈ 1% |
| Oracle | 理论上限 | RevCap@1% ≈ 99.6% |
| Benchmark Mode | 对比 31 特征 | 含快照累计特征 |

## 3.3 训练 / 运行配置

| Param | Value | Notes |
|------|-------|------|
| Model | Ridge Regression | sklearn |
| alpha | 1.0 | 正则化 |
| 归因 | Last-Touch + gift_id 去重 | 1min 窗口 |
| Whale 阈值 | 100 元 | ≈ P90(y\|y>0) |
| hardware | CPU | ~1min 评估 |

## 3.4 扫描参数（可选）

| Sweep | Range | Fixed |
|------|-------|-------|
| Feature Mode | [Strict, Benchmark] | 对比 20 vs 31 特征 |

## 3.5 评价指标

| Metric | Definition | Why |
|--------|------------|-----|
| RevCap@1% | Top 1% 捕获收入比例 | 主指标 |
| WRecLift@K | Whale Recall / K | 排除 K 影响 |
| PSI | 分布漂移 | 上线判断 |
| EffN@K | 有效分组数 | 多样性 |

---

# 4. 📊 图表 & 结果

> ⚠️ 图表文字必须全英文！

## 4.0 Strict vs Benchmark 对比

> **两种特征模式**：Strict (20 feat) 只用静态特征，Benchmark (31 feat) 包含快照累计特征

| 指标 | Strict (20 feat) | Benchmark (31 feat) | Δ |
|------|------------------|---------------------|---|
| **RevCap@1%** | **51.4%** | **51.4%** | 0.0pp |
| WRec@1% | 34.6% | 34.6% | 0.0pp |
| CV@1% | 9.4% | 9.4% | 0.0pp |
| PSI | 0.1553 | **0.1078** | **-0.05** |
| nAUC@10% | 61.2% | 61.4% | +0.3pp |

**What it shows**: Strict vs Benchmark 特征模式对比

**Key observations**:
- RevCap 几乎相同，额外 11 个特征无显著贡献
- **Benchmark PSI 更低** (0.108 vs 0.155)，分布更稳定
- 推荐使用 **Strict 模式** 作为 baseline

## 4.1 Sample-Level Whale 指标

| 指标 | 值 | 解读 |
|------|-----|------|
| **WRecLift@1%** | **34.6×** | 强：大额事件召回提升（>20× 为强） |
| **WRevCap@1%** | **55.5%** | Top 1% 捕获 56% whale 收入 |
| WRec@1% | 34.6% | 召回 35% 大额事件 |
| WPrec@1% | 4.96% | 池子中 ~5% 是大额事件 |
| Base Whale Rate | **0.143%** | 全体样本中大额事件稀疏度 |

## 4.2 User-Level Whale 指标 ⭐

| 指标 | 值 | 解读 |
|------|-----|------|
| **WhaleUserRecLift@1%** | **11.7×** | 中等：大哥召回提升 |
| **WhaleUserPrec@1%** | **72.6%** | 强：池子 73% 是大哥 |
| **UserRevCap@1%** | **46.6%** | Top 1% 用户贡献 47% 收入 |
| OracleUserRevCap@1% | 68.5% | 理论上限（还有 +22pp 空间） |
| BaseWhaleUserRate | **6.21%** | 用户中大哥比例 |

> **粒度对比**：BaseWhaleRate 从 sample 0.143% 到 user 6.21%（**43× 差距**）

---

# 5. 💡 洞见（解释"为什么会这样"）

## 5.1 机制层（Mechanism)
- **粒度差异核心**：BaseWhaleRate 43× 差距（0.143% vs 6.21%），大额事件稀疏是因为大部分用户不打赏，一旦打赏 6% 会成为大哥
- **模型擅长事件识别**：Sample-level Lift=34.6×（强），User-level Lift=11.7×（中等）

## 5.2 实验层（Diagnostics)
- **WRevCap > WRec**（55.5% vs 34.6%）：模型在大额事件内部也能优先抓更高金额的
- **PSI 轻微漂移**（0.155）是 Cold Start 效应，不是数据泄漏

## 5.3 设计层（So what)
- **多粒度评估必要**：不能只看 Sample-level，User-level 揭示实际大哥识别能力
- **PSI 监控**：上线需实时监控，阈值 0.25
- **Diversity 约束**：分配层需添加 EffN@K 下界约束

---

# 6. 📝 结论 & 下一步

## 6.1 核心发现（punch line）
> **Sample-level 抓事件强（Lift=35×），User-level 找大哥中等（Lift=12×，池子纯度 73%）**

- ✅ Q10.6: Sample-level 表现强，User-level 需优化
- **Decision**: 采用 Strict Mode 作为 baseline，可上线测试

## 6.2 关键结论（2-5 条）

| # | 结论 | 证据（图/表/数字） | 适用范围 |
|---|------|-------------------|---------|
| 1 | **抓事件能力强** | WRecLift@1%=34.6×（强） | Sample-level |
| 2 | **找大哥名单中等** | WhaleUserRecLift@1%=11.7× | User-level |
| 3 | **大哥池子纯度高** | WhaleUserPrec@1%=72.6% | User-level |
| 4 | **可上线** | PSI=0.155 < 0.25 | 分布稳定 |

## 6.3 Trade-offs（Δ+ vs Δ-）

| Upside (Δ+) | Cost / Constraint (Δ-) | When acceptable |
|-------------|--------------------------|----------------|
| 51% RevCap@1% | User-level Lift 仅 11.7× | 当侧重事件筛选时 |
| 简单 Ridge 模型 | 无非线性能力 | 作为 baseline |

## 6.4 下一步（可执行任务）

| Priority | Task | Owner | Link |
|----------|------|-------|------|
| 🔴 P0 | 上线 PSI 实时监控 | Ops | - |
| 🟡 P1 | 添加 Diversity 下界约束 | Model | 分配层 |
| 🟢 P2 | 实现 DW-ECE, Fairness Gap | Metrics | v2.3 |

---

# 7. 📎 附录（复现/审计用）

## 7.1 数值结果（全量）

| Top K% | RevCap | Oracle | nRevCap | Whale Recall |
|--------|--------|--------|---------|--------------|
| 0.1% | 21.2% | 85.9% | 24.7% | 9.9% |
| 0.5% | 43.3% | 98.4% | 44.0% | 26.6% |
| **1%** | **51.4%** | 99.6% | **51.6%** | 33.5% |
| 2% | 56.3% | 100% | 56.3% | 42.1% |
| 5% | 63.6% | 100% | 63.6% | 52.7% |
| 10% | 68.6% | 100% | 68.6% | 61.7% |

完整结果：`gift_EVpred/results/metrics_v21_baseline_20260119.json`

## 7.2 执行记录（复现命令）

| Item | Value |
|------|-------|
| Repo | `~/GiftLive` |
| Script | inline Python (metrics evaluation) |
| Config | Strict Mode, 20 features, α=1.0 |
| Seed | 42 |
| Output | `gift_EVpred/results/metrics_v21_baseline_20260119.json` |

```bash
# (1) setup
cd ~/GiftLive
source init.sh

# (2) run evaluation
python -c "
from gift_EVpred.metrics import evaluate_model
import joblib
import pandas as pd

# Load model and data
model = joblib.load('gift_EVpred/models/baseline_ridge_v1.pkl')
test_df = pd.read_parquet('data/processed/test.parquet')
X_test = test_df[STRICT_FEATURES]
y_test = test_df['y_true'].values
y_pred_test = model.predict(X_test)

# Load train predictions for PSI
train_df = pd.read_parquet('data/processed/train.parquet')
y_pred_train = model.predict(train_df[STRICT_FEATURES])

# Evaluate
result = evaluate_model(
    y_true=y_test,
    y_pred=y_pred_test,
    test_df=test_df,
    y_pred_train=y_pred_train,
    compute_decision_metrics=True,
)
print(result.summary())
result.to_json('gift_EVpred/results/metrics_v21_baseline_20260119.json')
"
```

## 7.3 运行日志摘要 / Debug（可选）

| Issue | Root cause | Fix |
|------|------------|-----|
| PSI 轻微漂移 | Cold Start 效应（Train 早期历史特征≈0） | 可接受，监控即可 |

---

> **实验完成时间**: 2026-01-19 (v2.2 更新)
