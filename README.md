# 🎁 GiftLive — 直播打赏预测与分配优化

> **离线实验仓库** | 数据: KuaiLive | 状态: **EVpred ✅ Baseline v1.0** | **Allocation Phase 5 🔄** | 💬 WeChat: Viska-

## A. 问题定义（核心）

**目标**不是"让某个主播爆"，而是最大化全局长期收益：
- Total Revenue + 用户留存/满意度 + 主播生态健康

**难点**：
- 打赏极稀疏、重尾、高方差（少数"大哥"贡献大头）
- 延迟反馈（进房后很久才打赏 / 多次打赏）
- 在线分配是资源约束问题：大哥是稀缺资源，主播也有承接上限，且要避免过度集中导致生态/体验损伤

## 🔑 核心结论

### 估计层 (gift_EVpred) - 20+ 实验验证

| # | 结论 | 证据 | 决策 |
|---|------|------|------|
| **E1** | **Raw Y >> Log Y** | RevCap +39.6% | 预测目标用 raw Y |
| **E2** | **Ridge > LightGBM（稀疏回归）** | LGB -14.5% | Linear 优于 Tree |
| **E3** | **RevCap@1% = 51.4%** | CV=9.4%, 95%CI [47.2%, 54.4%] | Final Baseline v1.0 |
| **E4** | **WRecLift = 34.6×** | 识别大额事件能力强 | Sample-level 有效 |
| **E5** | **Pair 历史是最强信号** | 系数 0.194 | 重点挖掘 pair 特征 |
| **E6** | **打赏与留存负相关** | 打赏用户留存 -10.3pp | 短期 ≠ 长期价值 |
| **E7** | **收入极度集中** | Gini=0.926, Top 1%占93.7% | 分配层需生态约束 |

### 分配层 (gift_allocation) - 16 实验验证

| # | 结论 | 证据 | 决策 |
|---|------|------|------|
| K1 | **直接回归优于两段式** | Top-1%: 54.5% vs 35.7% | 保留简单架构 |
| K2 | **延迟反馈不是问题** | 86.3%礼物立即发生 | 简单负例处理足够 |
| K3 | **凹收益分配无显著优势** | Δ=-1.17%, Gini改善-0.018 | 简化为Greedy |
| K4 | **软约束冷启动有效** | 收益+32%, 成功率+263% | 采用λ=0.5 |
| K5 | **多任务学习无优势** | Δ PR-AUC=-1.76pp | 保留单任务 |
| K6 | **Two-Stage适合精排但召回覆盖率不足** | gift子集Spearman=0.89，但Recall Coverage仅5.5% | ❌ 分工架构不可行，保留Direct-only |
| K7 | **交互历史是最强特征** | 重要性是第二名3倍 | 优先维护交互特征 |
| K8 | **SNIPS是最佳OPE方法** | RelErr<10% | 探索率≥0.3, 日志≥5000 |
| K9 | **分配策略是最大杠杆** | Greedy 3x Random | 优化分配层 > 优化估计层 |
| K10 | **高并发存在边际递减** | Revenue/User 下降24.4% | 需考虑容量约束 |
| K11 | **简单规则优于复杂框架** | Greedy+Rules +4.36% vs Shadow Price +2.74% | 保留Greedy+软约束 |

## 🏗️ 推荐架构

```
用户请求
    ↓
┌─────────────────────────────────────────────────────┐
│ 1️⃣ 估计层: Ridge Regression (gift_EVpred)           │
│    - 模型: Ridge (α=1.0) 回归 raw Y（非 log）        │
│    - 特征: 20 个 Strict 特征（Pair/User/Str 历史）   │
│    - 归因: Last-Touch + gift_id 去重 + 1min 窗口    │
│    - 输出: 期望收益 EV(u,s)                          │
│    - 性能: RevCap@1%=51.4%, WRecLift=34.6×          │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 2️⃣ 分配层: Greedy + 软约束 (gift_allocation)        │
│    - 策略: argmax EV(u,s) + λ·冷启动bonus           │
│    - 参数: λ=0.5, min_alloc=10                      │
│    - 效果: 收益+32%, 新主播成功率+263%               │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 3️⃣ 评估层: SNIPS OPE + Metrics v2.2                │
│    - 方法: Self-Normalized IPS                      │
│    - 要求: 探索率≥0.3, 日志量≥5000                  │
│    - 精度: RelErr < 10%                             │
│    - 指标: 7 层 38 指标体系                          │
└─────────────────────────────────────────────────────┘
```

## 📊 关键数字

| 类别 | 指标 | 值 |
|------|------|-----|
| **数据特征** | 打赏率 | 1.48% |
| | User Gini | 0.942 (Top 1%贡献60%收益) |
| | Streamer Gini | 0.926 (Top 1%占93.7%收入) |
| | 矩阵密度 | 0.0064% |
| | 冷启动主播占比 | 92.2% |
| **EVpred Baseline** | **RevCap@1%** | **51.4%** (Strict Mode) |
| | **CV (稳定性)** | **9.4%** (首次 <10%) |
| | 95% CI | [47.2%, 54.4%] |
| | WRecLift@1% (Sample) | 34.6× |
| | WhaleUserPrec@1% (User) | 72.6% |
| | 特征数 | 20 (Strict Mode) |
| | PSI (可上线性) | 0.155 (<0.25阈值) |
| **留存洞见** | 打赏用户留存 | 78.9% (vs 未打赏 89.2%) |
| | 留存 AUC | 0.57 (不可预测) |
| **预测性能 (旧)** | Direct Reg Top-1% | 54.5% |
| | Two-Stage NDCG@100 | 0.359 (精排更优) |
| **分配性能** | Greedy/Random收益比 | 2.94x |
| | 软约束收益提升 | +32% |
| | 软约束成功率提升 | +263% |
| **评估性能** | SNIPS最优RelErr | 0.57% (Softmax) |
| | Simulator Gini误差 | <5% |
| **Simulator V2+** | V2+ P50/P90/P99误差 | 0% / 13.2% / 26.0% |
| | V2+ Mean误差 | 24.0% |
| **并发容量** | Revenue/User下降 | 24.4% |
| | 拥挤率@800用户 | 67.9% |
| **召回-精排** | Recall Coverage @1000 | 5.5% |
| **影子价格** | Shadow Price Δ收益 | +2.74% (< Greedy+Rules +4.36%) |

## 🌲 核心假设树

```
🌲 核心: 如何在直播场景下最大化全局打赏收益？
│
├── Q1: 估计层 (gift_EVpred) - 如何识别高价值用户？ ✅ Baseline v1.0
│   ├── Q1.1: Raw Y vs Log(1+Y)？ → ✅ Raw Y 显著更优 (+39.6%)
│   ├── Q1.2: Direct vs Two-Stage？ → ✅ Direct 更优 (52.7% vs 45.7%)
│   ├── Q1.3: Ridge vs LightGBM？ → ✅ Ridge 更优，Tree 不适合稀疏 (+14.5%)
│   ├── Q1.4: 最强特征？ → ✅ pair_gift_cnt_hist（Pair 历史）
│   ├── Q1.5: 延迟反馈？ → ✅ 1min 窗口覆盖 92.6%，无需延迟校正
│   └── Q1.6: 预测目标扩展？ → ⚠️ 留存不可预测 (AUC=0.57)，作为护栏
│       └── 🔥 打赏与留存负相关：打赏用户留存 -10.3pp
│
├── Q2: 分配层 (gift_allocation) - 如何在全局最优下分配高价值用户？
│   ├── Q2.1: 凹收益分配 vs 贪心分配？ → ❌ Δ=-1.17%，无显著优势
│   ├── Q2.2: 冷启动约束？ → ✅ **软约束：收益+32%，成功率+263%**
│   ├── Q2.3: 并发容量？ → ✅ 边际递减 24.4%，需考虑容量约束
│   └── Q2.4: 影子价格？ → ❌ +2.74% < Greedy+Rules +4.36%
│
└── Q3: 评估层 - 如何可靠评估分配策略？
    ├── Q3.1: OPE 方法？ → ✅ SNIPS 最佳，RelErr<10%
    ├── Q3.2: Simulator？ → ✅ V2+ PASS，Gini误差<5%
    └── Q3.3: 指标体系？ → ✅ **Metrics v2.2：7 层 38 指标**

Legend: ✅ 已验证 | ❌ 已否定 | ⏳ 待验证 | ⚠️ 部分成功
```

## 📋 实验索引

### gift_EVpred (估计层) - Baseline v1.0 ✅

| 日期 | 报告 | 状态 | 关键结论 |
|------|------|------|---------|
| 01-19 | [Baseline Ridge](gift_EVpred/exp/exp_baseline_ridge_20260119.md) | ✅ | **RevCap@1%=51.4%, CV=9.4%** |
| 01-19 | [Metrics v2.2](gift_EVpred/exp/exp_metrics_20260119.md) | ✅ | 7 层 38 指标体系 |
| - | [Raw vs Log](gift_EVpred/exp/) | ✅ | Raw Y >> Log Y (+39.6%) |
| - | [LightGBM](gift_EVpred/exp/) | ❌ | Tree 不适合稀疏回归 (-14.5%) |
| - | [预测目标扩展](gift_EVpred/exp/) | ⚠️ | 留存不可预测 (AUC=0.57) |

**Slides**: [Baseline v2.2](gift_EVpred/slide/slides_baseline_ridge_20260119.md) | [Metrics v2.2](gift_EVpred/slide/slides_metrics_v22_20260119.md)

### gift_allocation (分配层) - 16/25 完成

| MVP | 🔗 | 状态 | 关键结论 |
|-----|------|------|---------|
| 0.1 | [KuaiLive EDA](KuaiLive/exp/exp_kuailive_eda_20260108.md) | ✅ | Gini=0.94, 两段式建模必要性确认 |
| 0.2 | [Baseline](gift_allocation/exp/exp_baseline_20260108.md) | ✅ | Top-1%=56.2%, 交互特征主导 |
| 0.3 | [Simulator V1](gift_allocation/exp/exp_simulator_v1_20260108.md) | ✅ | Gini误差<5%, Greedy 3x Random |
| 1.1 | [Two-Stage](gift_allocation/exp/exp_two_stage_20260108.md) | ✅ | 揭示与Baseline不可直接对比 |
| 1.1-fair | [公平对比](gift_allocation/exp/exp_fair_comparison_20260108.md) | ✅ | Direct胜出 (Δ=+18.8pp) |
| 1.2 | [延迟建模](gift_allocation/exp/exp_delay_modeling_20260108.md) | ✅ | 延迟不是问题，DG2关闭 |
| 1.2-audit | [延迟审计](gift_allocation/exp/exp_delay_audit_20260108.md) | ✅ | 代码bug修复，pct_late_50=13.3% |
| 1.3 | [多任务](gift_allocation/exp/exp_multitask_20260108.md) | ✅ | 无优势 (Δ=-1.76pp)，DG5关闭 |
| 1.4 | [诊断拆解](gift_allocation/exp/exp_two_stage_diagnosis_20260108.md) | ✅ | 主因=Stage1分类不足，Stage2精排有优势 |
| 2.1 | [凹收益分配](gift_allocation/exp/exp_concave_allocation_20260108.md) | ✅ | Δ=-1.17%无显著优势，DG3关闭 |
| 2.2 | [冷启动约束](gift_allocation/exp/exp_coldstart_constraint_20260108.md) | ✅ | 软约束+32%收益，Gate-2关闭 |
| 3.1 | [OPE验证](gift_allocation/exp/exp_ope_validation_20260108.md) | ✅ | SNIPS最佳，Gate-3关闭 |
| 4.1+ | [Simulator V2+](gift_allocation/exp/exp_simulator_v2_20260109.md) | ✅ | P50=0%, P90=13%, Mean=24% |
| 4.2 | 并发容量 | ✅ | 边际递减24.4%, 拥挤率68% |
| 5.1 | [召回-精排](gift_allocation/exp/exp_recall_rerank_20260109.md) | ❌ | Recall Coverage仅5.5%，Gate-5A FAIL |
| 5.2 | [影子价格](gift_allocation/exp/exp_shadow_price_20260109.md) | ❌ | +2.74% < Greedy+Rules +4.36%，Gate-5B FAIL |

## 📁 项目结构

```
GiftLive/
├── gift_EVpred/               # 估计层（EV 预测）
│   ├── gift_EVpred_hub.md             # 智库导航
│   ├── exp/                           # 实验报告
│   ├── slide/                         # Slides
│   ├── card/                          # 知识卡片
│   ├── data_utils.py                  # 数据处理入口（Final v1.0）
│   └── metrics.py                     # 评估模块（v2.2）
├── gift_allocation/           # 分配层
│   ├── gift_allocation_hub.md         # 智库导航
│   ├── gift_allocation_roadmap.md     # 实验追踪
│   ├── exp/                           # 16个实验报告
│   ├── results/                       # 结果JSON
│   └── img/                           # 图表
├── KuaiLive/                  # 数据 EDA
├── scripts/                   # 训练脚本
└── data/KuaiLive/             # 数据集
```

## 🔗 导航

| 模块 | Hub | 其他 |
|------|-----|------|
| **估计层** | [gift_EVpred_hub.md](gift_EVpred/gift_EVpred_hub.md) | [Slides](gift_EVpred/slide/) |
| **分配层** | [gift_allocation_hub.md](gift_allocation/gift_allocation_hub.md) | [Roadmap](gift_allocation/gift_allocation_roadmap.md) |
| **数据** | [kuailive_hub.md](KuaiLive/kuailive_hub.md) | - |

## 📌 问题与解法

| 问题 | 解法 | 验证结果 |
|------|------|---------|
| **稀疏+重尾** (打赏率1.48%, Gini=0.93) | Ridge + raw Y (非 log) | RevCap@1%=51.4%, WRecLift=34.6× |
| **数据泄漏风险** | Day-Frozen + Last-Touch 归因 | 200/200 样本验证通过 |
| **延迟反馈** | 1min 窗口 + gift_id 去重 | 92.6% 覆盖率，无 Over-Attribution |
| **资源约束** | Greedy + 软约束冷启动 | 收益+32%, 成功率+263% |
| **短期 vs 长期** | 留存作为护栏（非预测目标） | 打赏用户留存低 10.3pp |

## 📅 更新日志

| 日期 | 事件 |
|------|------|
| **2026-01-20** | **EVpred Baseline v1.0 发布**：RevCap@1%=51.4%, CV=9.4%, 20 Strict 特征 |
| **2026-01-20** | **Metrics v2.2 上线**：7 层 38 指标体系，PSI=0.155 可上线 |
| **2026-01-19** | EVpred 发现：打赏与留存负相关 (-10.3pp)，留存不可预测 (AUC=0.57) |
| **2026-01-19** | EVpred 修复：Last-Touch 归因 + gift_id 去重，消除 Over-Attribution |
| 2026-01-09 | Phase 5 进行中：Gate-5A/5B FAIL，保留Direct-only + Greedy+Rules |
| 2026-01-08 | Phase 0-3 完成：12/13 MVP，Gate-2/3关闭 |
| 2026-01-08 | 推荐策略确定：Ridge + raw Y + Greedy + 软约束 + SNIPS |

**作者**: Viska Wei | **数据**: KuaiLive | **完成度**: EVpred ✅ Baseline v1.0 | Allocation 64% (16/25 MVP)

---

## 📬 Contact Me
> 💬 WeChat:  **Viska-**  
> 📧 Email: **viskawei@gmail.com** 
