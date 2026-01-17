# 🎁 GiftLive — 直播打赏预测与分配优化

> **离线实验仓库** | 数据: KuaiLive | 状态: Phase 0-3 ✅ 完成 | Phase 5 🔄 进行中 | 💬 WeChat: Viska-

## A. 问题定义（核心）

**目标**不是"让某个主播爆"，而是最大化全局长期收益：
- Total Revenue + 用户留存/满意度 + 主播生态健康

**难点**：
- 打赏极稀疏、重尾、高方差（少数"大哥"贡献大头）
- 延迟反馈（进房后很久才打赏 / 多次打赏）
- 在线分配是资源约束问题：大哥是稀缺资源，主播也有承接上限，且要避免过度集中导致生态/体验损伤

## 🔑 核心结论 (16个实验验证)

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
│ 1️⃣ 估计层: Direct Regression                        │
│    - 模型: LightGBM 回归 log(1+Y)                    │
│    - 特征: 用户-主播交互历史为核心                    │
│    - 输出: 期望收益 EV(u,s)                          │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 2️⃣ 分配层: Greedy + 软约束                          │
│    - 策略: argmax EV(u,s) + λ·冷启动bonus           │
│    - 参数: λ=0.5, min_alloc=10                      │
│    - 效果: 收益+32%, 新主播成功率+263%               │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 3️⃣ 评估层: SNIPS OPE                               │
│    - 方法: Self-Normalized IPS                      │
│    - 要求: 探索率≥0.3, 日志量≥5000                  │
│    - 精度: RelErr < 10%                             │
└─────────────────────────────────────────────────────┘
```

## 📊 关键数字

| 类别 | 指标 | 值 |
|------|------|-----|
| **数据特征** | 打赏率 | 1.48% |
| | User Gini | 0.942 (Top 1%贡献60%收益) |
| | Streamer Gini | 0.930 |
| | 矩阵密度 | 0.0064% |
| | 冷启动主播占比 | 92.2% |
| **预测性能** | Direct Reg Top-1% | 54.5% |
| | Direct Reg Spearman | 0.331 |
| | Two-Stage NDCG@100 | 0.359 (精排更优) |
| | Stage1 PR-AUC | 0.651 |
| | Stage1 ECE | 0.018 |
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
├── Q1: 估计层 - 如何准确预测极稀疏、重尾、延迟反馈的打赏行为？
│   ├── Q1.1: 两段式(是否打赏+金额)是否优于直接回归？ → ❌ 整体不优，但精排场景有优势
│   │   ├── H1: Stage2数据量/信息量不足 → ❌ 否定：Oracle_m增益仅+3.1pp
│   │   ├── H2: p×m乘法放大误差 → ❌ 否定：Stage1-only≈Two-Stage
│   │   ├── H3: Stage2 OOD问题 → ❌ 否定：gift子集Spearman=0.89
│   │   └── **主因**: ✅ Stage1分类不足（Oracle_p增益+54.9pp），推荐召回-精排分工
│   ├── Q1.2: 延迟反馈建模能否提升预测准确性？ → ❌ 审计通过，延迟不是问题
│   │   ├── ✅ 样本数差异已解释: 77,824 = gift×click配对（1.14x膨胀正常）
│   │   ├── ✅ pct_late_50=13.3%（代码bug已修复，之前单位错误）
│   │   └── ✅ 86.3%礼物立即发生，简单负例处理已足够
│   ├── Q1.3: 多任务学习能否用密集信号扶起稀疏信号？ → ❌ 无优势 (Δ=-1.76pp)
│   └── Q1.4: 重尾金额用log(1+Y)还是分位数回归？ → ⏳待验证
│
├── Q2: 决策层 - 如何在全局最优下分配高价值用户？
│   ├── Q2.1: 凹收益分配 vs 贪心分配的收益差异？ → ❌ Δ=-1.17%，无显著优势但公平性改善(Gini -0.018)
│   ├── Q2.2: 边际收益递减系数g'(V)如何设定/学习？ → ❌ 模拟环境下影响不大
│   ├── Q2.3: 冷启动/公平约束如何嵌入分配层？ → ✅ **软约束：收益+32%，成功率+263%**
│   └── Q2.4: 羊群效应/攀比效应是否存在？如何建模？ → ⏳待验证
│
└── Q3: 评估层 - 如何可靠评估分配策略？
    ├── Q3.1: OPE(IPS/DR)在高方差场景的有效性？ → ✅ **SNIPS可用，RelErr<10%，Gate-3关闭**
    │   ├── ✅ SNIPS最佳方法：Softmax 0.57%, Concave 4.31%, Greedy 9.97%
    │   ├── ❌ DR表现不如预期：Q函数偏差大于减少的方差
    │   └── ✅ 最优探索率~0.3，日志量≥5000
    ├── Q3.2: Simulator能否复现真实场景的关键特性？ → ✅ Gini可校准(<5%误差)，金额分布误差大
    │   ├── Q3.2.2: Simulator V2 能否修复金额分布失真？ → ✅ V2+ PASS
    │   └── Q3.2.3: 并发容量能否建模边际递减？ → ✅ PASS (边际递减24.4%)
    └── Q3.3: 评估窗口H如何设定？对结论有多敏感？ → ⏳待验证

Legend: ✅ 已验证 | ❌ 已否定 | ⏳ 待验证
```

## 📋 实验索引 (16/25 完成)

| MVP | 🔗 | 状态 | 关键结论 |
|-----|------|------|---------|
| 0.1 | [KuaiLive EDA](gift_allocation/exp/exp_kuailive_eda_20260108.md) | ✅ | Gini=0.94, 两段式建模必要性确认 |
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
├── gift_allocation/           # 主实验目录
│   ├── gift_allocation_hub.md         # 智库导航
│   ├── gift_allocation_roadmap.md     # 实验追踪
│   ├── exp/                           # 16个实验报告
│   ├── results/                       # 结果JSON
│   └── img/                           # 图表
├── scripts/                   # 训练脚本
└── data/KuaiLive/             # 数据集
```

## 🔗 导航

| 文档 | 路径 |
|------|------|
| 🧠 Hub | [gift_allocation_hub.md](gift_allocation/gift_allocation_hub.md) |
| 🗺️ Roadmap | [gift_allocation_roadmap.md](gift_allocation/gift_allocation_roadmap.md) |

## 📌 问题与解法

| 问题 | 解法 | 验证结果 |
|------|------|---------|
| **稀疏+重尾** (打赏率1.48%, Gini=0.94) | Direct Regression + log(1+Y) | Top-1% Capture = 54.5% |
| **延迟反馈** | 简单负例处理 | 86.3%礼物立即发生，无需延迟校正 |
| **资源约束** | Greedy + 软约束冷启动 | 收益+32%, 成功率+263% |

## 📅 更新日志

| 日期 | 事件 |
|------|------|
| 2026-01-09 | Phase 5 进行中：Gate-5A/5B FAIL，保留Direct-only + Greedy+Rules |
| 2026-01-08 | Phase 0-3 完成：12/13 MVP，Gate-2/3关闭 |
| 2026-01-08 | 推荐策略确定：Direct Reg + Greedy + 软约束 + SNIPS |

**作者**: Viska Wei | **数据**: KuaiLive | **完成度**: 64% (16/25 MVP)

---

## 📬 Contact Me
> 💬 WeChat:  **Viska-**  
> 📧 Email: **viskawei@gmail.com** 
