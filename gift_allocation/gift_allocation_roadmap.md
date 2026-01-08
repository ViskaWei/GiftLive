# 🗺️ Gift Allocation Roadmap
> **Name:** 直播打赏预测与金主分配优化 | **ID:** `EXP-20260108-gift-allocation-roadmap`  
> **Topic:** `gift_allocation` | **Phase:** 0  
> **Author:** Viska Wei | **Date:** 2026-01-08 | **Status:** 🔄
```
💡 当前阶段目标  
Gate：Phase 0 - 数据探索 + Baseline建立
```

---

## 🔗 Related Files
| Type | File |
|------|------|
| 🧠 Hub | `gift_allocation_hub.md` |
| 📋 Kanban | `status/kanban.md` |
| 📗 Experiments | `exp/exp_*.md` |

---

# 1. 🚦 Decision Gates

> Roadmap 定义怎么验证，Hub 做战略分析

## 1.1 战略路线 (来自Hub)

| Route | 名称 | Hub推荐 | 验证Gate |
|-------|------|---------|----------|
| A | 端到端学习 (E2E) | 🔴 | - |
| **B** | 分层建模 (估计→分配) | 🟢 **推荐** | Gate-1, Gate-2 |
| C | 纯Bandit探索 | 🟡 | Gate-3 |

## 1.2 Gate定义

### Gate-1: 估计层验证
| 项 | 内容 |
|----|------|
| 验证 | 两段式建模 + 延迟校正是否显著优于baseline |
| MVP | MVP-1.1, MVP-1.2, MVP-1.3, **MVP-1.4, MVP-1.5, MVP-1.2-audit, MVP-1.2-pseudo** |
| 若A | PR-AUC提升>5% + ECE改善>0.02 → 确认分层路线 |
| 若B | 提升不显著 → 考虑端到端或更简单方案 |
| 状态 | 🔄 进行中 |
| **子Gate** | **内容** |
| DG1.1 | Two-Stage诊断：证明Stage2不足/OOD/乘法噪声是主因 → 关闭或转精排 |
| DG2.1 | 延迟数据审计：样本数/口径一致性 → 确认DG2结论基础 |
| DG2.2 | 伪在线截断验证：Chapelle在真实延迟场景的价值 |

### Gate-2: 分配层验证
| 项 | 内容 |
|----|------|
| 验证 | 凹收益分配是否优于贪心分配 |
| MVP | MVP-2.1, MVP-2.2 |
| 若A | 总收益提升>10% + Gini改善 → 确认分配层 |
| 若B | 提升不显著 → 简化为贪心+约束 |
| 状态 | ✅ **已关闭** |
| **结果** | **凹收益Δ=-1.17%无优势(concave_exp)；Gini改善-0.018；决策：Greedy+约束** |

### Gate-3: 评估层验证
| 项 | 内容 |
|----|------|
| 验证 | OPE评估是否可靠 |
| MVP | MVP-3.1 |
| 若A | OPE与Simulator结论一致 → OPE可用于线上评估 |
| 若B | 偏差过大 → 依赖Simulator或小流量A/B |
| 状态 | ⏳ |

## 1.3 本周重点

| 优先级 | MVP | Gate | 状态 |
|--------|-----|------|------|
| ✅ | MVP-0.1 | - | ✅ 数据探索完成 |
| ✅ | MVP-0.2 | - | ✅ Baseline完成 (Top-1%=56.2%) |
| ✅ | MVP-1.1 | Gate-1 | ✅ 完成 (揭示不可直接对比) |
| ✅ | MVP-1.1-fair | Gate-1 | ✅ Direct Reg 胜出 (Δ=-18.7pp) |
| ⚠️ | MVP-1.2 | Gate-1 | ⚠️ 完成但存在数据红旗，需审计 |
| 🔴 **P0** | **MVP-1.4** | Gate-1/DG1.1 | 🔴 **Two-Stage诊断拆解** |
| 🔴 **P0** | **MVP-1.2-audit** | Gate-1/DG2.1 | 🔴 **延迟数据审计** |
| 🟡 P1 | MVP-1.5 | Gate-1/DG1.1 | 🔴 Stage2改进 + 召回-精排分工 |
| 🟡 P1 | MVP-1.2-pseudo | Gate-1/DG2.2 | 🔴 伪在线截断验证 |

---

# 2. 📋 MVP列表

## 2.1 总览

| MVP | 名称 | Phase | Gate | 状态 | exp_id | 报告 |
|-----|------|-------|------|------|--------|------|
| 0.1 | KuaiLive数据探索 | 0 | - | ✅ | EXP-20260108-gift-allocation-01 | [exp_kuailive_eda_20260108.md](./exp/exp_kuailive_eda_20260108.md) |
| 0.2 | Baseline(直接回归) | 0 | - | ✅ | EXP-20260108-gift-allocation-02 | [exp_baseline_20260108.md](./exp/exp_baseline_20260108.md) |
| 0.3 | Simulator V1构建 | 0 | - | ✅ | EXP-20260108-gift-allocation-07 | [exp_simulator_v1_20260108.md](./exp/exp_simulator_v1_20260108.md) |
| 1.1 | 两段式建模 | 1 | Gate-1 | ✅ | EXP-20260108-gift-allocation-03 | [exp_two_stage_20260108.md](./exp/exp_two_stage_20260108.md) |
| **1.1-fair** | **两段式公平对比** | 1 | Gate-1 | ✅ | EXP-20260108-gift-allocation-04 | [exp_fair_comparison_20260108.md](./exp/exp_fair_comparison_20260108.md) |
| 1.2 | 延迟反馈建模(生存) | 1 | Gate-1 | ✅ | EXP-20260108-gift-allocation-05 | [exp_delay_modeling_20260108.md](./exp/exp_delay_modeling_20260108.md) |
| **1.3** | **多任务学习** | 1 | Gate-1 | ✅ | EXP-20260108-gift-allocation-06 | [exp_multitask_20260108.md](./exp/exp_multitask_20260108.md) |
| **1.4** | **Two-Stage诊断拆解** | 1 | DG1.1 | ✅ | EXP-20260108-gift-allocation-11 | [exp_two_stage_diagnosis_20260108.md](./exp/exp_two_stage_diagnosis_20260108.md) |
| **1.5** | **Stage2改进+召回-精排分工** | 1 | DG1.1 | ⏳ | EXP-20260108-gift-allocation-12 | [exp_two_stage_improve_20260108.md](./exp/exp_two_stage_improve_20260108.md) |
| **1.2-audit** | **延迟数据审计** | 1 | DG2.1 | ✅ | EXP-20260108-gift-allocation-13 | [exp_delay_audit_20260108.md](./exp/exp_delay_audit_20260108.md) |
| ~~1.2-pseudo~~ | ~~伪在线截断验证~~ | 1 | ~~DG2.2~~ | ❌取消 | - | 延迟问题不存在，无需验证 |
| **2.1** | **凹收益分配层** | 2 | Gate-2 | ✅ | EXP-20260108-gift-allocation-08 | [exp_concave_allocation_20260108.md](./exp/exp_concave_allocation_20260108.md) |
| **2.2** | **冷启动/公平约束** | 2 | Gate-2 | ✅ | EXP-20260108-gift-allocation-09 | [exp_coldstart_constraint_20260108.md](./exp/exp_coldstart_constraint_20260108.md) |
| 3.1 | OPE验证(IPS/DR) | 3 | Gate-3 | ⏳ | EXP-20260108-gift-allocation-10 | [exp_ope_validation_20260108.md](./exp/exp_ope_validation_20260108.md) |

**状态**: ⏳计划 | 🔴就绪 | 🚀运行 | ✅完成 | ⚠️需审计 | ❌取消

## 2.2 配置速查

| MVP | 数据 | 模型 | 关键变量 |
|-----|------|------|---------|
| 0.1 | KuaiLive | - | 统计分析 |
| 0.2 | KuaiLive | GBDT | 直接回归Y |
| 0.3 | 合成数据 | - | Simulator参数 |
| 1.1 | KuaiLive | GBDT×2 | p(x), m(x) |
| **1.1-fair** | KuaiLive (click全量) | GBDT | 统一候选集对比 |
| 1.2 | KuaiLive + timestamp | GBDT+Weibull | 延迟分布F(d), 软标签w |
| 1.3 | KuaiLive | 多任务NN | click/watch/gift |
| **1.4** | KuaiLive (复用1.1-fair) | 分析型 | Oracle p/m分解，Stage-wise评估 |
| **1.5** | KuaiLive | GBDT + 后处理 | shrinkage α, 召回-精排pipeline |
| **1.2-audit** | KuaiLive原始数据 | - | gift→click匹配，口径审计 |
| **1.2-pseudo** | KuaiLive + 截断 | GBDT+Chapelle | 伪在线截断，混合Weibull |
| 2.1 | Simulator | 凹分配 | g(V), g'(V) |
| 2.2 | Simulator | 约束优化 | λ冷启动 |

---

# 3. 🔧 MVP规格

## Phase 0: Baseline + 数据探索

### MVP-0.1: KuaiLive数据探索
| 项 | 配置 |
|----|------|
| 目标 | 理解数据分布：打赏率、金额分布、时间模式 |
| 数据 | KuaiLive全量 |
| 分析 | 1) 打赏率统计 2) 金额分布(Pareto?) 3) 时间延迟分布 4) 用户/主播长尾 |
| 验收 | 产出关键数字：打赏率、P50/P90/P99金额、Gini系数 |

### MVP-0.2: Baseline(直接回归)
| 项 | 配置 |
|----|------|
| 目标 | 建立直接回归Y的baseline |
| 数据 | KuaiLive, 按天切分 |
| 模型 | LightGBM回归log(1+Y) |
| 验收 | 产出：MAE(log), Top-1%捕获率 |

### MVP-0.3: Simulator V1构建
| 项 | 配置 |
|----|------|
| 目标 | 构建可控的打赏+分配模拟器 |
| 设计 | 用户财富w~LogNormal+Pareto尾; 偏好向量p,q; 匹配度c=p·q |
| 打赏概率 | σ(θ₀+θ₁w+θ₂c+θ₃engage) |
| 打赏金额 | exp(μ₀+μ₁w+μ₂c+ε), ε~N(0,σ²) |
| 延迟 | T~Weibull; 删失if T>观看时长W |
| 外部性 | 边际递减: Y/(1+γN_big) |
| 验收 | 能复现真实数据的关键统计量（打赏率、Gini） |

## Phase 1: 估计层

### MVP-1.1: 两段式建模
| 项 | 配置 |
|----|------|
| 目标 | 验证两段式(p×m)是否优于直接回归 |
| 模型 | Stage1: LightGBM分类(是否打赏) + focal loss; Stage2: LightGBM回归(条件金额) |
| 评估 | Gift: PR-AUC, ECE; EV: Top-1%捕获率, Spearman |
| 验收 | 对比MVP-0.2，PR-AUC/捕获率提升 |
| 结论 | ⚠️ 揭示与Baseline在不同数据集，不可直接对比 |

### MVP-1.1-fair: 两段式公平对比 🔴
| 项 | 配置 |
|----|------|
| 目标 | 在相同click全量数据上公平对比两段式与直接回归 |
| 数据 | click全量 (4.9M)，统一train/val/test切分 |
| 模型A | Direct Regression: LightGBM回归log(1+Y)，含0值 |
| 模型B | Two-Stage: 已有Stage1 + Stage2 |
| 评估 | Top-1%/5%/10% Capture, Spearman, NDCG@100, PR-AUC, ECE |
| 验收 | Top-1%提升≥5% → 确认两段式; Else → 保留Baseline |
| 关闭 | DG1 |

### MVP-1.2: 延迟反馈建模 🟡
| 项 | 配置 |
|----|------|
| 目标 | 验证延迟校正/生存建模的增益 |
| 数据 | click全量 + timestamp (click_time, gift_time, watch_duration) |
| 方法A | Chapelle方法: 软标签加权 $w_i = 1 - F(H - t_i)$ |
| 方法B | 生存分析: Weibull AFT, 估计 $\Pr(T \leq H|x)$ |
| 评估 | ECE, Brier Score, 近期样本PR-AUC (分窗口) |
| 验收 | ECE改善≥0.02, 近期样本预测稳定 |
| 关闭 | DG2 |

### MVP-1.3: 多任务学习
| 项 | 配置 |
|----|------|
| 目标 | 用密集信号(click/watch/comment)扶起稀疏打赏 |
| 模型 | 共享表征f(x) + 多头: CTR, 观看时长, 打赏, 金额 |
| 架构 | 两塔(用户塔+主播塔) + MLP |
| 验收 | 对比单任务，打赏预测PR-AUC提升 |

### MVP-1.4: Two-Stage诊断拆解 🔴 P0
| 项 | 配置 |
|----|------|
| 目标 | 定位Two-Stage输的主因：Stage2不足 / 乘法噪声 / OOD |
| 关闭 | DG1.1 |
| **实验1** | **Stage1-only排序能力**：用p(x)直接做Top-K/NDCG/Spearman |
| **实验2** | **Stage2在gift子集能力**：筛test中Y>0，对比m(x) vs ŷ(x)的Spearman_gift |
| **实验3** | **Oracle p分解**：用真实1(Y>0)替换p(x)，看Top指标上界 |
| **实验4** | **Oracle m分解**：用真实log(1+Y)替换m(x)（仅gift），看上界 |
| 验收 | 明确主因是哪个组件；若Stage2主因→进MVP-1.5改进；若乘法主因→关闭Two-Stage |

### MVP-1.5: Stage2改进 + 召回-精排分工 🟡 P1
| 项 | 配置 |
|----|------|
| 目标 | 验证改进后的Two-Stage能否在精排场景胜出 |
| 依赖 | MVP-1.4 诊断结论 |
| **方法A** | **Stage2 Shrinkage**: $m'(x)=\alpha(x) \cdot m(x)+(1-\alpha(x)) \cdot \bar{m}$，α由历史gift次数决定 |
| **方法B** | **Stage2强正则**：更浅树/更大min_data_in_leaf/更强L2 |
| **方法C** | **召回-精排分工**：Direct选Top-M候选，Two-Stage在候选上re-rank |
| 评估 | Top-1%/NDCG@100/总收益，验证分工策略是否优于单一模型 |
| 验收 | If 分工策略>单一模型→采纳；Else→关闭Two-Stage路线 |

### MVP-1.2-audit: 延迟数据审计 🔴 P0
| 项 | 配置 |
|----|------|
| 目标 | 验证延迟实验的数据一致性，关闭DG2.1 |
| 关闭 | DG2.1 |
| **审计1** | **Gift→Click一对一匹配**：每条gift找click候选集，检查是否>1匹配 |
| **审计2** | **0延迟质量核验**：gift_ts==click_ts占比 + watch_live_time分布 |
| **审计3** | **pct_late_*定义统一**：明确delay/watch_time>0.5 vs (watch-delay)/watch>0.5 |
| **审计4** | **样本数对齐**：72,646(EDA) vs 77,824(delay)差异来源 |
| 验收 | 审计通过→DG2结论成立；审计失败→重做延迟分析 |

### MVP-1.2-pseudo: 伪在线截断验证 🟡 P1
| 项 | 配置 |
|----|------|
| 目标 | 验证Chapelle在真实延迟场景（有未成熟标签）下的价值 |
| 依赖 | MVP-1.2-audit 审计通过 |
| 关闭 | DG2.2 |
| **实验设计** | 设截止时刻cut（按天滚动），cut之后gift暂当0，用成熟标签在test评估 |
| **对比** | Baseline(不校正) vs Chapelle vs 生存模型 |
| **混合分布建模** | $F(t)=\pi+(1-\pi)F_{Weibull}(t)$，π为0延迟占比 |
| 评估 | ECE/Brier/Top-K排序指标 |
| 验收 | If ECE改善≥0.02→有价值；Else→关闭延迟校正 |

## Phase 2: 决策层

### MVP-2.1: 凹收益分配层
| 项 | 配置 |
|----|------|
| 目标 | 验证凹收益分配优于贪心 |
| 环境 | Simulator (MVP-0.3) |
| 方法 | 贪心: argmax v_{u,s}; 凹收益: argmax g'(V_s)·v_{u,s} |
| 凹函数 | g(V)=log(1+V) 或 g(V)=B(1-e^{-V/B}) |
| 评估 | 总收益, Gini系数(主播侧), 头部占比 |
| 验收 | 总收益提升, Gini改善(更公平) |

**Gate影响**: 
- 总收益Δ≥10% + Gini改善 → 确认分配层设计
- Δ<10% → 简化为贪心+硬约束

### MVP-2.2: 冷启动/公平约束
| 项 | 配置 |
|----|------|
| 目标 | 验证约束对生态健康的影响 |
| 方法 | 拉格朗日: max Σg(V) + λΣ(探索流量约束) |
| 约束 | 新主播最低曝光量; 头部主播上限 |
| 评估 | 冷启动成功率, 长期留存代理 |
| 验收 | 总收益损失<5%情况下，新主播成功率提升 |

## Phase 3: 评估层

### MVP-3.1: OPE验证
| 项 | 配置 |
|----|------|
| 目标 | 验证OPE在高方差场景的有效性 |
| 方法 | IPS, SNIPS, Doubly Robust |
| 数据 | Simulator生成的日志(含propensity) |
| 评估 | OPE估计值 vs 真实值的bias和variance |
| 验收 | DR方法误差可控，可用于策略对比 |

---

# 4. 📊 进度追踪

## 4.1 看板

```
⏳计划       🔴就绪P0      🔴就绪P1     🚀运行    ⚠️需审计     ✅完成
                                                            MVP-0.1
                                                            MVP-0.2
                                                            MVP-0.3 ✅
                                                            MVP-1.1
                                                            MVP-1.1-fair
                                                            MVP-2.1 ✅
                                                            MVP-2.2 ✅
                                                  MVP-1.2
             MVP-1.4
             MVP-1.2-audit
                          MVP-1.5
                          MVP-1.2-pseudo
MVP-3.1
                                                            MVP-1.3
```

### 优先级说明
- **P0 (立即执行)**: MVP-1.4 诊断拆解, MVP-1.2-audit 数据审计
- **P1 (依赖P0)**: MVP-1.5 Stage2改进, MVP-1.2-pseudo 伪在线验证
- **P2 (Simulator)**: MVP-0.3 构建后做变量控制实验

## 4.2 Gate进度

| Gate | MVP | 状态 | 结果 |
|------|-----|------|------|
| Gate-1 | MVP-1.1~1.5, 1.2-audit/pseudo | 🔄 进行中 | ✅ DG1关闭; 🔆 DG1.1诊断中; ⚠️ DG2暂缓; 🔆 DG2.1审计中 |
| Gate-2 | MVP-2.1,2.2 | ✅ **完成** | 凹收益无优势(Δ=-1.17%)；公平性改善(Gini -0.018)；决策：**Greedy+约束** |
| Gate-3 | MVP-3.1 | ⏳ | - |

### 子Gate详情
| Sub-Gate | MVP | 状态 | 结果 |
|----------|-----|------|------|
| DG1.1 | MVP-1.4, 1.5 | ✅ 完成 | 主因是Stage1分类不足，推荐召回-精排分工 |
| DG2.1 | MVP-1.2-audit | ✅ 通过 | 发现代码bug(单位错误)并修复；pct_late_50=13.3% |
| ~~DG2.2~~ | ~~MVP-1.2-pseudo~~ | ❌ 取消 | 延迟问题不存在(86.3%立即发生)，无需验证 |
| DG6 | MVP-0.3(扩展) | ⏳ | Simulator变量控制产出相图 |

## 4.3 结论快照

| MVP | 结论 | 关键指标 | 同步Hub |
|-----|------|---------|---------|
| 0.1 | 打赏金额极端重尾(Gini=0.94)，Top 1%用户贡献60%收益，确认两段式建模必要性 | User Gini=0.942, Streamer Gini=0.930, Matrix Density=0.0064% | ✅ §6.3 |
| 0.2 | Baseline LightGBM 直接回归表现超预期，Top-1%=56.2%远超30%基准，交互特征主导 | MAE(log)=0.263, Top-1%=56.2%, Spearman=0.891 | ✅ §6.3 |
| 1.1 | ⚠️ 两段式与Baseline不可直接对比（数据集不同），Stage1分类有效(PR-AUC=0.65) | PR-AUC=0.646, ECE=0.018 | ✅ |
| **1.1-fair** | ❌ **公平对比：Direct Reg 大幅胜出**，但Two-Stage在NDCG@100更优(+14.2pp) | Direct Top-1%=54.5%, Two-Stage=35.7%, NDCG: 35.9% vs 21.7% | ✅ |
| **1.2** | ❌ **DG2关闭**：延迟中位数=0s(86.3%立即发生)，ECE改善=-0.010(变差)；代码bug已修复 | pct_late_50=13.3%, ECE改善=-0.010 | ✅ |
| **1.2-audit** | ✅ **审计通过**：发现并修复代码bug(单位错误)；样本膨胀1.14x正常 | DG2.1关闭，延迟问题不存在 | ✅ |
| **0.3** | ✅ **Simulator V1 完成**：Gini误差<5%可校准，金额分布误差大；Greedy策略收益最高(3x Random) | User Gini误差=4.9%, Greedy/Random=2.94x | ✅ §Q3.2 |
| **1.4** | ✅ **诊断完成**：主因=Stage1分类不足，Stage2在gift子集表现好(Spearman=0.89>0.74) | Oracle_p增益+54.9pp，推荐召回-精排分工 | ✅ DG1.1 |
| **2.1** | ❌ **DG3未通过**：凹收益无显著优势(Δ=-1.17%)，但公平性改善(Gini -0.018) | Concave Exp Δ=-1.17%, Δ Gini=-0.018 | ✅ §DG3 |
| **2.2** | ✅ **软约束冷启动显著**：探索发现高潜力新主播，同时提升收益(+32%)和成功率(+263%) | Revenue +32%, Success +263%, 决策：采用约束 | ✅ §Q2.3 |
| **Gate-2** | ✅ **分配层验证完成**：凹收益简化为Greedy；软约束冷启动采用；推荐策略：**Greedy+Soft Cold-Start (λ=0.5)** | - | ✅ §Gate-2 |
| **1.3** | ❌ **多任务学习无优势**：密集信号(watch/comment)未能提升稀疏打赏预测，DG5关闭 | Multi PR-AUC=0.165 < Single=0.182 (Δ=-1.76pp) | ✅ §DG5 |

## 4.4 时间线

| 日期 | 事件 |
|------|------|
| 2026-01-08 | 项目立项 |
| 2026-01-08 | MVP-0.1 完成：KuaiLive 数据探索 |
| 2026-01-08 | MVP-0.2 完成：Baseline LightGBM 回归 (Top-1%=56.2%, Spearman=0.891) |
| 2026-01-08 | MVP-1.1 完成：两段式建模，揭示与Baseline不可直接对比 |
| 2026-01-08 | MVP-1.1-fair 立项：公平对比实验 (P0) |
| 2026-01-08 | MVP-1.2 立项：延迟反馈建模 (P1) |
| 2026-01-08 | **MVP-1.1-fair 完成**：Direct Reg 胜出 (Top-1%=54.5% vs 35.7%)，DG1 关闭 |
| 2026-01-08 | MVP-1.2 完成：初步结论延迟中位数=0s |
| 2026-01-08 | **问题拆解**：发现数据红旗🚩，DG2暂缓关闭，需先审计 |
| 2026-01-08 | **新增 DG1.1/DG2.1/DG2.2/DG6**：诊断Two-Stage主因 + 验证延迟数据 + Simulator相图 |
| 2026-01-08 | **MVP-1.4/1.2-audit 立项 (P0)**：Two-Stage诊断拆解 + 延迟数据审计 |
| 2026-01-08 | **MVP-1.5 立项 (P1)**：Stage2改进+召回-精排分工 |
| 2026-01-08 | **MVP-1.4 完成**：Two-Stage诊断，主因=Stage1分类不足，推荐召回-精排分工 |
| 2026-01-08 | **MVP-1.2-audit 完成**：发现代码bug(单位错误)并修复，pct_late_50=13.3%，DG2.1通过 |
| 2026-01-08 | **DG2 关闭确认**：审计后确认86.3%礼物立即发生，延迟校正无价值 |
| 2026-01-08 | **MVP-1.2 完成**：延迟中位数=0s，ECE改善=-0.010(变差)，**DG2关闭** |
| 2026-01-08 | **MVP-1.3 立项**：多任务学习，用密集信号扶起稀疏打赏，关闭 DG5 |
| 2026-01-08 | **MVP-1.3 完成**：❌ 多任务未提升，Δ PR-AUC=-1.76pp，**DG5 关闭** |
| 2026-01-08 | **MVP-0.3, 2.1, 2.2, 3.1 Coding Prompt 生成**：可并行执行 MVP-0.3 + MVP-1.3 |
| 2026-01-08 | **MVP-0.3 完成**：Simulator V1 Gini误差<5%，Greedy策略收益3x Random |
| 2026-01-08 | **MVP-2.1 完成**：凹收益Δ=-1.17%无显著优势，公平性改善(Gini -0.018)，DG3未通过 |
| 2026-01-08 | **MVP-2.2 完成**：软约束冷启动 +32%收益 +263%成功率，决策：采用约束 |
| 2026-01-08 | **Gate-2 关闭**：分配层验证完成，推荐策略 Greedy+Soft Cold-Start (λ=0.5) |

---

# 5. 🔗 文件索引

## 5.1 实验索引

| exp_id | topic | 状态 | MVP |
|--------|-------|------|-----|
| EXP-20260108-gift-allocation-01 | KuaiLive EDA | ✅ | 0.1 |
| EXP-20260108-gift-allocation-02 | Baseline LightGBM | ✅ | 0.2 |
| EXP-20260108-gift-allocation-03 | Two-Stage Model | ✅ | 1.1 |
| EXP-20260108-gift-allocation-04 | Fair Comparison | ✅ | 1.1-fair |
| EXP-20260108-gift-allocation-05 | Delay Modeling | ✅ | 1.2 |
| EXP-20260108-gift-allocation-13 | Delay Audit | ✅ | 1.2-audit |
| EXP-20260108-gift-allocation-06 | Multi-Task Learning | ✅ | 1.3 |
| EXP-20260108-gift-allocation-07 | Simulator V1 | ✅ | 0.3 |
| EXP-20260108-gift-allocation-08 | Concave Allocation | ✅ | 2.1 |
| EXP-20260108-gift-allocation-09 | Coldstart Constraint | ⏳ | 2.2 |
| EXP-20260108-gift-allocation-10 | OPE Validation | ⏳ | 3.1 |
| **EXP-20260108-gift-allocation-11** | **Two-Stage Diagnosis** | ✅ | **1.4** |
| **EXP-20260108-gift-allocation-12** | **Stage2 Improve + Pipeline** | ⏳ | **1.5** |
| **EXP-20260108-gift-allocation-13** | **Delay Audit** | 🔴 | **1.2-audit** |
| **EXP-20260108-gift-allocation-14** | **Delay Pseudo-Online** | ⏳ | **1.2-pseudo** |

## 5.2 数据源

| 数据集 | 来源 | 字段 | 用途 |
|--------|------|------|------|
| KuaiLive | [Zenodo](https://imgkkk574.github.io/KuaiLive) | click/comment/like/gift, watch_time, gift_price | 主实验数据 |
| VTuber 1B | [GitHub/Kaggle](https://github.com/sigvt/vtuber-livechat-dataset) | live_chat, SuperChat, moderation | 补充时序+打赏 |

## 5.3 参考文献

| 主题 | 论文 | 要点 |
|------|------|------|
| 稀疏分类 | Focal Loss (Lin et al., 2017) | 处理极度不均衡 |
| 延迟反馈 | Delayed Conversion (Chapelle, 2014) | 生存分析思路 |
| 凹收益分配 | Online Matching with Concave Returns (Devanur et al.) | 边际递减分配 |
| Bandit延迟 | Neural Contextual Bandits Under Delayed Feedback (2025) | 延迟奖励bandits |
| OPE | Doubly Robust (Dudík et al., 2011) | 减小方差 |

---

# 6. 📎 附录

## 6.1 数值汇总

| MVP | 配置 | PR-AUC | Top-1%捕获 | Spearman | NDCG@100 | MAE(log) |
|-----|------|--------|-----------|----------|----------|----------|
| 0.1 | KuaiLive EDA | - | - | - | - | Gini=0.94 |
| 0.2 | Baseline LightGBM (gift-only) | - | 56.2% | 0.891 | 0.716 | 0.263 |
| 1.1 | Two-Stage (click全量) | 0.646 (S1) | ⚠️不可对比 | - | - | - |
| **1.1-fair** | **Direct Reg (click全量)** | - | **54.5%** | **0.331** | 0.217 | **0.044** |
| 1.1-fair | Two-Stage V2 (click全量) | 0.991 (S1) | 35.7% | 0.246 | 0.359 | 0.081 |
| **1.2** | **Baseline (binary)** | **0.651** | - | - | - | ECE=**0.018** |
| 1.2 | Chapelle (weighted) | 0.692 | - | - | - | ECE=0.028 ❌ |

> **结论**: 
> - Direct Regression 在 Top-1% Capture 上胜出 (+18.8pp)，但 Two-Stage 在 NDCG@100 上更优 (+14.2pp)
> - 延迟中位数=0s，Chapelle方法ECE变差(-0.010)，DG2关闭

## 6.2 Simulator参数参考

```yaml
# 用户财富分布
wealth:
  type: mixture
  components:
    - dist: lognormal
      mean: 3.0
      std: 1.0
      weight: 0.95
    - dist: pareto
      alpha: 1.5
      min: 100
      weight: 0.05

# 打赏概率
gift_prob:
  intercept: -5.0  # 使基础打赏率很低
  wealth_coef: 0.5
  match_coef: 1.0
  engage_coef: 0.3
  crowd_coef: -0.1  # 边际递减

# 打赏金额
gift_amount:
  intercept: 2.0
  wealth_coef: 0.8
  match_coef: 0.5
  noise_std: 0.5

# 延迟分布
delay:
  dist: weibull
  shape: 1.5
  scale: 10.0  # 分钟

# 观看时长
watch_duration:
  dist: lognormal
  mean: 2.0
  std: 1.0
```

## 6.3 更新日志

| 日期 | 变更 | 章节 |
|------|------|------|
| 2026-01-08 | 创建Roadmap | 全部 |
| 2026-01-08 | MVP-1.1 Coding Prompt 生成 | §2.1, §4.1 |
| 2026-01-08 | MVP-1.1 完成：两段式模型实验，发现与Baseline不可直接对比 | §2.1, §4.1, §4.3 |
| 2026-01-08 | MVP-1.1-fair 完成：Direct Reg 胜出 (54.5% vs 35.7%)，DG1 关闭 | §2.1, §4.3, §6.1 |
| 2026-01-08 | MVP-1.1-fair V2 验证：修复 Stage1 训练后结论一致 | §4.3 |
| 2026-01-08 | MVP-1.3 立项：多任务学习 Coding Prompt 生成 | §2.1, §4.1, §5.1 |
| 2026-01-08 | **MVP-1.3 完成**：多任务无优势，Δ PR-AUC=-1.76pp，DG5 关闭 | §2.1, §4.3, §5.1 |
| 2026-01-08 | **批量生成 Coding Prompt**: MVP-0.3, 2.1, 2.2, 3.1 | §2.1, §4.1, §5.1 |
| 2026-01-08 | **MVP-0.3 完成**：Simulator V1，Gini误差<5%，Greedy策略收益3x | §2.1, §4.3, §5.1 |
| 2026-01-08 | **问题拆解+新假设**: 添加H1-H4/DG1.1/DG2.1/DG2.2/DG6, 发现数据红旗 | §1.2, §2.1, §3, §4, §5.1 |
| 2026-01-08 | **MVP-1.4 立项**: Two-Stage诊断拆解 (P0) | §2.1, §3, §5.1 |
| 2026-01-08 | **MVP-1.2-audit 立项**: 延迟数据审计 (P0) | §2.1, §3, §5.1 |
| 2026-01-08 | **MVP-1.5, 1.2-pseudo 立项**: Stage2改进+伪在线验证 (P1) | §2.1, §3, §5.1 |
