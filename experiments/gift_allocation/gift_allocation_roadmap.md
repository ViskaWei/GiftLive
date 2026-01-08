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
| MVP | MVP-1.1, MVP-1.2, MVP-1.3 |
| 若A | PR-AUC提升>5% + ECE改善>0.02 → 确认分层路线 |
| 若B | 提升不显著 → 考虑端到端或更简单方案 |
| 状态 | ⏳ |

### Gate-2: 分配层验证
| 项 | 内容 |
|----|------|
| 验证 | 凹收益分配是否优于贪心分配 |
| MVP | MVP-2.1, MVP-2.2 |
| 若A | 总收益提升>10% + Gini改善 → 确认分配层 |
| 若B | 提升不显著 → 简化为贪心+约束 |
| 状态 | ⏳ |

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
| 🔴 P0 | MVP-1.1 | Gate-1 | 🔴 就绪 (Prompt 已生成) |

---

# 2. 📋 MVP列表

## 2.1 总览

| MVP | 名称 | Phase | Gate | 状态 | exp_id | 报告 |
|-----|------|-------|------|------|--------|------|
| 0.1 | KuaiLive数据探索 | 0 | - | ✅ | EXP-20260108-gift-allocation-01 | [exp_kuailive_eda_20260108.md](./exp/exp_kuailive_eda_20260108.md) |
| 0.2 | Baseline(直接回归) | 0 | - | ✅ | EXP-20260108-gift-allocation-02 | [exp_baseline_20260108.md](./exp/exp_baseline_20260108.md) |
| 0.3 | Simulator V1构建 | 0 | - | ⏳ | - | - |
| 1.1 | 两段式建模 | 1 | Gate-1 | 🔴 | EXP-20260108-gift-allocation-03 | - |
| 1.2 | 延迟反馈建模(生存) | 1 | Gate-1 | ⏳ | - | - |
| 1.3 | 多任务学习 | 1 | Gate-1 | ⏳ | - | - |
| 2.1 | 凹收益分配层 | 2 | Gate-2 | ⏳ | - | - |
| 2.2 | 冷启动/公平约束 | 2 | Gate-2 | ⏳ | - | - |
| 3.1 | OPE验证(IPS/DR) | 3 | Gate-3 | ⏳ | - | - |

**状态**: ⏳计划 | 🔴就绪 | 🚀运行 | ✅完成 | ❌取消

## 2.2 配置速查

| MVP | 数据 | 模型 | 关键变量 |
|-----|------|------|---------|
| 0.1 | KuaiLive | - | 统计分析 |
| 0.2 | KuaiLive | GBDT | 直接回归Y |
| 0.3 | 合成数据 | - | Simulator参数 |
| 1.1 | KuaiLive | GBDT×2 | p(x), m(x) |
| 1.2 | KuaiLive | 生存模型 | hazard h(t|x) |
| 1.3 | KuaiLive | 多任务NN | click/watch/gift |
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

### MVP-1.2: 延迟反馈建模
| 项 | 配置 |
|----|------|
| 目标 | 验证延迟校正/生存建模的增益 |
| 方法A | 生存分析: 估计F(H|x)=Pr(T≤H|x) |
| 方法B | 延迟校正: Chapelle方法，软标签加权 |
| 评估 | 校准误差ECE, 时间窗口敏感性 |
| 验收 | ECE改善, 近期样本预测稳定性 |

### MVP-1.3: 多任务学习
| 项 | 配置 |
|----|------|
| 目标 | 用密集信号(click/watch/comment)扶起稀疏打赏 |
| 模型 | 共享表征f(x) + 多头: CTR, 观看时长, 打赏, 金额 |
| 架构 | 两塔(用户塔+主播塔) + MLP |
| 验收 | 对比单任务，打赏预测PR-AUC提升 |

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
⏳计划    🔴就绪    🚀运行    ✅完成
                              MVP-0.1
                              MVP-0.2
MVP-0.3   MVP-1.1
MVP-1.2
MVP-1.3
MVP-2.1
MVP-2.2
MVP-3.1
```

## 4.2 Gate进度

| Gate | MVP | 状态 | 结果 |
|------|-----|------|------|
| Gate-1 | MVP-1.1,1.2,1.3 | ⏳ | - |
| Gate-2 | MVP-2.1,2.2 | ⏳ | - |
| Gate-3 | MVP-3.1 | ⏳ | - |

## 4.3 结论快照

| MVP | 结论 | 关键指标 | 同步Hub |
|-----|------|---------|---------|
| 0.1 | 打赏金额极端重尾(Gini=0.94)，Top 1%用户贡献60%收益，确认两段式建模必要性 | User Gini=0.942, Streamer Gini=0.930, Matrix Density=0.0064% | ✅ §6.3 |
| 0.2 | Baseline LightGBM 直接回归表现超预期，Top-1%=56.2%远超30%基准，交互特征主导 | MAE(log)=0.263, Top-1%=56.2%, Spearman=0.891 | ✅ §6.3 |

## 4.4 时间线

| 日期 | 事件 |
|------|------|
| 2026-01-08 | 项目立项 |
| 2026-01-08 | MVP-0.1 完成：KuaiLive 数据探索 |
| 2026-01-08 | MVP-0.2 完成：Baseline LightGBM 回归 (Top-1%=56.2%, Spearman=0.891) |

---

# 5. 🔗 文件索引

## 5.1 实验索引

| exp_id | topic | 状态 | MVP |
|--------|-------|------|-----|
| EXP-20260108-gift-allocation-01 | KuaiLive EDA | ✅ | 0.1 |
| EXP-20260108-gift-allocation-02 | Baseline LightGBM | ✅ | 0.2 |

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

| MVP | 配置 | PR-AUC | Top-1%捕获 | 总收益 | Gini |
|-----|------|--------|-----------|--------|------|
| - | - | - | - | - | - |

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
