# 🗺️ Gift Allocation Roadmap

> **Topic:** gift_allocation | **Phase:** 5 (分配优化)  
> **Author:** Viska Wei | **Date:** 2026-01-09 | **Status:** 🔄 规划中

```
💡 Phase 5 目标：从"预测更准"转向"分配更对"
Gate：把分配问题当成"带约束、带外部性、带风险"的在线资源分配/匹配问题
```

---

## 🔗 Related Files

| Type | File |
|------|------|
| 🧠 Hub | [`./gift_allocation_hub.md`](./gift_allocation_hub.md) |
| 📋 Kanban | [`status/kanban.md`](../status/kanban.md) |
| 📗 Experiments | [`exp/`](./exp/) |
| 📊 Results | [`results/`](./results/) |

---

# 1. 🚦 Decision Gates

## 1.1 战略路线 (来自Hub)

| Route | 名称 | Hub推荐 | 验证Gate |
|-------|------|---------|----------|
| A | 继续优化估计层 | 🔴 降优先级 | - |
| **B** | **分配层优化** | 🟢 **推荐** | Gate-5A/5B/5C/5D |

**判断依据**：
- 估计层已相对够用（Direct Top-1%=54.5%）
- 分配策略是最大杠杆（Greedy 3x Random，冷启动+32%）
- 资源约束 + 生态目标才是上线收益与风险的核心

## 1.2 Gate定义

### Gate-5A: 召回-精排分工

| 项 | 内容 |
|----|------|
| 验证 | 分工架构能否同时保住Top-1%和提升NDCG |
| MVP | MVP-5.1 |
| 若PASS | Top-1%≥56% & NDCG@100↑ → 采用分工架构 |
| 若FAIL | Top-1%下降 → 保留Direct-only |
| 状态 | ❌ **FAIL** |
| 结论 | Recall Coverage仅5.5%@Top-1000，架构不可行。保留Direct-only |

### Gate-5B: 影子价格/供需匹配

| 项 | 内容 |
|----|------|
| 验证 | 影子价格能否统一处理多约束且提升收益 |
| MVP | MVP-5.2 |
| 若PASS | 收益≥+5% vs Greedy & 约束满足 → 替换Greedy+规则 |
| 若FAIL | 收益无提升 → 保留Greedy+软约束 |
| 状态 | ❌ **FAIL** |
| 结论 | 收益仅+2.74%(目标+5%)，容量满足率82.8%(目标>90%)。Greedy+Rules更优(+4.36%)。保留现有方案 |

### Gate-5C: 鲸鱼分散

| 项 | 内容 |
|----|------|
| 验证 | 鲸鱼b-matching能否降低生态集中度 |
| MVP | MVP-5.3 |
| 若PASS | 超载率<10% & Gini↓ → 鲸鱼单独匹配层 |
| 若FAIL | 无改善 → 统一分配 |
| 状态 | ⏳ 待验证 |

### Gate-5D: 风险控制

| 项 | 内容 |
|----|------|
| 验证 | 不确定性排序能否降低策略波动 |
| MVP | MVP-5.4 |
| 若PASS | CVaR@5%改善 & 波动↓ → 采用UCB/LCB |
| 若FAIL | 无改善 → 纯EV排序 |
| 状态 | ⏳ 待验证 |

## 1.3 本周重点

| 优先级 | MVP | Gate | 状态 |
|--------|-----|------|------|
| 🔴 P0 | MVP-5.1 召回-精排分工 | Gate-5A | ❌ FAIL (Top-1%=54.3%<56%) |
| 🔴 P0 | MVP-5.2 影子价格 | Gate-5B | ❌ FAIL (收益+2.74%<5%) |
| 🟡 P1 | MVP-5.3 鲸鱼分散 | Gate-5C | 🔴 就绪 |

---

# 2. 📋 MVP列表

## 2.1 总览

| MVP | 名称 | Phase | Gate | 状态 | exp_id | 报告 |
|-----|------|-------|------|------|--------|------|
| 0.1-Enhanced | 全方位 EDA (Comprehensive) | 0 | - | 🔄 | EXP-20260109-kuailive-02 | [已迁移至 KuaiLive](../KuaiLive/exp/exp_kuailive_eda_comprehensive_20260109.md) |
| 5.1 | 召回-精排分工 | 5 | Gate-5A | ❌ FAIL | EXP-20260109-gift-allocation-51 | [exp_recall_rerank_20260109.md](exp/exp_recall_rerank_20260109.md) |
| 5.2 | 影子价格/供需匹配 | 5 | Gate-5B | ❌ FAIL | EXP-20260109-gift-allocation-16 | [exp_shadow_price_20260109.md](exp/exp_shadow_price_20260109.md) |
| 5.3 | 鲸鱼分散 (b-matching) | 5 | Gate-5C | ⏳ | EXP-20260109-gift-allocation-53 | [exp_whale_matching_20260109.md](exp/exp_whale_matching_20260109.md) |
| 5.4 | 风险控制 (UCB/CVaR) | 5 | Gate-5D | ⏳ | - | - |
| 5.5 | 多目标生态调度 | 5 | - | ⏳ | - | - |

**状态**: ⏳计划 | 🔴就绪 | 🚀运行 | ✅完成 | ❌取消

## 2.2 配置速查

| MVP | 数据 | 模型/策略 | 关键变量 | 验收指标 |
|-----|------|----------|---------|---------|
| 0.1-Enhanced | click/gift/user/streamer全量 | EDA分析 | Session构建、多维度分析 | 至少12张Hero图，可行动洞见 |
| 5.1 | click全量 | Direct召回 + Stage2精排 | Top-M候选数 | Top-1%≥56%, NDCG@100↑ |
| 5.2 | Simulator V2 | Primal-Dual分配 | λ_c约束价格 | 收益+5%, 约束满足 |
| 5.3 | Simulator V2 | b-matching/min-cost flow | k鲸鱼上限 | 超载率<10%, Gini↓ |
| 5.4 | Simulator V2 | UCB/LCB/CVaR | 置信界参数 | CVaR@5%改善 |
| 5.5 | Simulator V2 | 多目标加权 | α权重扫描 | Pareto前沿 |

---

# 3. 🔧 MVP规格

## Phase 0: 数据探索（增强版）

### MVP-0.1-Enhanced: 全方位 EDA (Comprehensive)

| 项 | 配置 |
|----|------|
| **目标** | 对 KuaiLive 数据做"全方位、专业、可落地"的探索性数据分析，从用户行为、直播间供给、互动结构、时序规律、稀疏性/冷启动、异常与数据质量等多个维度挖出可行动的洞见 |
| **动机** | 现有 EDA (MVP-0.1) 仅覆盖金额分布、集中度、稀疏性、时间模式，缺少 session 级分析、用户分群、供给侧质量、交互结构、异常检测等关键维度 |
| **数据** | click.csv (4.9M), gift.csv (72K), user.csv (23K), streamer.csv (452K), room.csv (11.8M), comment.csv (196K), like.csv (179K), negative.csv (12.7M) |
| **核心任务** | 1. Session 构建与转化漏斗分析<br>2. 用户二维象限（观看 vs 付费）<br>3. 主播二维象限（观看 vs 变现）<br>4. 用户-主播匹配结构（专一度、多样性）<br>5. 时序深度分析（7×24 热力图、峰值分解）<br>6. 异常检测（作弊/数据污染）<br>7. 数据质量评分 |
| **关键输出** | 至少 12 张 Hero 图（每张图必须能"一眼得到结论"）+ 可执行的建模/产品建议 |
| **验收** | 1. 所有"待执行"部分填充完成<br>2. 至少 12 张高质量图表<br>3. 至少 5 条可行动洞见<br>4. 数据质量评分表完整<br>5. 后续字段需求清单明确 |
| **产出** | `../KuaiLive/exp/exp_kuailive_eda_comprehensive_20260109.md`, 图表文件 `../KuaiLive/img/*.png`, 中间表 `../KuaiLive/results/sessions_*.csv` |

**关键研究问题**：
- H2.1: 打赏是"即时冲动"还是"随观看累积"？（首礼时间分布）
- H2.2: 高观看低付费人群是否存在？规模多大？
- H2.3: 主播供给是否存在"高吸引低变现"类型？
- H2.4: 付费是否强绑定于特定主播？（用户专一度）
- H2.5: 数据是否存在异常/作弊/污染？

**图表生成任务**: 28 个任务（至少 12 个 P0 优先级），详见实验报告 §10.3

---

## Phase 5: 分配优化

### MVP-5.1: 召回-精排分工

| 项 | 配置 |
|----|------|
| **目标** | 验证Direct召回+Two-Stage精排能否兼顾Top-1%和NDCG |
| **动机** | Direct全量排序强(Top-1%=54.5%), Two-Stage在NDCG@100上更好(+14pp), Stage2在gift子集排序强(Spearman=0.89) |
| **数据** | click全量（训练）, gift子集（精排评估） |
| **流程** | 1. Direct打分全量候选 → 2. 取Top-M候选 → 3. Stage2精排 → 4. 分配 |
| **变量** | Top-M ∈ {50, 100, 200, 500} |
| **验收** | Top-1% ≥ 56% (不低于Direct), NDCG@100 > 21.7% (Direct基线) |
| **产出** | `exp_recall_rerank_YYYYMMDD.md`, `scripts/train_recall_rerank.py` |

**Gate影响**: 
- Top-1%≥56% & NDCG↑ → Gate-5A PASS → 采用分工架构
- Top-1%下降 → 保留Direct-only

---

### MVP-5.2: 影子价格/供需匹配

| 项 | 配置 |
|----|------|
| **目标** | 用Primal-Dual框架统一处理多约束分配 |
| **动机** | 当前难点：稀缺大哥资源 + 主播承接上限 + 生态约束，适合用影子价格统一框架 |
| **核心公式** | $\text{score}(u,s) = \widehat{EV}(u,s) - \sum_c \lambda_c \cdot \Delta\text{violation}_c(u \to s)$ |
| **约束类型** | 并发容量、冷启动覆盖、头部占比上限、鲸鱼/大哥互斥、频控 |
| **更新规则** | $\lambda_c \gets \lambda_c + \eta \cdot (\text{violation}_c - \text{threshold}_c)$ (超载涨价，不超载降价) |
| **数据** | Simulator V2（含容量外部性） |
| **变量** | 学习率η, 初始λ, 约束阈值 |
| **验收** | 收益 ≥ Greedy+5%, 各约束满足率>90% |
| **产出** | `exp_shadow_price_YYYYMMDD.md`, `scripts/simulator/policies_shadow_price.py` |

**影子价格分配器接口**:

```
输入:
  - candidates: List[(user_id, streamer_id, EV)]  # 候选及预估收益
  - capacity: Dict[streamer_id, int]              # 主播容量上限
  - coldstart_set: Set[streamer_id]               # 冷启动主播集合
  - whale_set: Set[user_id]                       # 鲸鱼用户集合
  - constraints: Dict[str, float]                 # 各约束阈值

输出:
  - allocations: List[(user_id, streamer_id)]     # 最终分配
  - lambda_values: Dict[str, float]               # 学习到的影子价格
  - metrics: Dict[str, float]                     # 收益/约束满足度
```

**Gate影响**: 
- 收益≥+5% & 约束满足 → Gate-5B PASS → 替换Greedy+规则
- 否则 → 保留Greedy+软约束

---

### MVP-5.3: 鲸鱼分散 (b-matching)

| 项 | 配置 |
|----|------|
| **目标** | 分散Top-0.1%/1%鲸鱼，避免过度集中伤生态 |
| **动机** | 鲸鱼同时涌入同一主播会导致：承接饱和、马太效应加剧、用户体验下降 |
| **策略** | 鲸鱼单独做匹配层，每个主播设置"同时最多承接k个鲸鱼" |
| **算法** | b-matching / min-cost flow / greedy with swaps |
| **数据** | Simulator V2 |
| **变量** | k ∈ {1, 2, 3, 5}, 鲸鱼定义阈值 (Top 0.1%, 1%, 5%) |
| **验收** | 超载率<10%, Streamer Gini↓, 收益不显著下降(<5%) |
| **产出** | `exp_whale_matching_YYYYMMDD.md`, `scripts/simulator/policies_whale_matching.py` |

**分层匹配策略**:
1. **鲸鱼层**: 先做"互斥 + 分散 + 价值最大化"
2. **普通层**: 再填充容量剩余，走轻量greedy/softmax

---

### MVP-5.4: 风险控制 (UCB/CVaR)

| 项 | 配置 |
|----|------|
| **目标** | 引入不确定性量化，降低策略波动和尾部风险 |
| **动机** | 重尾+高方差 → 均值EV偏差被极端样本放大 → 线上波动大、策略不稳定 |
| **方法选项** | UCB（探索倾向）, LCB（保守上线）, CVaR@5%（尾部保护） |
| **公式** | UCB: $\text{score} = \hat{\mu} + \beta \cdot \hat{\sigma}$ / LCB: $\text{score} = \hat{\mu} - \beta \cdot \hat{\sigma}$ |
| **数据** | Simulator V2 + Ensemble/分位数回归 |
| **变量** | β ∈ {0.5, 1.0, 2.0}, CVaR分位数 ∈ {1%, 5%, 10%} |
| **验收** | CVaR@5%改善>10%, 跨seed波动↓, 收益不显著下降 |
| **产出** | `exp_risk_control_YYYYMMDD.md`, `scripts/simulator/policies_risk.py` |

---

### MVP-5.5: 多目标生态调度

| 项 | 配置 |
|----|------|
| **目标** | 构建Pareto前沿，显式化生态权衡，为上线护栏定阈值 |
| **动机** | 只优化Revenue会让生态指标以事故形式出现 |
| **目标函数** | $\max \alpha_1 \cdot \text{Revenue} + \alpha_2 \cdot \text{Retention} + \alpha_3 \cdot \text{Diversity} - \alpha_4 \cdot \text{Gini} - \alpha_5 \cdot \text{Overload}$ |
| **代理指标** | 用户侧: 停留/回访proxy; 主播侧: 曝光覆盖/冷启动成功率/Gini; 系统侧: 超载率 |
| **数据** | Simulator V2 |
| **变量** | α权重网格扫描 |
| **验收** | 可解释的Pareto前沿曲线，识别关键权衡点 |
| **产出** | `exp_multiobjective_YYYYMMDD.md`, Pareto图 |

---

# 4. 📊 评估协议

> 每个新策略都输出同一组指标，形成可比的评估体系

## 4.1 指标体系

| 类别 | 指标 | 说明 | 权重 |
|------|------|------|------|
| **收益类** | Total Revenue | 总收益 | 核心 |
| | Revenue/User | 人均收益 | 效率 |
| | Top-0.1% Capture | 鲸鱼捕获率 | 头部价值 |
| | Top-1% Capture | 高价值用户捕获 | 头部价值 |
| **生态类** | Streamer Gini | 主播收益集中度 | 公平性 |
| | Top主播份额 | 头部集中度 | 马太效应 |
| | 冷启动成功率 | 新主播激活 | 生态健康 |
| | 冷启动覆盖率 | 新主播曝光 | 生态健康 |
| **约束类** | 超载率 | 容量违约比例 | 可行性 |
| | 拥挤率 | 高并发场景占比 | 外部性 |
| | 容量违约次数 | 硬约束违反 | 可行性 |
| **风险类** | CVaR@5% | 最差5%情况收益 | 尾部保护 |
| | 收益方差 | 策略稳定性 | 波动 |
| | 跨seed波动 | 不同随机种子差异 | 鲁棒性 |
| **体验代理** | 停留proxy | 用户停留时长代理 | 留存 |
| | 回访proxy | 次日回访代理 | 留存 |

## 4.2 评估流程

```
策略 → Simulator V2 (真值对比) → SNIPS OPE (上线化演练)
                ↓                        ↓
          获得绝对指标              获得相对排序可信度
                ↓                        ↓
              两者一致 → 高置信度可上线
```

---

# 5. 📚 策略库

> 可组合的分配机制，按落地难度分组

## A. 规则增强型（最容易落地）

| # | 策略 | 描述 | 适用场景 |
|---|------|------|---------|
| A1 | 频控+冷却 | per-user & per-streamer短时间限制 | 上线初期护栏 |
| A2 | 头部曝光cap | 头部超标时提高惩罚/价格 | 生态保护 |
| A3 | 拥挤动态降权 | $\widehat{EV} \times g(\text{load}/C_s)$ | 容量管理 |

## B. 优化/匹配型（Phase 5重点）

| # | 策略 | 描述 | 适用场景 |
|---|------|------|---------|
| B1 | Primal-Dual影子价格 | 多约束统一框架 | MVP-5.2 |
| B2 | 在线二分图匹配 | 主播容量为右侧约束 | 实时分配 |
| B3 | 分层匹配 | 鲸鱼层+普通层分开 | MVP-5.3 |

## C. Bandit/探索型

| # | 策略 | 描述 | 适用场景 |
|---|------|------|---------|
| C1 | 约束Contextual Bandit | Lagrangian Bandit | 在线学习 |
| C2 | 分段探索 | 探索预算集中在冷启动/不确定性高 | MVP-5.4 |

## D. 风险/鲁棒型

| # | 策略 | 描述 | 适用场景 |
|---|------|------|---------|
| D1 | CVaR/Worst-case | 最差分位数优化 | 上线初期 |
| D2 | DRO/人群鲁棒 | 分人群约束最低收益 | 避免只讨好鲸鱼 |

## E. 列表/多样性型

| # | 策略 | 描述 | 适用场景 |
|---|------|------|---------|
| E1 | Submodular/DPP | 列表多样性重排 | Feed推荐 |
| E2 | 探索位固定 | 每个列表固定1个探索位 | 生态控制 |

---

# 6. 📊 进度追踪

## 6.1 看板

```
⏳计划    🔴就绪    🚀运行    ❌失败      ✅完成
MVP-5.3                       MVP-5.1     MVP-4.1+
MVP-5.4                       MVP-5.2     MVP-4.2
MVP-5.5                                   (Phase 0-3)
```

## 6.2 Gate进度

| Gate | MVP | 状态 | 结果 |
|------|-----|------|------|
| Gate-5A 召回精排 | MVP-5.1 | ❌ FAIL | Recall Coverage仅5.5%，保留Direct-only |
| Gate-5B 影子价格 | MVP-5.2 | ❌ FAIL | 收益+2.74%<5%, 容量82.8%<90%, 保留Greedy+Rules |
| Gate-5C 鲸鱼分散 | MVP-5.3 | ⏳ | - |
| Gate-5D 风险控制 | MVP-5.4 | ⏳ | - |

## 6.3 结论快照

| MVP | 结论 | 关键指标 | 同步Hub |
|-----|------|---------|---------|
| 4.1+ | V2+金额校准通过 | P50=0%, P90=13% | ✅ K |
| 4.2 | 并发容量建模有效 | 边际递减24.4% | ✅ K10 |
| 5.1 | ❌ 召回-精排架构失败 | Recall Coverage仅5.5%@Top-1000 | ✅ DG6关闭 |
| 5.2 | ❌ 影子价格框架失败 | 收益+2.74%<5%，容量82.8%<90% | ✅ DG7关闭，I27/I28添加 |

## 6.4 时间线

| 日期 | 事件 |
|------|------|
| 2026-01-09 | Phase 5 规划完成 |
| 2026-01-09 | 战略转向：从预测优化转向分配优化 |
| 2026-01-09 | MVP-5.1 召回-精排分工 ❌ FAIL，Gate-5A 关闭 |
| 2026-01-09 | MVP-5.2 影子价格 ❌ FAIL，Gate-5B 关闭，保留Greedy+Rules |

---

# 7. 📎 附录

## 7.1 日志字段需求（为未来线上准备）

| 字段 | 用途 | MVP |
|------|------|-----|
| `propensity` | OPE评估 | 所有策略 |
| `capacity_at_alloc` | 容量状态 | MVP-5.2 |
| `is_whale` | 鲸鱼标识 | MVP-5.3 |
| `lambda_values` | 影子价格快照 | MVP-5.2 |
| `uncertainty_score` | 不确定性 | MVP-5.4 |
| `constraint_violations` | 约束违反 | MVP-5.2/5.3 |

## 7.2 文件索引

| 类型 | 路径 |
|------|------|
| Roadmap | `../gift_EVpred/gift_EVpred_roadmap.md` |
| Hub | `../gift_EVpred/gift_EVpred_hub.md` |
| 图表 | `img/` |
| 结果 | `results/` |

## 7.3 更新日志

| 日期 | 变更 | 章节 |
|------|------|------|
| 2026-01-09 | 创建 Phase 5 Roadmap | 全部 |
| 2026-01-09 | 定义5个MVP规格 | §3 |
| 2026-01-09 | 添加策略库 | §5 |
| 2026-01-09 | 添加评估协议 | §4 |
