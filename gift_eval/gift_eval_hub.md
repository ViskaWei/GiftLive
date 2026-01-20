# 🧠 gift_eval Hub
> **ID:** EXP-20260120-gift-eval-hub | **Status:** 🌱探索 |  
> **Date:** 2026-01-20 | **Update:** 2026-01-20  

| # | 💡 共识[抽象洞见] | 证据 | 决策 |
|---|----------------------------|----------------|------|
| K1 | SNIPS 是高方差打赏场景的首选 OPE 方法 | RelErr 0.57%-9.97% | 线上 OPE 使用 SNIPS |
| K2 | 目标策略与行为策略的相似度决定 OPE 难度 | Softmax 0.57% vs Greedy 9.97% | 行为策略需保持足够探索率 |
| K3 | DR 方法在高方差奖励场景效果不如预期 | DR 偏差 > IPS | 暂不使用 DR 作为主方法 |

**🦾 现阶段信念 [≤10条，写"所以呢"]**
- **SNIPS 优于 IPS/DR**：自归一化有效控制权重爆炸 → 推荐 SNIPS 作为主 OPE 方法
- **探索率存在最优点**：过低覆盖不足，过高降低效率 → 行为策略 epsilon ≈ 0.3
- **样本量 ≥ 5000**：小样本 OPE 误差大 → 日志采集需保证量级

**👣 下一步最有价值  [≤2条，直接可进 Roadmap Gate]**
- 🔴 **P0**：线上日志采集系统改造（添加 propensity） → If 可行 then 开始收集日志 else 优先 Simulator
- 🟡 **P1**：OPE + Simulator 联合评估流程 → OPE 粗筛 + Simulator 精筛

> **权威数字（一行即可）**：Best=SNIPS 0.57% (Softmax)；Baseline=IPS 1.72%；Δ=-1.15%；条件=ε=0.3, n=5000

| 模型/方法 | 指标值 | 配置 | 备注 |
|-----------|--------|------|------|
| IPS | 1.72%-15.34% | ε=0.3, n=5000 | 方差大 |
| SNIPS | **0.57%-9.97%** | ε=0.3, n=5000 | ✅ 最佳 |
| DR | 12%-21% | ε=0.3, n=5000 | Q 函数偏差大 |

## 1) 🌲 核心假设树
```
🌲 核心: 如何离线评估推荐/分配策略的期望收益？
│
├── Q1: OPE 方法在高方差场景的有效性
│   ├── Q1.1: IPS 在高方差奖励场景是否可用？ → ⚠️ 部分可用，方差大
│   ├── Q1.2: SNIPS 能否有效控制方差？ → ✅ 验证通过，RelErr < 10%
│   └── Q1.3: DR 是否比 IPS 更稳定？ → ❌ 否，Q 函数偏差主导
│
├── Q2: OPE 的适用条件
│   ├── Q2.1: 最小样本量要求？ → ✅ n ≥ 5000
│   ├── Q2.2: 最优探索率？ → ✅ ε ≈ 0.3
│   └── Q2.3: 策略相似度影响？ → ✅ 越相似越准
│
└── Q3: OPE 与其他评估方法的结合
    ├── Q3.1: OPE 粗筛 + Simulator 精筛？ → ⏳ 待验证
    └── Q3.2: OPE 置信区间估计？ → ⏳ 待验证

Legend: ✅ 已验证 | ❌ 已否定 | 🔆 进行中 | ⏳ 待验证 | 🗑️ 已关闭
```

## 2) 口径冻结（唯一权威）
| 项目 | 规格 |
|---|---|
| Dataset / Version | SimulatorV2+ (500 users × 50 streamers) |
| Behavior Policy | ε-greedy, ε=0.3 |
| Target Policies | Greedy / Softmax / Concave |
| Metric | Relative Error = \|estimate - truth\| / truth |
| Seed / Repeats | 42 / 15-20 repeats |
> 规则：任何口径变更必须写入 §8 变更日志。

---

## 3) 当前答案 & 战略推荐（对齐问题树）

### 3.1 战略推荐（只保留"当前推荐"）
- **推荐路线：Route SNIPS**（一句话理由：自归一化控制方差，跨策略稳定）
- 需要 Roadmap 关闭的 Gate：Gate-1 OPE 可行性（已关闭）

| Route | 一句话定位 | 当前倾向 | 关键理由 | 需要的 Gate |
|---|---|---|---|---|
| Route IPS | 标准重要性采样 | 🔴 | 方差爆炸风险 | - |
| Route DR | 双稳健估计 | 🟡 | Q 函数难估计 | - |
| **Route SNIPS** | 自归一化 IPS | 🟢 推荐 | 稳定、误差 <10% | Gate-1 ✅ |

### 3.2 分支答案表（每行必须回答"所以呢"）
| 分支 | 当前答案（1句话） | 置信度 | 决策含义（So what） | 证据（exp/MVP） |
|---|---|---|---|---|
| OPE 可行性 | SNIPS 可达 <10% RelErr | 🟢 | 可用于策略离线对比 | exp_ope_validation_20260108 |
| 探索率设置 | ε=0.3 最优 | 🟢 | 线上行为策略配置 | exp_ope_validation_20260108 |
| 样本量要求 | n≥5000 | 🟢 | 日志采集目标 | exp_ope_validation_20260108 |

---

## 4) 洞见汇合（多实验 → 共识）
> 只收录"会改变决策"的洞见，建议 5–8 条。

| # | 洞见（标题） | 观察（What） | 解释（Why） | 决策影响（So what） | 证据 |
|---|---|---|---|---|---|
| I1 | SNIPS 控制方差最有效 | RelErr 从 15% 降到 10% | 自归一化消除权重爆炸 | 主 OPE 方法用 SNIPS | exp_ope_validation |
| I2 | 策略相似度是关键 | Softmax 0.57% vs Greedy 9.97% | 概率分布重叠度决定方差 | 评估前需检查策略差异 | exp_ope_validation |
| I3 | DR 依赖 Q 函数质量 | DR 偏差 12%-21% | 高方差奖励难预测 | 不优先使用 DR | exp_ope_validation |
| I4 | 存在最优探索率 | ε=0.3 表现最佳 | 覆盖与效率的平衡 | 线上配置 ε=0.3 | exp_ope_validation |

---

## 5) 决策空白（Decision Gaps）
> 写"要回答什么"，不写"怎么做实验"。建议 3–6 条。

| DG | 我们缺的答案 | 为什么重要（会改哪个决策） | 什么结果能关闭它 | 决策规则 |
|---|---|---|---|---|
| DG1 | OPE 置信区间如何估计？ | 决定策略差异是否显著 | Bootstrap CI 可用 | If CI < 5% → 可信；Else → 需更多样本 |
| DG2 | OPE + Simulator 如何结合？ | 优化评估流程 | 工作流定义 | OPE 粗筛 → Sim 精筛 → A/B 验证 |
| DG3 | 线上日志如何采集 propensity？ | OPE 可行性的前提 | 日志系统改造方案 | If 可行 → 开始采集；Else → 依赖 Simulator |

---

## 6) 设计原则（可复用规则）

### 6.1 已确认原则
| # | 原则 | 建议（做/不做） | 适用范围 | 证据 |
|---|---|---|---|---|
| P1 | 高方差场景优先 SNIPS | ✅ 采用 | 打赏/收益预测 | exp_ope_validation |
| P2 | 行为策略保持 ε≥0.3 | ✅ 采用 | 线上探索配置 | exp_ope_validation |
| P3 | 日志量 ≥ 5000 | ✅ 采用 | 数据采集目标 | exp_ope_validation |
| P4 | 必须记录 propensity | ✅ 采用 | 日志格式规范 | - |

### 6.2 待验证原则
| # | 原则 | 初步建议 | 需要验证（MVP/Gate） |
|---|---|---|---|
| P5 | OPE 差异 <5% 需 Simulator 验证 | 待定 | MVP-1.1 |
| P6 | 置信区间用 Bootstrap | 待定 | MVP-1.2 |

### 6.3 关键数字速查（只留会反复用到的）
| 指标 | 值 | 条件 | 来源 |
|---|---|---|---|
| Best SNIPS RelErr | 0.57% | Softmax, ε=0.3, n=5000 | exp_ope_validation |
| Worst SNIPS RelErr | 9.97% | Greedy, ε=0.3, n=5000 | exp_ope_validation |
| 最优探索率 | 0.3 | - | exp_ope_validation |
| 最小样本量 | 5000 | - | exp_ope_validation |

### 6.4 已关闭方向（避免重复踩坑）
| 方向 | 否定证据 | 关闭原因 | 教训 |
|---|---|---|---|
| ~~DR 作为主方法~~ | exp_ope_validation | Q 函数偏差大于方差减少 | 高方差奖励难预测 |
| ~~IPS 硬裁剪~~ | exp_ope_validation | 效果有限，不如 SNIPS | 自归一化更优雅 |

---

## 7) 指针（详细信息在哪）
| 类型 | 路径 | 说明 |
|---|---|---|
| 🗺️ Roadmap | `./gift_eval_roadmap.md` | Decision Gates + MVP 执行 |
| 📋 Kanban | `status/kanban.md` | 进度中控 |
| 📗 Experiments | `./exp/exp_*.md` | 单实验报告 |

---

## 8) 变更日志（只记录"知识改变"）
| 日期 | 变更 | 影响 |
|---|---|---|
| 2026-01-20 | 创建 gift_eval Hub，从 gift_EVpred 迁移 OPE 实验 | - |
| 2026-01-08 | OPE 验证实验完成，Gate-1 关闭 | SNIPS 确认为主方法 |

<details>
<summary><b>附录：术语表 / 背景</b></summary>

### 术语表
| 术语 | 定义 |
|---|---|
| OPE | Off-Policy Evaluation，离线策略评估 |
| IPS | Importance Sampling，重要性采样 |
| SNIPS | Self-Normalized IPS，自归一化重要性采样 |
| DR | Doubly Robust，双稳健估计 |
| Propensity | 行为策略选择动作的概率 |

### 背景
- 直播打赏场景的奖励分布重尾（Gini=0.94），方差极大
- 传统 IPS 在此场景容易权重爆炸
- 目标：找到适合高方差场景的可靠 OPE 方法

</details>
