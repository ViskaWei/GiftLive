# 🗺️ gift_eval Roadmap
> **Name:** Off-Policy Evaluation for Gift Allocation | **ID:** `EXP-20260120-gift-eval-roadmap`  
> **Topic:** `gift_eval` | **Phase:** 1  
> **Author:** Viska Wei | **Date:** 2026-01-20 | **Status:** 🔄
```
💡 当前阶段目标  
Gate：Gate-1 OPE 可行性验证（已关闭）
Phase 1 目标：完善 OPE 工具链，建立评估流程
```

---

## 🔗 Related Files
| Type | File |
|------|------|
| 🧠 Hub | `gift_eval_hub.md` |
| 📋 Kanban | `status/kanban.md` |
| 📗 Experiments | `exp/exp_*.md` |

---

# 1. 🚦 Decision Gates

> Roadmap 定义怎么验证，Hub 做战略分析

## 1.1 战略路线 (来自Hub)

| Route | 名称 | Hub推荐 | 验证Gate |
|-------|------|---------|----------|
| IPS | 标准重要性采样 | 🔴 | - |
| DR | 双稳健估计 | 🟡 | - |
| **SNIPS** | 自归一化 IPS | 🟢 **推荐** | Gate-1 ✅ |

## 1.2 Gate定义

### Gate-1: OPE 可行性验证 ✅ CLOSED

| 项 | 内容 |
|----|------|
| 验证 | OPE 在高方差打赏场景能达到 <10% 相对误差 |
| MVP | MVP-0.1 OPE Validation |
| 若 <10% | ✅ SNIPS 可用于离线策略对比 |
| 若 ≥10% | ❌ 需依赖 Simulator 或 A/B |
| 状态 | ✅ 已关闭 (SNIPS 0.57%-9.97%) |

### Gate-2: OPE 置信区间估计 ⏳

| 项 | 内容 |
|----|------|
| 验证 | Bootstrap CI 是否可靠 |
| MVP | MVP-1.2 |
| 若 CI 可靠 | 可判断策略差异显著性 |
| 若 CI 不可靠 | 需增加样本或 Simulator 验证 |
| 状态 | ⏳ 待启动 |

## 1.3 本周重点

| 优先级 | MVP | Gate | 状态 |
|--------|-----|------|------|
| 🟢 Done | MVP-0.1 | Gate-1 | ✅ |
| 🟡 P1 | MVP-1.1 | - | ⏳ OPE + Sim 流程 |
| 🟡 P1 | MVP-1.2 | Gate-2 | ⏳ 置信区间 |

---

# 2. 📋 MVP列表

## 2.1 总览

| MVP | 名称 | Phase | Gate | 状态 | exp_id | 报告 |
|-----|------|-------|------|------|--------|------|
| 0.1 | OPE Validation | 0 | Gate-1 | ✅ | `EXP-20260108-gift-allocation-10` | [exp_ope_validation](./exp/exp_ope_validation_20260108.md) |
| 1.1 | OPE + Simulator 流程 | 1 | - | ⏳ | - | - |
| 1.2 | Bootstrap 置信区间 | 1 | Gate-2 | ⏳ | - | - |
| 1.3 | 线上日志采集方案 | 1 | - | ⏳ | - | - |

**状态**: ⏳计划 | 🔴就绪 | 🚀运行 | ✅完成 | ❌取消

## 2.2 配置速查

| MVP | 数据量 | 行为策略 | OPE 方法 | 关键变量 |
|-----|--------|----------|----------|---------|
| 0.1 | 5000 logs | ε-greedy 0.3 | IPS/SNIPS/DR | ε, n_logs |
| 1.1 | TBD | ε-greedy | SNIPS + Sim | 筛选阈值 |
| 1.2 | 5000+ | ε-greedy 0.3 | SNIPS | n_bootstrap |

---

# 3. 🔧 MVP规格

## Phase 0: 验证 OPE 可行性

### MVP-0.1: OPE Validation ✅

| 项 | 配置 |
|----|------|
| 目标 | 验证 OPE 在高方差场景的有效性 |
| 数据 | SimulatorV2+ (500 users × 50 streamers) |
| 方法 | IPS / IPS-Clip10 / SNIPS / DM / DR / DR-Clip10 |
| 验收 | 至少一种方法 RelErr < 10% |
| 结果 | ✅ SNIPS 0.57%-9.97% |

## Phase 1: 完善工具链

### MVP-1.1: OPE + Simulator 评估流程

| 项 | 配置 |
|----|------|
| 目标 | 定义 OPE 粗筛 + Simulator 精筛流程 |
| 依赖 | MVP-0.1 |
| 验收 | 流程可操作，有阈值定义 |

### MVP-1.2: Bootstrap 置信区间

| 项 | 配置 |
|----|------|
| 目标 | 为 SNIPS 估计添加置信区间 |
| Gate | Gate-2 |
| 数据 | n=5000+, n_bootstrap=1000 |
| 验收 | CI 覆盖率 ≥ 95% |

### MVP-1.3: 线上日志采集方案

| 项 | 配置 |
|----|------|
| 目标 | 设计日志格式，添加 propensity 字段 |
| 验收 | 技术方案文档 + 实现路径 |

---

# 4. 📊 进度追踪

## 4.1 看板

```
⏳计划    🔴就绪    🚀运行    ✅完成
MVP-1.2   MVP-1.1             MVP-0.1
MVP-1.3
```

## 4.2 Gate进度

| Gate | MVP | 状态 | 结果 |
|------|-----|------|------|
| Gate-1 | MVP-0.1 | ✅ | SNIPS < 10% RelErr |
| Gate-2 | MVP-1.2 | ⏳ | - |

## 4.3 结论快照

| MVP | 结论 | 关键指标 | 同步Hub |
|-----|------|---------|---------|
| 0.1 | SNIPS 在高方差场景表现最佳，Gate-1 关闭 | RelErr 0.57%-9.97% | ✅ §3 |

## 4.4 时间线

| 日期 | 事件 |
|------|------|
| 2026-01-08 | MVP-0.1 OPE Validation 完成 |
| 2026-01-20 | 从 gift_EVpred 迁移到独立 gift_eval Topic |

---

# 5. 🔗 文件索引

## 5.1 实验索引

| exp_id | topic | 状态 | MVP |
|--------|-------|------|-----|
| `EXP-20260108-gift-allocation-10` | gift_eval | ✅ | MVP-0.1 |

## 5.2 运行路径

| MVP | 脚本 | 配置 | 输出 |
|-----|------|------|------|
| 0.1 | `scripts/train_ope_validation.py` | inline | `results/ope_validation_20260108.json` |

---

# 6. 📎 附录

## 6.1 数值汇总

| 策略 | 方法 | RelErr | Bias | 备注 |
|------|------|--------|------|------|
| Softmax | IPS | 1.72% | 0.34 | ✅ 极佳 |
| Softmax | **SNIPS** | **0.57%** | -0.11 | ✅ 最佳 |
| Greedy | IPS | 15.34% | -4.11 | 权重爆炸 |
| Greedy | **SNIPS** | **9.97%** | -2.27 | ✅ 通过 |
| Concave | IPS | 8.79% | -1.84 | ✅ 良好 |
| Concave | **SNIPS** | **4.31%** | -0.90 | ✅ 最佳 |

## 6.2 文件索引

| 类型 | 路径 |
|------|------|
| Roadmap | `gift_eval_roadmap.md` |
| Hub | `gift_eval_hub.md` |
| 图表 | `img/` |
| 结果 | `results/` |

## 6.3 更新日志

| 日期 | 变更 | 章节 |
|------|------|------|
| 2026-01-20 | 创建 Roadmap，从 gift_EVpred 迁移 | - |
