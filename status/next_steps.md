# 📌 Next Steps

> **最后更新**: 2026-01-19 | **当前焦点**: gift_EVpred 识别大哥阶段

---

## 🎯 当前焦点：gift_EVpred

**核心目标**：提升 RevCap@1%（当前最优 52.60%）

**已验证结论**：
- ✅ Direct + raw Y 最优（+39.6% vs log Y）
- ❌ LightGBM 失败（-14.5%，树模型不适合稀疏数据）
- ❌ Three-Stage 失败（Direct raw Y 已隐式学会找 whale）

---

## 🔴 P0 - 必须完成（gift_EVpred）

| # | 实验 ID | 任务 | 状态 | 验收标准 | 备注 |
|---|---------|------|------|----------|------|
| 1 | **EXP-20260119-EVpred-01** | **指标体系落地** | ⏳ 待执行 | 统一模板 + RevCap 曲线 | 所有实验使用 |
| 2 | **EXP-20260119-EVpred-02** | **历史观看先验特征** | ⏳ 待执行 | RevCap > 54% | 替代 watch_live_time |

---

## 🟡 P1 - 应该完成（gift_EVpred）

| # | 实验 ID | 任务 | 状态 | 验收标准 | 备注 |
|---|---------|------|------|----------|------|
| 1 | **EXP-20260119-EVpred-03** | **样本加权回归** | ⏳ 待执行 | RevCap > 55% | $w_i=(1+y_i)^\alpha$ |

---

## 🟢 P2 - 可选完成（gift_EVpred）

| # | 任务 | 状态 | 验收标准 | 备注 |
|---|------|------|----------|------|
| 1 | Whale 专属特征 | 待立项 | RevCap > 54% | 历史大额打赏次数/金额 |
| 2 | 尝试 MLP | 待立项 | RevCap > 55% | NN 可能比树更适合稀疏 |

---

## 📦 gift_eval（OPE 离线评估）

**核心目标**：建立可靠的离线策略评估体系

**已验证结论**：
- ✅ SNIPS RelErr 0.57%-9.97%，Gate-1 关闭
- ✅ 最优探索率 ε=0.3，最小样本量 n≥5000
- ❌ DR 暂不推荐（Q 函数偏差大）

**待验证**：
- ⏳ MVP-1.1 OPE + Simulator 联合评估流程
- ⏳ MVP-1.2 Bootstrap 置信区间估计
- ⏳ MVP-1.3 线上日志采集方案（添加 propensity）

---

## 📦 Allocation 待续（暂停）

**已关闭**：
- ❌ MVP-5.1 召回-精排分工 — Recall Coverage 仅 5.5%
- ❌ MVP-5.2 影子价格 — 收益仅 +2.74%，不如 Greedy+Rules

**待验证**：
- ⏳ MVP-5.3 鲸鱼分散 (b-matching)
- ⏳ MVP-5.4 风险控制 (UCB/CVaR)

**可选 P2（如需精确模拟）**：
- ⏳ MVP-4.3 Simulator V3：时序冲动模型 + 用户忠诚度 + Gini 校准
  - 差距分析：[exp_simulator_v2 §5.4](../gift_allocation/exp/exp_simulator_v2_20260109.md)

---

## ✅ 已完成

| # | 任务 | 完成日期 | 产出 |
|---|------|---------|------|
| **gift_EVpred** |||
| 1 | MVP-1.0 Day-Frozen Baseline | 2026-01-18 | RevCap@1%=37.73% |
| 2 | MVP-1.1 Three-Stage | 2026-01-18 | RevCap@1%=43.60% |
| 3 | MVP-1.2 Raw Y vs Log Y | 2026-01-18 | **RevCap@1%=52.60% (+39.6%)** ✅ 当前最优 |
| 4 | MVP-2.0 LightGBM | 2026-01-18 | ❌ 失败（-14.5%） |
| 5 | MVP-3.0 三层指标体系设计 | 2026-01-18 | 识别层/估值层/分配层框架 |
| **gift_allocation Phase 0-3** |||
| 6 | MVP-0.1 KuaiLive EDA | 2026-01-08 | Gini=0.94, Top1%=60% |
| 7 | MVP-0.2 Baseline | 2026-01-08 | Top-1%=56.2%, Spearman=0.891 |
| 8 | MVP-0.3 Simulator V1 | 2026-01-08 | Gini误差<5%, Greedy 3x Random |
| 9 | MVP-1.1~1.4 估计层 | 2026-01-08 | Direct 胜出 (54.5% vs 35.7%) |
| 10 | MVP-2.1~2.2 分配层 | 2026-01-08 | 软约束+32%收益 |
| 11 | MVP-3.1 OPE验证 | 2026-01-08 | SNIPS RelErr<10% |
| **gift_allocation Phase 4** |||
| 12 | MVP-4.1+ Simulator V2金额 | 2026-01-09 | P50=0%, P90=13%, Mean=24% ✅ |
| 13 | MVP-4.2 Simulator V2并发 | 2026-01-09 | 边际递减24.4%, 拥挤率68% ✅ |
| **gift_allocation Phase 5** |||
| 14 | MVP-5.1 召回-精排 | 2026-01-09 | ❌ FAIL (Recall Coverage 5.5%) |
| 15 | MVP-5.2 影子价格 | 2026-01-09 | ❌ FAIL (收益仅+2.74%) |

---

## 📝 快速命令

### 查看 EVpred Hub
```bash
cat gift_EVpred/gift_EVpred_hub.md
```

### 查看 EVpred Roadmap
```bash
cat gift_EVpred/gift_EVpred_roadmap.md
```

### 开始 EXP-01 (指标体系落地)
```bash
# 1. 阅读实验计划
cat gift_EVpred/exp/exp_metrics_landing_20260119.md
# 2. 实现 RevCap 曲线 + 稳定性评估
# 3. 生成统一报告模板
```

### 开始 EXP-02 (历史观看先验)
```bash
# 1. 阅读实验计划
cat gift_EVpred/exp/exp_watch_history_prior_20260119.md
# 2. 在 data_utils.py 中添加新特征
# 3. 评估 RevCap@1% > 54%
```
