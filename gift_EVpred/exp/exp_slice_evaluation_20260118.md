# 🍃 Slice Evaluation
> **Name:** Slice Evaluation
> **ID:** `EXP-20260118-gift_EVpred-06`
> **Topic:** `gift_EVpred` | **MVP:** MVP-1.4
> **Author:** Viska Wei | **Date:** 2026-01-18 | **Status:** ✅ 完成

> 🎯 **Target:** 分析模型在不同切片（冷启动 pair/streamer/user，头部/长尾用户）上的表现
> 🚀 **Next:** 冷启动性能极差 → 必须设计冷启动专用策略（fallback/协同过滤/内容特征）

## ⚡ 核心结论速览

> **一句话**: 冷启动是致命瓶颈——冷启动 pair/streamer/user 的 RevCap@1% 分别仅为基线的 16%/3%/2%，模型本质上无法预测冷启动场景。

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| 冷启动 pair 性能如何？ | RevCap@1% = 3.2% (基线的 16%) | ❌ 极差，需要 fallback 策略 |
| 冷启动 streamer 性能如何？ | RevCap@1% = 0.7% (基线的 3%) | ❌ 几乎无预测能力 |
| Top-1% 用户 vs 长尾用户性能差异？ | Top-1%: 12.3% vs Tail: 1.5% | 8x 差距，头部用户更可预测 |

| 指标 | 值 | 启示 |
|------|-----|------|
| 冷启动 pair RevCap@1% | **3.2%** | 需要协同过滤或内容特征 |
| 冷启动 streamer RevCap@1% | **0.7%** | 需要主播侧冷启动策略 |
| 冷启动 user RevCap@1% | **0.5%** | 需要用户侧冷启动策略 |

| Type | Link |
|------|------|
| 🧠 Hub | `gift_EVpred/gift_EVpred_hub.md` § Q2.3 |
| 🗺️ Roadmap | `gift_EVpred/gift_EVpred_roadmap.md` § MVP-1.4 |

---
# 1. 🎯 目标

**问题**: 模型在不同切片上的性能是否有显著差异？冷启动场景表现如何？

**验证**: Q2.3（切片评估是否必要）

| 预期 | 判断标准 | 实际结果 |
|------|---------|---------|
| 冷启动性能明显下降 | RevCap@1% < 全量 * 0.5 | ✅ 冷启动 pair=16%, streamer=3%, user=2% |
| 头部用户预测更准 | Top-1% > 全量 * 1.2 | ❌ Top-1%=12.3% < 全量=20.8% |
| 性能差异不大 | 差异 < 20% | ❌ 差异巨大（冷启动 vs 热启动 ~10x） |

---

# 2. 🦾 算法

> 📌 本实验主要是评估分析，无需新算法

**切片定义**：

1. **冷启动 pair**: test 中 (user, streamer) 在 train 无交互
2. **冷启动 streamer**: test 中 streamer 在 train 无收礼记录
3. **冷启动 user**: test 中 user 在 train 无送礼记录
4. **Top-1% user**: 历史总打赏金额 Top 1% 的用户
5. **Top-10% user**: 历史总打赏金额 Top 10% 的用户
6. **长尾 user**: 历史总打赏金额 Bottom 50% 的用户

---

# 3. 🧪 实验设计

## 3.1 数据

| 项 | 值 |
|----|-----|
| 来源 | KuaiLive click-level 数据 |
| 路径 | `data/KuaiLive/` |
| Train | 3,436,660 样本 |
| Test | 736,428 样本 |
| 特征维度 | 49 (past-only frozen) |

## 3.2 切片构造（实际比例）

| 切片 | 样本数 | 占比 | Gift Rate |
|------|--------|------|-----------|
| 全量 | 736,428 | 100% | 1.52% |
| 冷启动 pair | 452,689 | 61.5% | 1.29% |
| 热启动 pair | 283,739 | 38.5% | 1.89% |
| 冷启动 streamer | 506,097 | 68.7% | 1.07% |
| 热启动 streamer | 230,331 | 31.3% | 2.50% |
| 冷启动 user | 164,931 | 22.4% | 2.26% |
| 热启动 user | 571,497 | 77.6% | 1.30% |

---

# 4. 📊 图表

### Fig 1: Slice Performance Comparison
![](./img/slice_performance_comparison.png)

**观察**:
- 冷启动切片（cold_pair, cold_streamer, cold_user）性能极差
- 热启动切片（warm_*）性能明显优于基线

### Fig 2: Cold-Start Analysis
![](./img/coldstart_analysis.png)

**观察**:
- pair 历史越多，性能越好
- 0 历史的 pair 性能几乎为 0

---

# 5. 💡 洞见

## 5.1 宏观
- **冷启动是当前模型的致命瓶颈**：模型本质上依赖历史交互信号，无法预测新 pair/user/streamer
- **数据分布严重偏斜**：61.5% 的 test 样本是冷启动 pair，但贡献的收入预测能力几乎为 0

## 5.2 模型层
- **Frozen 特征的天然局限**：冷启动场景下所有 pair_gift_* 特征都是 0，模型只能依赖静态特征
- **静态特征预测力不足**：仅靠用户/主播的固有属性无法预测打赏行为

## 5.3 细节
- **冷启动 user 的 gift rate 更高**（2.26% vs 平均 1.52%），但模型无法识别
- **Top-1% 用户性能不如预期**：可能是因为高价值用户行为更难预测（更稀疏）

---

# 6. 📝 结论

## 6.1 核心发现
> **冷启动是当前模型的致命瓶颈，冷启动切片的 RevCap@1% 仅为热启动的 ~10%，必须设计专门的冷启动策略。**

- ✅ H5.1: 冷启动 pair 性能 < 全量 50% → **确认**（3.2% vs 20.8%，仅 16%）
- ✅ H5.2: 冷启动 streamer 性能下降显著 → **确认**（0.7% vs 20.8%，仅 3%）
- ❌ H5.3: Top-1% user 预测更准 → **否定**（12.3% < 20.8%）
- ✅ H5.4: 长尾 user 预测接近随机 → **确认**（1.5%，接近随机的 1%）

## 6.2 关键结论

| # | 结论 | 证据 |
|---|------|------|
| 1 | **冷启动 pair 预测能力极差** | RevCap@1% = 3.2%（基线的 16%） |
| 2 | **冷启动 streamer 几乎无法预测** | RevCap@1% = 0.7%（基线的 3%） |
| 3 | **热启动场景性能尚可** | warm_pair RevCap@1% = 31.1%（基线的 149%） |
| 4 | **长尾用户预测接近随机** | tail_user RevCap@1% = 1.5%（~随机） |

## 6.3 设计启示

| 原则 | 建议 |
|------|------|
| 冷启动 fallback | 对冷启动 pair 使用 popularity-based 或协同过滤 fallback |
| 分层模型 | 考虑为冷启动/热启动分别建模 |
| 内容特征 | 探索 user-streamer 内容匹配度特征（不依赖历史交互） |

| ⚠️ 陷阱 | 原因 |
|---------|------|
| 忽略冷启动问题 | 61.5% 的流量是冷启动，直接影响收益 |
| 只看全量指标 | 掩盖了冷启动的严重问题 |

## 6.4 关键数字

| 指标 | 值 | 条件 |
|------|-----|------|
| 全量 RevCap@1% | **20.8%** | baseline |
| 冷启动 pair RevCap@1% | **3.2%** | 16% of baseline |
| 冷启动 streamer RevCap@1% | **0.7%** | 3% of baseline |
| 冷启动 user RevCap@1% | **0.5%** | 2% of baseline |
| 热启动 pair RevCap@1% | **31.1%** | 149% of baseline |
| Top-1% user RevCap@1% | **12.3%** | - |
| Tail user RevCap@1% | **1.5%** | ~random |

## 6.5 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| 冷启动策略 | 设计 popularity-based fallback for cold-start | 🔴 |
| 内容特征 | 探索 user-streamer 内容匹配特征 | 🔴 |
| 分层建模 | 考虑 cold/warm 分层模型 | 🟡 |

---

# 7. 📎 附录

## 7.1 数值结果

| 切片 | 样本数 | Gift Rate | RevCap@1% | RevCap@5% | Spearman |
|------|--------|-----------|-----------|-----------|----------|
| all | 736,428 | 1.52% | 20.8% | 26.9% | 0.096 |
| cold_pair | 452,689 | 1.29% | 3.2% | 15.2% | 0.020 |
| warm_pair | 283,739 | 1.89% | 31.1% | 36.9% | 0.138 |
| cold_streamer | 506,097 | 1.07% | 0.7% | 7.0% | - |
| warm_streamer | 230,331 | 2.50% | 32.7% | 39.3% | 0.128 |
| cold_user | 164,931 | 2.26% | 0.5% | 7.4% | - |
| warm_user | 571,497 | 1.30% | 27.3% | 32.1% | 0.121 |
| top1pct_user | 10,426 | 3.29% | 12.3% | 28.1% | 0.189 |
| top10pct_user | 80,126 | 1.89% | 28.2% | 43.4% | 0.108 |
| middle_user | 245,921 | 1.57% | 14.8% | 25.3% | 0.109 |
| tail_user | 399,955 | 1.36% | 1.5% | 13.2% | 0.074 |

## 7.2 执行记录

| 项 | 值 |
|----|-----|
| 脚本 | `scripts/evaluate_slices.py` |
| Output | `results/slice_evaluation_20260118.json` |
| 图表 | `exp/img/slice_performance_comparison.png`, `exp/img/coldstart_analysis.png` |

```bash
# 评估
python scripts/evaluate_slices.py
```

---

> **实验完成时间**: 2026-01-18
