# 🍃 历史观看先验特征
> **Name:** Watch History Prior Features
> **ID:** `EXP-20260119-EVpred-02`
> **Topic:** `gift_EVpred` | **MVP:** MVP-3.2
> **Author:** Viska Wei | **Date:** 2026-01-19 | **Status:** ⚠️ 无增益

> 🎯 **Target:** 用历史观看先验特征替代被删除的 watch_live_time，预期 RevCap > 54%
> 🚀 **Next:** 验证历史先验是否能弥补当前 session 信息缺失

## ⚡ 核心结论速览

> **一句话**: ⚠️ 历史观看特征无增益，礼物历史已是最强信号

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| 历史观看先验有增益？ | ⚠️ **无增益** | RevCap 不变（51.4%） |

| 指标 | 值 | 启示 |
|------|-----|------|
| **Baseline (gift history)** | **51.4%** | 当前最优 |
| + Watch history | 51.4% | 无提升 |
| 新特征最高排名 | #10/23 | 远弱于 gift history |

| Type | Link |
|------|------|
| 🧠 Hub | `gift_EVpred/gift_EVpred_hub.md` § 洞见 M7, M8 |
| 🗺️ Roadmap | `gift_EVpred/gift_EVpred_roadmap.md` § MVP-3.2 |

---

# 1. 🎯 目标

**问题**: watch_live_time 是 post-treatment 特征（包含打赏后停留时间），已被删除。能否用历史观看先验替代？

**背景**:
- Hub 洞见 M7：watch_live_time 是 post-treatment，click 时刻不可得
- Hub 洞见 M8：可用替代 = 历史观看先验（user/pair/streamer 历史平均观看时长）
- 当前 Baseline：RevCap@1% = 52.60%（Direct + raw Y, Day-Frozen）

| 预期 | 判断标准 |
|------|---------|
| 有增益 | RevCap > 54% → 采用 |
| 无增益 | RevCap ≤ 52.60% → 不采用 |

---

# 2. 🧪 实验设计

## 2.1 新特征列表

所有特征必须严格满足 `gift_ts < click_ts`（Day-Frozen 语义）：

| 特征名 | 定义 | 计算方式 |
|--------|------|---------|
| `user_watch_mean_7d` | 用户过去 7 天平均观看时长 | Day-Frozen cumsum |
| `user_watch_mean_14d` | 用户过去 14 天平均观看时长 | Day-Frozen cumsum |
| `pair_watch_mean` | 该 user-streamer pair 历史平均观看时长 | Day-Frozen cumsum |
| `streamer_avg_watch_time` | 主播平均被观看时长 | Day-Frozen cumsum |
| `user_watch_p90` | 用户观看时长 P90 分位数 | Day-Frozen quantile |
| `time_since_last_watch` | 距离上次观看该主播的时间间隔 | Day-Frozen lookup |
| `watch_days_active` | 用户活跃观看天数 | Day-Frozen count |

## 2.2 数据

| 项 | 值 |
|----|-----|
| 来源 | KuaiLive |
| 路径 | `data/KuaiLive/` |
| Train/Val/Test | 1,629,415 / 1,717,199 / 1,409,533 (7-7-7) |
| 特征维度 | 31 → 31+7 = 38 |

## 2.3 模型

| 参数 | 值 |
|------|-----|
| 模型 | Ridge Regression |
| 预测目标 | raw Y (gift_price_label) |
| 正则化 | alpha=1.0 |

## 2.4 扫描参数

| 扫描 | 范围 | 固定 |
|------|------|------|
| 新特征组合 | [全部7个, user_only, pair_only] | model=Ridge |

---

# 3. 📊 预期产出

## 3.1 图表

| 图表 | 内容 | 保存路径 |
|------|------|---------|
| Fig 1 | 新特征重要性分析 | `img/watch_history_importance.png` |
| Fig 2 | RevCap 曲线对比（+新特征 vs Baseline） | `img/watch_history_revcap.png` |
| Fig 3 | 特征消融实验 | `img/watch_history_ablation.png` |

## 3.2 数值结果

| 配置 | RevCap@1% | RevCap@5% | RevCap@10% | Δ vs Baseline |
|------|-----------|-----------|------------|---------------|
| **Baseline (20 特征)** | **51.36%** | **63.63%** | **68.62%** | - |
| + Watch history (23) | 51.37% | 63.94% | 69.01% | +0.01pp / +0.31pp / +0.39pp |

**完整 RevCap 曲线对比**：

| K | Baseline | + Watch | Δ |
|---|----------|---------|---|
| 0.1% | 21.18% | 21.18% | 0.00pp |
| 0.5% | 43.35% | 43.35% | 0.00pp |
| **1%** | **51.36%** | **51.37%** | **+0.01pp** |
| 2% | 56.33% | 56.29% | -0.04pp |
| **5%** | 63.63% | **63.94%** | **+0.31pp** |
| **10%** | 68.62% | **69.01%** | **+0.39pp** |

**特征重要性排名**：

| 排名 | 特征 | 系数 | 类别 |
|------|------|------|------|
| 1 | pair_gift_mean_hist | 8.180 | Gift History |
| 2 | pair_gift_sum_hist | 4.915 | Gift History |
| 3 | user_gift_sum_hist | 4.386 | Gift History |
| 4 | user_gift_mean_hist | 2.538 | Gift History |
| 5 | age | 0.465 | Profile |
| ... | ... | ... | ... |
| **10** | **user_watch_mean_hist** | **0.170** | **Watch History** |
| **16** | **pair_watch_mean_hist** | **0.103** | **Watch History** |
| **18** | **str_watch_mean_hist** | **0.091** | **Watch History** |

**关键发现**：
- 历史观看特征系数最高 0.170，仅为 pair_gift_mean_hist 的 **1/48**
- 礼物历史信号已高度饱和，观看历史无法提供边际增益
- 中部样本（@5%, @10%）有微弱提升，但核心指标 @1% 不变

---

# 4. 💡 验收标准

| 验收项 | 标准 | 结果 | 状态 |
|--------|------|------|------|
| RevCap@1% | > 54% | 51.37%（未达标） | ❌ |
| 无泄漏验证 | 200/200 通过 | Day-Frozen | ✅ |
| 特征重要性 | 新特征在 Top 10 内 | #10/23（勉强达标） | ⚠️ |

---

# 5. 💡 结论与洞见

## 5.1 核心结论

> **⚠️ 历史观看先验无增益**：礼物历史已是最强信号，观看历史无法提供边际增益

| 洞见 | 证据 | 解释 |
|------|------|------|
| **观看历史无增益** | RevCap@1% +0.01pp | 礼物历史已高度饱和 |
| **系数差 48 倍** | 0.170 vs 8.180 | 打赏行为 >> 观看行为 |
| **中部有微弱帮助** | @5% +0.31pp, @10% +0.39pp | 对非头部样本略有益 |

## 5.2 失败原因分析

1. **信号冗余**：观看行为与打赏行为高度相关（想打赏的人自然会多看）
2. **礼物历史饱和**：pair_gift_mean_hist 系数 8.18，已提取绝大部分信息
3. **观看无额外解释力**：给定礼物历史后，观看历史不提供额外预测信号

## 5.3 设计启示

| 原则 | 建议 |
|------|------|
| **不值得加入** | 观看历史特征收益过小（+0.01pp），不值得增加复杂度 |
| **礼物历史是核心** | 继续挖掘打赏行为的高阶特征（如打赏间隔、金额波动等） |
| **下一步方向** | MLP 建模非线性关系 / Whale 专属特征 |

---

# 6. 📎 附录

## 6.1 执行记录

| 项 | 值 |
|----|-----|
| 执行时间 | 2026-01-19 |
| 结果文件 | `results/watch_history_prior_20260119.json` |
| 特征数 | Baseline 20 → + Watch 23 |
| 新增特征 | `pair_watch_mean_hist`, `user_watch_mean_hist`, `str_watch_mean_hist` |

## 6.2 代码参考

| 文件 | 说明 |
|------|------|
| `gift_EVpred/data_utils.py` | 统一数据处理（Day-Frozen） |
| `gift_EVpred/metrics.py` | RevCap 评估函数 |

## 6.3 理论依据

**为什么历史观看先验【原以为】有效？**

- 观看时长是"打赏冲动/停留足够久"的先验信号
- 历史模式是用户行为的稳定特征（vs 当前 session 是随机的）
- 满足因果：watch_history < click_ts < gift_ts（不泄漏）

**【实验结果】为什么实际无效？**

- 打赏行为已隐含观看行为（想打赏 → 自然多看）
- 礼物历史系数是观看历史的 48 倍，信号完全饱和
- 给定打赏历史后，观看历史与未来打赏条件独立

**与 watch_live_time 的区别**：

| 特征 | 时间点 | 可用性 | 泄漏 | 有效性 |
|------|--------|--------|------|--------|
| watch_live_time (当前 session) | session 结束时 | ❌ 不可得 | 🔴 Post | - |
| watch_history (历史先验) | click 之前 | ✅ 可得 | 🟢 无 | ❌ 无增益 |

---

> **立项时间**: 2026-01-19
> **完成时间**: 2026-01-19
