# 🍃 历史观看先验特征
> **Name:** Watch History Prior Features
> **ID:** `EXP-20260119-EVpred-02`
> **Topic:** `gift_EVpred` | **MVP:** MVP-3.2
> **Author:** Viska Wei | **Date:** 2026-01-19 | **Status:** 🔄

> 🎯 **Target:** 用历史观看先验特征替代被删除的 watch_live_time，预期 RevCap > 54%
> 🚀 **Next:** 验证历史先验是否能弥补当前 session 信息缺失

## ⚡ 核心结论速览

> **一句话**: TODO - 待实验完成

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| 历史观看先验有增益？ | ⏳ 待验证 | RevCap > 54% 则采用 |

| 指标 | 值 | 启示 |
|------|-----|------|
| Baseline (Direct raw Y) | 52.60% | 当前最优 |
| Target | >54% | +1.4% 增益 |

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

| 配置 | Spearman | RevCap@1% | RevCap@5% | Δ vs Baseline |
|------|----------|-----------|-----------|---------------|
| Baseline (31 特征) | 0.0608 | 52.60% | 64.72% | - |
| + 全部新特征 (38) | | | | |
| + user_only (34) | | | | |
| + pair_only (32) | | | | |

---

# 4. 💡 验收标准

| 验收项 | 标准 | 状态 |
|--------|------|------|
| RevCap@1% | > 54% | ⏳ |
| 无泄漏验证 | 200/200 通过 | ⏳ |
| 特征重要性 | 新特征在 Top 10 内 | ⏳ |

---

# 5. 📎 附录

## 5.1 代码参考

| 文件 | 说明 |
|------|------|
| `gift_EVpred/data_utils.py` | 需新增 watch_history 特征计算 |
| `scripts/run_baseline.py` | 参考训练脚本 |

## 5.2 理论依据

**为什么历史观看先验有效？**

- 观看时长是"打赏冲动/停留足够久"的先验信号
- 历史模式是用户行为的稳定特征（vs 当前 session 是随机的）
- 满足因果：watch_history < click_ts < gift_ts（不泄漏）

**与 watch_live_time 的区别**：

| 特征 | 时间点 | 可用性 | 泄漏风险 |
|------|--------|--------|---------|
| watch_live_time (当前 session) | session 结束时 | ❌ click 时刻不可得 | 🔴 Post-treatment |
| watch_history (历史先验) | click 之前 | ✅ click 时刻可得 | 🟢 无泄漏 |

---

> **立项时间**: 2026-01-19
