# 🍃 KuaiLive 全方位探索性数据分析（Comprehensive EDA）
> **Name:** KuaiLive Comprehensive EDA  
> **ID:** `EXP-20260109-gift-allocation-02`  
> **Topic:** `gift_allocation` | **MVP:** MVP-0.1-Enhanced  
> **Author:** Viska Wei | **Date:** 2026-01-09 | **Status:** 🔄 计划中

> 🎯 **Target:** 对 KuaiLive 数据做"全方位、专业、可落地"的探索性数据分析，从用户行为、直播间供给、互动结构、时序规律、稀疏性/冷启动、异常与数据质量等多个维度挖出可行动的洞见  
> 🚀 **Next:** 产出高质量图表方案（每张图必须能"一眼得到结论"）和可执行的建模/产品建议

## ⚡ 核心结论速览

> **一句话**: [待执行后填写] 全方位 EDA 将揭示用户观看→互动→付费的完整漏斗、session 级行为模式、供给侧质量分布、用户-主播匹配结构、时序规律、异常风险，为建模和产品策略提供量化依据

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H2.1: 打赏是"即时冲动"还是"随观看累积"？ | ⏳ 待验证 | 需 session 级分析首礼时间分布 |
| H2.2: 高观看低付费人群是否存在？规模多大？ | ⏳ 待验证 | 需用户二维象限分析 |
| H2.3: 主播供给是否存在"高吸引低变现"类型？ | ⏳ 待验证 | 需主播二维象限分析 |
| H2.4: 付费是否强绑定于特定主播？ | ⏳ 待验证 | 需用户专一度分布分析 |
| H2.5: 数据是否存在异常/作弊/污染？ | ⏳ 待验证 | 需异常检测规则 |

| 指标 | 值 | 启示 |
|------|-----|------|
| [待执行后填写] | | |

| Type | Link |
|------|------|
| 🧠 Hub | `gift_allocation/gift_allocation_hub.md` § H1.x |
| 🗺️ Roadmap | `gift_allocation/gift_allocation_roadmap.md` § MVP-0.1-Enhanced |
| 📊 Baseline | `gift_allocation/exp/exp_kuailive_eda_20260108.md` |

---

# 0. Executive Summary（不超过12行）

> ⚠️ **待执行后填写**：最重要的 5~8 条洞见（每条=结论 + 证据图编号 + So what）+ 3 条最优先的下一步行动

**核心洞见（待补充）**：
1. [结论] → Fig X → [So what]
2. [结论] → Fig Y → [So what]
3. [结论] → Fig Z → [So what]
4. [结论] → Fig W → [So what]
5. [结论] → Fig V → [So what]

**最优先行动（待补充）**：
1. [做什么] / [为什么] / [成功标准]
2. [做什么] / [为什么] / [成功标准]
3. [做什么] / [为什么] / [成功标准]

---

# 1. Baseline Review（基于现有 EDA 的差距分析）

## 1.1 现有 EDA 已覆盖的主题清单

基于 `exp_kuailive_eda_20260108.md`，已有分析包括：

| 维度 | 已覆盖内容 | 图表编号 |
|------|----------|---------|
| **金额分布** | Mean/Median/P90/P95/P99/Max, log 变换后分布 | Fig 1-3 |
| **集中度** | User Gini=0.942, Streamer Gini=0.930, Top-K% share | Fig 4-5 |
| **稀疏性** | Matrix density=0.0064%, Cold start streamer=92.2% | Fig 9 |
| **时间模式** | Peak hour=12, 小时分布 | Fig 8 |
| **频次分布** | 每用户/每主播打赏次数分布 | Fig 6-7 |

**关键数字（已冻结口径）**：
- 打赏率 (per-click): 1.48% = 72,646 / 4,909,515
- User Gini: 0.942
- Streamer Gini: 0.930
- Top 1% User Share: 59.9%
- Amount P50/P90/P99: 2/88/1,488

## 1.2 现有 EDA 的关键缺口

| 缺口类别 | 缺失内容 | 影响 | 优先级 |
|---------|---------|------|--------|
| **会话/漏斗** | ❌ 无 session 级分析 | 无法理解"观看→打赏"的转化路径 | 🔴 P0 |
| | ❌ 无转化漏斗（click→session→gift） | 无法定位流失环节 | 🔴 P0 |
| | ❌ 无首礼时间分析 | 无法判断打赏是"即时"还是"累积" | 🔴 P0 |
| **用户分群** | ❌ 无用户二维象限（观看 vs 付费） | 无法识别"高观看低付费"人群 | 🔴 P0 |
| | ❌ 无用户留存/回访分析 | 无法评估长期价值 | 🟡 P1 |
| | ❌ 无付费价位阶梯分析 | 无法理解价格敏感度 | 🟡 P1 |
| **供给侧** | ❌ 无主播二维象限（观看 vs 变现） | 无法识别"高吸引低变现"主播 | 🔴 P0 |
| | ❌ 无主播转化效率分析 | 无法评估供给侧质量 | 🔴 P0 |
| | ❌ 无冷启动细分（新主播表现） | 无法制定扶持策略 | 🟡 P1 |
| **交互结构** | ❌ 无用户专一度分析 | 无法判断偏好稳定性 | 🔴 P0 |
| | ❌ 无二部图/网络分析 | 无法理解匹配结构 | 🟡 P1 |
| | ❌ 无跨主播付费迁移分析 | 无法评估推荐扩展性 | 🟡 P1 |
| **时序深度** | ⚠️ 仅有小时分布 | 缺少星期/周期/事件峰分析 | 🟡 P1 |
| | ❌ 无峰值贡献分解 | 无法识别驱动因素 | 🟡 P1 |
| **异常/风险** | ❌ 无异常检测 | 无法识别作弊/数据污染 | 🔴 P0 |
| | ❌ 无数据质量评分 | 无法评估数据可信度 | 🔴 P0 |
| **口径风险** | ⚠️ gift_rate=100% 误导 | user.csv 仅包含有打赏用户 | 🔴 P0 |
| | ⚠️ 缺少 session 级口径 | 所有指标都是 click-level | 🔴 P0 |

## 1.3 口径冻结：核心指标定义

为避免后续分析口径不一致，先冻结以下核心指标：

### Click-level 指标（事件级）
- **打赏率 (per-click)**: `gift_count / click_count` = 72,646 / 4,909,515 = 1.48%
- **平均打赏金额**: Mean=82.7, Median=2
- **时间分布**: 按 `timestamp` 小时聚合

### Session-level 指标（会话级，需构建）
- **Session 定义**: 同一 `user_id` + `live_id` 的连续观看窗口
  - `session_start = timestamp`
  - `session_end = timestamp + watch_live_time`
  - `session_duration = watch_live_time`
- **Session 转化率**: `gift_sessions / total_sessions`
- **首礼时间**: `t_first_gift - session_start`（相对 session 开始）

### User-level 指标（用户级）
- **用户观看强度**: 总观看时长、总 session 数、DAU（如有日期）
- **用户付费强度**: 总打赏金额、总打赏次数、付费率（是否曾付费）
- **用户专一度**: Top1 主播占其观看/打赏的比例

### Streamer-level 指标（主播级）
- **主播曝光**: 总观看次数、总观看时长、观看用户数
- **主播转化效率**: `gift_sessions / watch_sessions`, `revenue / watch_hour`
- **主播收入集中度**: Top1 用户占其收入的比例

### ⚠️ 口径风险警告

| 风险 | 描述 | 影响 | 解决方案 |
|------|------|------|---------|
| **gift_rate=100% 误导** | 旧报告提到"user.csv 仅包含有打赏用户"，导致用户级打赏率=100% | 高估用户付费意愿 | ✅ 使用 click.csv 全量计算 per-click 打赏率=1.48% |
| **缺少 session 级口径** | 所有转化率都是 click-level，无法理解"观看→打赏"路径 | 无法定位流失环节 | ✅ 构建 session 表，计算 session-level 转化率 |
| **分母不一致** | 不同指标使用不同分母（click vs session vs user） | 结论不可比 | ✅ 明确标注每个指标的分母和口径 |

---

# 2. Data Audit & Quality（数据质量与可用性）

## 2.1 Schema 总览

### 核心表结构

| 表名 | 行数 | 字段 | 主键候选 | 连接键 |
|------|------|------|---------|--------|
| `click.csv` | 4,909,515 | user_id, live_id, streamer_id, timestamp, watch_live_time | (user_id, live_id, timestamp) | user_id, streamer_id, live_id |
| `gift.csv` | 72,646 | user_id, live_id, streamer_id, timestamp, gift_price | (user_id, live_id, timestamp) | user_id, streamer_id, live_id |
| `user.csv` | 23,773 | user_id, age, gender, country, device_brand, device_price, reg_timestamp, fans_num, follow_num, first_watch_live_timestamp, accu_watch_live_cnt, accu_watch_live_duration, is_live_streamer, is_photo_author, onehot_feat0-6 | user_id | user_id |
| `streamer.csv` | 452,622 | streamer_id, gender, age, country, device_brand, device_price, live_operation_tag, fans_user_num, fans_group_fans_num, follow_user_num, first_live_timestamp, accu_live_cnt, accu_live_duration, accu_play_cnt, accu_play_duration, reg_timestamp, onehot_feat0-6 | streamer_id | streamer_id |
| `room.csv` | 11,819,965 | [待检查字段] | [待检查] | live_id, streamer_id |
| `comment.csv` | 196,527 | [待检查字段] | [待检查] | user_id, live_id, streamer_id |
| `like.csv` | 179,312 | [待检查字段] | [待检查] | user_id, live_id, streamer_id |
| `negative.csv` | 12,705,836 | [待检查字段] | [待检查] | user_id, live_id, streamer_id |

**字段类型**：
- `timestamp`: int64 (毫秒级 Unix 时间戳)
- `watch_live_time`: int64 (毫秒)
- `gift_price`: int64
- 其他字段多为字符串或分箱值

### 连接关系

```
click.csv ──[user_id]──> user.csv
         └─[streamer_id]──> streamer.csv
         └─[live_id]──> room.csv (假设)

gift.csv ──[user_id]──> user.csv
        └─[streamer_id]──> streamer.csv
        └─[live_id]──> room.csv (假设)
        └─[user_id, live_id, timestamp]──> click.csv (需时间邻近匹配)
```

## 2.2 时间范围与采样

| 检查项 | 方法 | 结果（待执行） |
|--------|------|---------------|
| **时间范围** | `min(timestamp)`, `max(timestamp)` | [待填写] |
| **时间跨度** | `(max - min) / (1000*3600*24)` 天 | [待填写] |
| **缺天检查** | 按天聚合，检查连续日期 | [待填写] |
| **缺小时检查** | 按小时聚合，检查 0-23 覆盖 | [待填写] |
| **时区问题** | 检查 timestamp 是否 UTC/本地时间 | [待填写] |
| **采样说明** | 是否为全量/采样数据 | [待填写] |

## 2.3 缺失/异常检查

### Click 表异常

| 检查项 | 方法 | 阈值 | 结果（待执行） |
|--------|------|------|---------------|
| **watch_live_time <= 0** | `count(watch_live_time <= 0)` | 0 | [待填写] |
| **watch_live_time 异常大** | `count(watch_live_time > 24h)` | <1% | [待填写] |
| **timestamp 逆序** | `count(timestamp[i] < timestamp[i-1])` | 0 | [待填写] |
| **重复记录** | `duplicated([user_id, live_id, timestamp])` | <0.1% | [待填写] |
| **缺失值** | `isna().sum()` | 0 | [待填写] |

### Gift 表异常

| 检查项 | 方法 | 阈值 | 结果（待执行） |
|--------|------|------|---------------|
| **gift_price <= 0** | `count(gift_price <= 0)` | 0 | [待填写] |
| **gift_price 异常值** | `count(gift_price > P99*10)` | <0.01% | [待填写] |
| **timestamp 逆序** | 同上 | 0 | [待填写] |
| **重复记录** | 同上 | <0.1% | [待填写] |
| **缺失值** | 同上 | 0 | [待填写] |

### 一致性检查

| 检查项 | 方法 | 结果（待执行） |
|--------|------|---------------|
| **Gift 能否映射到 Click** | 对每个 gift，查找同 `(user_id, live_id)` 且时间邻近（±5min）的 click | [待填写] |
| **无法映射比例** | `1 - mapped_gifts / total_gifts` | [待填写] |
| **可能原因** | 分析无法映射的 gift 特征 | [待填写] |

## 2.4 数据健康评分表

| 维度 | 评分 (0-5) | 说明 | 证据 |
|------|-----------|------|------|
| **Completeness** | [待填写] | 缺失值比例、覆盖度 | [待填写] |
| **Consistency** | [待填写] | Gift-Click 映射率、时间一致性 | [待填写] |
| **Validity** | [待填写] | 异常值比例、范围合理性 | [待填写] |
| **Timeliness** | [待填写] | 时间覆盖完整性、缺天/缺小时 | [待填写] |
| **总体评分** | [待填写] | 平均分 | [待填写] |

---

# 3. Entity & Sessionization（把行为串成"会话"）

## 3.1 Session 定义

**Session 构建规则**：
- **同一 session**: `user_id` + `live_id` 相同
- **Session 边界**: 
  - `session_start = timestamp`（click 记录的时间戳）
  - `session_end = timestamp + watch_live_time`
  - `session_duration = watch_live_time`（毫秒）
- **Session 合并规则**: 如果同一 `(user_id, live_id)` 有多个 click 记录，且时间间隔 < 5 分钟，合并为一个 session（待验证合理性）

**Session 级特征计算**：
- `session_watch_time`: 总观看时长（可能跨多个 click 合并）
- `session_gift_count`: session 内打赏次数
- `session_gift_amount`: session 内打赏总金额
- `t_first_gift`: 首次打赏时间（相对 session_start）
- `gift_intervals`: 多次打赏的间隔序列
- `is_bounce`: 是否跳出（watch_time < 阈值，如 10 秒）

## 3.2 关键研究问题

### Q3.1: 礼物是"即时冲动"还是"随观看累积"？

**分析方法**：
- 计算 `t_first_gift / session_duration` 的分布
- 如果大部分 `t_first_gift / session_duration < 0.1`，说明是"即时冲动"
- 如果 `t_first_gift / session_duration` 均匀分布或右偏，说明是"随观看累积"

**预期图表**：
- **Fig 3.1**: 首礼时间相对 session 的分布（`t_first_gift / session_duration` 直方图）
- **Fig 3.2**: 首礼时间 vs 观看时长的散点图（log-log scale）

### Q3.2: 多次打赏的节奏：burst 还是均匀？

**分析方法**：
- 对有多于 1 次打赏的 session，计算打赏间隔序列
- 分析间隔分布：如果大部分间隔 < 1 分钟，说明是"burst"；如果均匀分布，说明是"均匀"

**预期图表**：
- **Fig 3.3**: 打赏间隔分布（直方图，log-x scale）

## 3.3 Session 级关键指标

| 指标 | 定义 | 预期值（待执行） |
|------|------|----------------|
| **Session 总数** | `count(distinct (user_id, live_id))` | [待填写] |
| **平均 Session 时长** | `mean(session_duration)` | [待填写] |
| **Session 时长 P50/P90/P99** | 分位数 | [待填写] |
| **Session 转化率** | `gift_sessions / total_sessions` | [待填写] |
| **立即打赏比例** | `count(t_first_gift < 10s) / gift_sessions` | [待填写] |
| **多次打赏 Session 比例** | `count(gift_count > 1) / gift_sessions` | [待填写] |

## 3.4 预期图表清单

| Fig | 问题 | 图类型 | X/Y | 分群 | 读图结论（待执行） |
|-----|------|--------|-----|------|------------------|
| **3.1** | 首礼时间相对 session 的分布 | 直方图 | `t_first_gift / session_duration` | - | [待填写] |
| **3.2** | 首礼时间 vs 观看时长 | 散点图 | `watch_time` / `t_first_gift` | log-log | [待填写] |
| **3.3** | Session 时长分布 | CCDF | `session_duration` | - | [待填写] |
| **3.4** | 打赏间隔分布 | 直方图 | `gift_interval` | log-x | [待填写] |
| **3.5** | 转化漏斗 | 条形图 | 步骤 / 转化率 | - | [待填写] |

---

# 4. User Behavior（用户侧：观看→互动→付费）

## 4.1 用户观看强度分布

| 指标 | 定义 | 预期图表 |
|------|------|---------|
| **DAU** | 每日活跃用户数（如有日期） | 时间序列 |
| **人均观看 Session 数** | `mean(sessions_per_user)` | 分布直方图 |
| **人均观看时长** | `mean(total_watch_time_per_user)` | CCDF（长尾分布） |

**预期图表**：
- **Fig 4.1**: 用户观看时长分布（CCDF，log-log scale）
- **Fig 4.2**: 用户观看 Session 数分布（直方图）

## 4.2 付费漏斗（多套口径）

### 口径 A: Click-level 转化
- **Click → Gift**: 1.48% (已知)

### 口径 B: Session-level 转化
- **Session → Gift Session**: `gift_sessions / total_sessions`（待计算）
- **Gift Session → Multi-gift Session**: `multi_gift_sessions / gift_sessions`（待计算）

### 口径 C: User-level 转化
- **User → Paid User**: `paid_users / total_users`（待计算，需注意 user.csv 可能只包含有打赏用户）

**预期图表**：
- **Fig 4.3**: 转化漏斗（条形图，多套口径并排）

## 4.3 用户分群

### 分群 1: 新/老用户
- **定义**: 按 `first_watch_live_timestamp` 或首次出现日期
- **阈值**: 新用户 = 首次出现 < 7 天

### 分群 2: 轻/中/重度观看
- **定义**: 按 `total_watch_time` 分位
- **阈值**: 轻=0-33%, 中=33-66%, 重=66-100%

### 分群 3: 轻/中/重度付费
- **定义**: 按 `total_gift_amount` 或 `gift_count` 分位
- **阈值**: 同上

### 分群 4: 观看-付费二维象限
- **X轴**: 总观看时长（log scale）
- **Y轴**: 总打赏金额（log scale）
- **四象限**:
  - 高观看高付费（理想用户）
  - 高观看低付费（待转化）
  - 低观看高付费（冲动付费？）
  - 低观看低付费（普通用户）

**预期图表**：
- **Fig 4.4**: 用户二维象限（散点图，log-log，标注四象限占比）

## 4.4 关键研究问题

### Q4.1: watch_time 与 gift_prob 的关系是单调还是阈值型？

**分析方法**：
- 将 `watch_time` 分箱（如 0-10s, 10-30s, 30-1min, 1-5min, 5-15min, 15min+）
- 计算每个箱的 `gift_rate`
- 绘制 `watch_time_bin` vs `gift_rate` 曲线

**预期图表**：
- **Fig 4.5**: watch_time 分位数分箱的 gift_rate 曲线（可加置信区间）

### Q4.2: "高观看低付费"人群是否存在？规模多大？他们看什么主播？

**分析方法**：
- 定义"高观看低付费": `total_watch_time > P75` 且 `total_gift_amount < P25`
- 计算该人群规模
- 分析他们观看的主播分布（Top 主播、类别等）

**预期图表**：
- **Fig 4.6**: 高观看低付费人群的主播偏好（条形图，Top 20 主播）

### Q4.3: 付费用户的回访/连续天活跃是否更高？（如有日粒度）

**分析方法**：
- 计算付费用户 vs 非付费用户的 DAU 比例、连续活跃天数
- 如有日期字段，计算留存率

**预期图表**：
- **Fig 4.7**: 付费用户 vs 非付费用户的留存曲线（如有日期）

### Q4.4: 礼物金额是否存在典型价位阶梯？

**分析方法**：
- 统计 `gift_price` 的离散值分布
- 识别高频价格点（如 1, 2, 6, 12, 88, 520 等）
- 计算每个价格点的累计贡献

**预期图表**：
- **Fig 4.8**: 付费价位阶梯（条形图，Top 20 价位点 + 累计贡献）

### Q4.5: 多次打赏用户的"打赏间隔"和"复购周期"是什么？

**分析方法**：
- 对多次打赏用户，计算打赏间隔分布
- 如有日期，计算"复购周期"（同一用户两次付费的间隔天数）

**预期图表**：
- **Fig 4.9**: 打赏间隔分布（直方图）
- **Fig 4.10**: 复购周期分布（如有日期）

## 4.5 用户侧"可行动"建议（待执行后填写）

| 人群 | 特征 | 策略建议 | 优先级 |
|------|------|---------|--------|
| 高观看低付费 | [待填写] | [待填写] | [待填写] |
| 低观看高付费 | [待填写] | [待填写] | [待填写] |
| 新用户 | [待填写] | [待填写] | [待填写] |

---

# 5. Supply Side（主播/直播间侧：供给结构与质量）

## 5.1 主播曝光与观看

| 指标 | 定义 | 预期图表 |
|------|------|---------|
| **每主播被观看次数** | `count(click) per streamer` | 分布（长尾） |
| **每主播总观看时长** | `sum(watch_live_time) per streamer` | 分布（长尾） |
| **每主播观看用户数** | `count(distinct user_id) per streamer` | 分布（长尾） |

**预期图表**：
- **Fig 5.1**: 主播观看时长分布（CCDF，log-log scale）

## 5.2 主播转化效率

| 指标 | 定义 | 说明 |
|------|------|------|
| **Session 转化率** | `gift_sessions / watch_sessions` | 每 session 转化概率 |
| **Revenue per Watch Hour** | `total_revenue / total_watch_hours` | 单位观看时长收益 |
| **Revenue per Viewer** | `total_revenue / unique_viewers` | 人均收益 |

**预期图表**：
- **Fig 5.2**: 主播转化效率分布（直方图，log-x scale）

## 5.3 主播二维象限分析

**二维象限定义**：
- **X轴**: 总观看时长 或 观看用户数（log scale）
- **Y轴**: 总收入 或 转化率（log scale）

**四象限**：
- **高吸引高变现**（双高）: 头部主播，理想状态
- **高吸引低变现**（待优化）: 有流量但转化差，需优化内容/互动
- **低吸引高变现**（精准）: 小众但高价值，适合精准推荐
- **低吸引低变现**（双低）: 长尾主播，需扶持或淘汰

**预期图表**：
- **Fig 5.3**: 主播二维象限（散点图，log-log，标注四象限占比）
- **Fig 5.4**: 主播二维象限（双面板：一张 X=观看时长 Y=收入，一张 X=观看用户数 Y=转化率）

## 5.4 冷启动细分

**冷启动定义**：
- **新主播**: 首次直播 < 7 天 或 曝光 < 阈值
- **低曝光主播**: 总观看次数 < P25

**分析内容**：
- 新主播/低曝光主播的 `revenue per watch_hour` 分布
- 对比头部主播，识别差距

**预期图表**：
- **Fig 5.5**: 主播冷启动分层（箱线图：按曝光分位的 `revenue per watch_hour`）

## 5.5 供给侧策略建议（待执行后填写）

| 主播类型 | 特征 | 策略建议 | 优先级 |
|---------|------|---------|--------|
| 高吸引低变现 | [待填写] | [待填写] | [待填写] |
| 低吸引高变现 | [待填写] | [待填写] | [待填写] |
| 冷启动主播 | [待填写] | [待填写] | [待填写] |

---

# 6. Interaction Structure（用户-主播匹配结构 / 网络视角）

## 6.1 二部图构建

**图定义**：
- **节点**: User (左侧) + Streamer (右侧)
- **边**: User-Streamer 交互
- **边权**（三套）:
  - `watch_time`: 总观看时长
  - `gift_count`: 打赏次数
  - `gift_amount`: 打赏金额

## 6.2 度分布分析

| 指标 | 定义 | 预期图表 |
|------|------|---------|
| **用户度** | 用户看过多少主播（watch 与 gift 分别统计） | 分布直方图 |
| **主播度** | 主播被多少用户看过（watch 与 gift 分别统计） | 分布直方图 |

**预期图表**：
- **Fig 6.1**: 用户度分布（直方图，log-log scale，watch vs gift 对比）
- **Fig 6.2**: 主播度分布（直方图，log-log scale，watch vs gift 对比）

## 6.3 "忠诚度/专一度"分析

### 用户专一度
- **定义**: Top1 主播占其观看/打赏的比例
- **计算**: 
  - `watch_loyalty = max(watch_time_per_streamer) / total_watch_time`
  - `gift_loyalty = max(gift_amount_per_streamer) / total_gift_amount`

### 主播收入集中度
- **定义**: Top1 用户占其收入的比例
- **计算**: `revenue_concentration = max(gift_amount_per_user) / total_revenue`

**预期图表**：
- **Fig 6.3**: 用户专一度分布（直方图，watch 与 gift 两套）
- **Fig 6.4**: 主播收入集中度分解（Lorenz 曲线 + Top1 用户占比散点）

## 6.4 "替代性/多样性"分析

### 用户观看多样性熵
- **定义**: Shannon entropy of `watch_time_per_streamer` distribution
- **公式**: $H = -\sum_i p_i \log p_i$，其中 $p_i = \text{watch_time}_i / \text{total_watch_time}$

### 主播观众多样性熵
- **定义**: Shannon entropy of `viewers` distribution（按观看时长加权）

**预期图表**：
- **Fig 6.5**: 用户多样性熵分布（直方图）

## 6.5 关键研究问题

### Q6.1: 付费是否强绑定于特定主播？

**分析方法**：
- 计算用户 `gift_loyalty` 分布
- 如果大部分用户 `gift_loyalty > 0.8`，说明付费强绑定
- 如果 `gift_loyalty` 均匀分布，说明付费分散

**预期结论**：
- [待执行后填写]

### Q6.2: 是否存在"跨主播付费迁移"？

**分析方法**：
- 计算"在多个主播付费的用户"比例
- 分析这些用户的付费分布（是否均匀）

**预期结论**：
- [待执行后填写]

## 6.6 对推荐/分配策略的启示（待执行后填写）

| 观察 | 启示 | 策略建议 |
|------|------|---------|
| 偏好稳定（高专一度） | [待填写] | 重视 pair 特征 |
| 付费迁移存在 | [待填写] | 可做相似主播扩展 |
| [待填写] | [待填写] | [待填写] |

---

# 7. Time & Seasonality（时序、周期、事件）

## 7.1 小时×星期热力图

**分析维度**：
- `watch_count`: 观看次数
- `watch_time`: 观看时长
- `gift_count`: 打赏次数
- `gift_amount`: 打赏金额
- `conversion_rate`: 转化率（gift_count / watch_count）

**预期图表**：
- **Fig 7.1**: 7×24 热力图（三张并排：watch_time, gift_amount, conversion_rate）

## 7.2 "黄金时段"识别

**定义**：不是只看 `gift_count`，而是多指标综合：
- **观看高峰**: `watch_count` 峰值
- **打赏高峰**: `gift_count` 峰值
- **转化高峰**: `conversion_rate` 峰值
- **收益高峰**: `gift_amount` 峰值

**预期结论**：
- [待执行后填写]

## 7.3 事件/异常峰分析

**分析方法**：
- 如有日期字段，按天聚合，识别极端峰值日
- 分析峰值日由哪些主播/活动驱动

**预期图表**：
- **Fig 7.2**: 峰值贡献分解（堆叠面积图，Top N 主播在峰值时段的贡献）

## 7.4 可用的时间特征清单（待执行后填写）

| 特征 | 定义 | 用途 |
|------|------|------|
| `hour` | 小时 (0-23) | 建模输入 |
| `dow` | 星期几 (0-6) | 建模输入 |
| `is_weekend` | 是否周末 | 建模输入 |
| `recent_trend` | 近期趋势（滑动窗口统计） | 建模输入 |
| [待补充] | [待补充] | [待补充] |

---

# 8. Anomaly & Risk（异常/作弊/污染）

## 8.1 异常用户检测

### 规则 1: 极高打赏但极短观看
- **定义**: `gift_amount > P99` 且 `watch_time < 10s`
- **阈值**: 可疑比例 < 0.1%

### 规则 2: 礼物间隔极小
- **定义**: `gift_interval < 1s`（同一 session 内）
- **阈值**: 可疑比例 < 0.1%

### 规则 3: 多主播瞬时扫礼
- **定义**: 同一用户在 < 1 分钟内对 > 5 个主播打赏
- **阈值**: 可疑比例 < 0.01%

**预期图表**：
- **Fig 8.1**: 异常规则可视化（散点图：gift_interval vs gift_amount，标注可疑区域）

## 8.2 异常主播检测

### 规则 1: 收入异常集中于极少用户
- **定义**: Top1 用户占比 > 0.9
- **阈值**: 可疑主播比例 < 1%

### 规则 2: 异常价格点
- **定义**: 大部分打赏金额为同一异常值（如 9999, 8888）
- **阈值**: 单一价格占比 > 0.8

### 规则 3: 可疑时间分布
- **定义**: 打赏时间集中在特定异常时段（如凌晨 2-4 点）
- **阈值**: 需人工判断

**预期结论**：
- [待执行后填写]

## 8.3 日志污染检测

### 规则 1: 重复上报
- **定义**: 完全相同的记录（所有字段相同）
- **阈值**: 重复率 < 0.1%

### 规则 2: 机器人点击
- **定义**: `watch_time` 分布异常（如大量 0 或固定值）
- **阈值**: 需人工判断

### 规则 3: watch_time 不合理
- **定义**: `watch_time <= 0` 或 `watch_time > 24h`
- **阈值**: 异常比例 < 1%

**预期结论**：
- [待执行后填写]

## 8.4 异常检测规则输出（待执行后填写）

| 规则 | 定义 | 阈值 | 可疑样本数 | 处理建议 |
|------|------|------|-----------|---------|
| 极高打赏极短观看 | [待填写] | [待填写] | [待填写] | [待填写] |
| 礼物间隔极小 | [待填写] | [待填写] | [待填写] | [待填写] |
| 多主播瞬时扫礼 | [待填写] | [待填写] | [待填写] | [待填写] |
| 收入异常集中 | [待填写] | [待填写] | [待填写] | [待填写] |

**后续可训练模型方向**：
- [待执行后填写]

---

# 9. Modeling & Product Implications（把洞见翻译成决策）

## 9.1 预测任务定义

| 任务 | 定义 | 窗口 | 评估指标 |
|------|------|------|---------|
| **Gift Probability** | 是否打赏 | Session-level / User-level | PR-AUC, Top-K Capture |
| **Gift Amount** | 打赏金额 | Session-level / User-level | MAE(log), Spearman |
| **Expected Value (EV)** | 期望收益 = p × m | Session-level / User-level | Top-K Capture, NDCG |
| **Long-term Value** | 长期价值（如有留存数据） | User-level | [待补充] |

## 9.2 建模建议（3~5 条）

| # | 建议 | 理由 | 优先级 |
|---|------|------|--------|
| 1 | **标签定义**: Session-level `is_gift` + `gift_amount` | 更符合业务逻辑，避免 click-level 稀疏 | 🔴 P0 |
| 2 | **目标变换**: `log(1+Y)` 或分位数回归 | 重尾分布，P99/P50=744x | 🔴 P0 |
| 3 | **评估指标**: PR-AUC (概率) + Top-K Capture (排序) + Spearman (金额) | 稀疏场景 + 排序任务 | 🔴 P0 |
| 4 | **冷启动处理**: 分层模型（有交互历史 vs 无交互历史） | 92.2% 主播无打赏记录 | 🟡 P1 |
| 5 | **稀疏处理**: 负采样 + 加权损失 | Matrix density=0.0064% | 🔴 P0 |

## 9.3 产品/策略建议（3~5 条）

| # | 建议 | 理由 | 成功标准 |
|---|------|------|---------|
| 1 | **触发策略**: 对"高观看低付费"人群，在观看时长 > 阈值时触发打赏提示 | [待执行后填写] | 转化率提升 > 10% |
| 2 | **分配约束**: 软约束冷启动（λ=0.5） | 已有证据：收益+32%，成功率+263% | [已有] |
| 3 | **生态公平指标**: 监控主播侧 Gini/HHI，设置护栏阈值 | 避免马太效应过度 | Gini < 0.95 |
| 4 | [待执行后填写] | [待执行后填写] | [待执行后填写] |
| 5 | [待执行后填写] | [待执行后填写] | [待执行后填写] |

---

# 10. Appendix（复现与图表索引）

## 10.1 所有图表清单

| Fig | 问题 | 图类型 | X/Y | 分群 | 读图结论（待执行） | 生成代码入口 |
|-----|------|--------|-----|------|------------------|------------|
| **2.1** | 时间覆盖 & 缺天 | 折线图 | 日期 / 样本量 | click vs gift | [待填写] | `plot_time_coverage()` |
| **2.2** | watch_time 异常分布 | 直方图 | `watch_time` | 负值/0占比标注 | [待填写] | `plot_watch_time_anomaly()` |
| **3.1** | 首礼时间相对 session 的分布 | 直方图 | `t_first_gift / session_duration` | - | [待填写] | `plot_first_gift_time()` |
| **3.2** | 首礼时间 vs 观看时长 | 散点图 | `watch_time` / `t_first_gift` | log-log | [待填写] | `plot_first_gift_vs_watch()` |
| **3.3** | Session 时长分布 | CCDF | `session_duration` | P50/P90/P99标注 | [待填写] | `plot_session_duration_ccdf()` |
| **3.4** | 打赏间隔分布 | 直方图 | `gift_interval` | log-x | [待填写] | `plot_gift_interval()` |
| **3.5** | 转化漏斗 | 条形图 | 步骤 / 转化率 | click→session→gift | [待填写] | `plot_conversion_funnel()` |
| **4.1** | 用户观看时长分布 | CCDF | `total_watch_time` | log-log | [待填写] | `plot_user_watch_time_ccdf()` |
| **4.2** | 用户观看 Session 数分布 | 直方图 | `sessions_per_user` | - | [待填写] | `plot_user_sessions()` |
| **4.3** | 转化漏斗（多口径） | 条形图 | 口径 / 转化率 | A/B/C | [待填写] | `plot_conversion_funnel_multi()` |
| **4.4** | 用户二维象限 | 散点图 | `total_watch_time` / `total_gift_amount` | log-log, 四象限 | [待填写] | `plot_user_quadrant()` |
| **4.5** | watch_time 分位数分箱的 gift_rate | 曲线图 | `watch_time_bin` / `gift_rate` | 置信区间 | [待填写] | `plot_watch_time_vs_gift_rate()` |
| **4.6** | 高观看低付费人群的主播偏好 | 条形图 | 主播 / 观看次数 | Top 20 | [待填写] | `plot_high_watch_low_pay_streamers()` |
| **4.7** | 付费用户 vs 非付费用户留存 | 曲线图 | 天数 / 留存率 | 付费 vs 非付费 | [待填写] | `plot_retention_by_payment()` |
| **4.8** | 付费价位阶梯 | 条形图 | 价格点 / 频次+累计贡献 | Top 20 | [待填写] | `plot_gift_price_tiers()` |
| **4.9** | 打赏间隔分布 | 直方图 | `gift_interval` | - | [待填写] | `plot_gift_interval_dist()` |
| **4.10** | 复购周期分布 | 直方图 | `days_between_gifts` | - | [待填写] | `plot_repurchase_cycle()` |
| **5.1** | 主播观看时长分布 | CCDF | `total_watch_time` | log-log | [待填写] | `plot_streamer_watch_time_ccdf()` |
| **5.2** | 主播转化效率分布 | 直方图 | `revenue_per_watch_hour` | log-x | [待填写] | `plot_streamer_conversion_efficiency()` |
| **5.3** | 主播二维象限（观看时长 vs 收入） | 散点图 | `total_watch_time` / `total_revenue` | log-log, 四象限 | [待填写] | `plot_streamer_quadrant_revenue()` |
| **5.4** | 主播二维象限（观看用户数 vs 转化率） | 散点图 | `unique_viewers` / `conversion_rate` | log-log, 四象限 | [待填写] | `plot_streamer_quadrant_conversion()` |
| **5.5** | 主播冷启动分层 | 箱线图 | 曝光分位 / `revenue_per_watch_hour` | - | [待填写] | `plot_streamer_coldstart_tiers()` |
| **6.1** | 用户度分布 | 直方图 | 度 / 频次 | log-log, watch vs gift | [待填写] | `plot_user_degree_dist()` |
| **6.2** | 主播度分布 | 直方图 | 度 / 频次 | log-log, watch vs gift | [待填写] | `plot_streamer_degree_dist()` |
| **6.3** | 用户专一度分布 | 直方图 | `loyalty` | watch 与 gift 两套 | [待填写] | `plot_user_loyalty_dist()` |
| **6.4** | 主播收入集中度分解 | Lorenz + 散点 | 主播排序 / 累计收入占比 | Top1用户占比 | [待填写] | `plot_streamer_revenue_concentration()` |
| **6.5** | 用户多样性熵分布 | 直方图 | `entropy` | - | [待填写] | `plot_user_diversity_entropy()` |
| **7.1** | 7×24 热力图 | 热力图 | 小时×星期 | watch_time, gift_amount, conversion_rate | [待填写] | `plot_hour_dow_heatmap()` |
| **7.2** | 峰值贡献分解 | 堆叠面积图 | 日期 / 贡献 | Top N 主播 | [待填写] | `plot_peak_contribution()` |
| **8.1** | 异常规则可视化 | 散点图 | `gift_interval` / `gift_amount` | 可疑区域标注 | [待填写] | `plot_anomaly_rules()` |
| **8.2** | 用户-主播边权分布对比 | 直方图 | 边权 / 频次 | watch vs gift | [待填写] | `plot_edge_weight_comparison()` |

**总计**: 至少 30 张图表（用户要求至少 12 张 Hero 图）

## 10.2 关键中间表 Schema

### Session 表
```sql
CREATE TABLE sessions AS (
  SELECT 
    user_id,
    live_id,
    streamer_id,
    MIN(timestamp) AS session_start,
    MAX(timestamp + watch_live_time) AS session_end,
    SUM(watch_live_time) AS session_duration,
    COUNT(*) AS click_count,
    COUNT(DISTINCT gift.timestamp) AS gift_count,
    SUM(gift.gift_price) AS gift_amount,
    MIN(gift.timestamp) AS t_first_gift
  FROM click
  LEFT JOIN gift ON (
    click.user_id = gift.user_id 
    AND click.live_id = gift.live_id
    AND gift.timestamp BETWEEN click.timestamp AND click.timestamp + click.watch_live_time
  )
  GROUP BY user_id, live_id, streamer_id
);
```

### 聚合口径
- **User-level**: `GROUP BY user_id`
- **Streamer-level**: `GROUP BY streamer_id`
- **Session-level**: `GROUP BY user_id, live_id`
- **Time-level**: `GROUP BY DATE(timestamp), HOUR(timestamp)`

## 10.3 后续要补的数据字段清单

| 优先级 | 字段 | 用途 | 当前状态 |
|--------|------|------|---------|
| **P0** | 日期字段（如 `date`） | 留存/回访分析、事件峰分析 | ⚠️ 需从 timestamp 提取 |
| **P0** | Session ID | 直接标识 session，避免构建 | ❌ 缺失，需构建 |
| **P1** | 用户注册日期 | 新/老用户分群 | ⚠️ user.csv 有 `reg_timestamp`，需验证 |
| **P1** | 主播首次直播日期 | 冷启动分析 | ⚠️ streamer.csv 有 `first_live_timestamp`，需验证 |
| **P1** | 直播间类别/标签 | 供给侧质量分析 | ⚠️ streamer.csv 有 `live_operation_tag`，需验证 |
| **P2** | 用户画像（年龄/性别/设备） | 用户分群细化 | ✅ user.csv 已有 |
| **P2** | 主播画像（年龄/性别/设备） | 供给侧分析 | ✅ streamer.csv 已有 |

## 10.4 最可能的 10 条新洞见候选

| # | 洞见候选 | 验证方法 | 所需字段 |
|---|---------|---------|---------|
| 1 | **打赏是"即时冲动"（首礼时间 < 10% session）** | Fig 3.1 分布 | Session 表 |
| 2 | **"高观看低付费"人群占用户 20-30%** | Fig 4.4 象限占比 | User-level 聚合 |
| 3 | **主播存在"高吸引低变现"类型（占主播 10-20%）** | Fig 5.3 象限占比 | Streamer-level 聚合 |
| 4 | **付费强绑定于特定主播（用户专一度 > 0.8 占 60%+）** | Fig 6.3 分布 | User-Streamer 聚合 |
| 5 | **存在典型价位阶梯（1, 2, 6, 12, 88, 520 等）** | Fig 4.8 价格点分布 | Gift 表 |
| 6 | **watch_time 与 gift_prob 呈阈值型关系（>5min 转化率显著提升）** | Fig 4.5 曲线 | Session 表 |
| 7 | **黄金时段是 20:00-22:00（多指标综合）** | Fig 7.1 热力图 | Time 聚合 |
| 8 | **异常用户占 < 0.1%（极高打赏极短观看）** | Fig 8.1 散点图 | Session + Gift 表 |
| 9 | **冷启动主播 revenue_per_watch_hour 低于头部 50%+** | Fig 5.5 箱线图 | Streamer-level 聚合 |
| 10 | **用户观看多样性熵与付费金额正相关** | 相关性分析 | User-level 聚合 |

---

# 11. 图表生成任务列表（可直接给工程/脚本）

> ⚠️ **待执行**: 每条包含输入表、SQL/pandas 聚合口径、图类型、保存路径

| 任务ID | 输入表 | 聚合口径 | 图类型 | 保存路径 | 优先级 |
|--------|--------|---------|--------|---------|--------|
| T2.1 | click, gift | `GROUP BY DATE(timestamp)` | 折线图 | `img/time_coverage.png` | 🔴 P0 |
| T2.2 | click | `watch_live_time` 直方图 + 异常标注 | 直方图 | `img/watch_time_anomaly.png` | 🔴 P0 |
| T3.1 | sessions | `t_first_gift / session_duration` 分布 | 直方图 | `img/first_gift_time_ratio.png` | 🔴 P0 |
| T3.2 | sessions | `watch_time` vs `t_first_gift` 散点 | 散点图 | `img/first_gift_vs_watch.png` | 🔴 P0 |
| T3.3 | sessions | `session_duration` CCDF | CCDF | `img/session_duration_ccdf.png` | 🔴 P0 |
| T3.4 | sessions (multi-gift) | `gift_interval` 分布 | 直方图 | `img/gift_interval_dist.png` | 🟡 P1 |
| T3.5 | click, sessions, gift | 转化漏斗 | 条形图 | `img/conversion_funnel.png` | 🔴 P0 |
| T4.1 | user-level | `total_watch_time` CCDF | CCDF | `img/user_watch_time_ccdf.png` | 🔴 P0 |
| T4.2 | user-level | `sessions_per_user` 分布 | 直方图 | `img/user_sessions_dist.png` | 🟡 P1 |
| T4.3 | click, sessions, user | 多口径转化率 | 条形图 | `img/conversion_funnel_multi.png` | 🔴 P0 |
| T4.4 | user-level | `total_watch_time` vs `total_gift_amount` | 散点图 | `img/user_quadrant.png` | 🔴 P0 |
| T4.5 | sessions | `watch_time_bin` vs `gift_rate` | 曲线图 | `img/watch_time_vs_gift_rate.png` | 🔴 P0 |
| T4.6 | user-level (high-watch-low-pay) | Top 20 主播观看次数 | 条形图 | `img/high_watch_low_pay_streamers.png` | 🟡 P1 |
| T4.7 | user-level (如有日期) | 留存曲线 | 曲线图 | `img/retention_by_payment.png` | 🟡 P1 |
| T4.8 | gift | `gift_price` 离散值分布 | 条形图 | `img/gift_price_tiers.png` | 🔴 P0 |
| T5.1 | streamer-level | `total_watch_time` CCDF | CCDF | `img/streamer_watch_time_ccdf.png` | 🔴 P0 |
| T5.2 | streamer-level | `revenue_per_watch_hour` 分布 | 直方图 | `img/streamer_conversion_efficiency.png` | 🔴 P0 |
| T5.3 | streamer-level | `total_watch_time` vs `total_revenue` | 散点图 | `img/streamer_quadrant_revenue.png` | 🔴 P0 |
| T5.4 | streamer-level | `unique_viewers` vs `conversion_rate` | 散点图 | `img/streamer_quadrant_conversion.png` | 🔴 P0 |
| T5.5 | streamer-level | 按曝光分位的 `revenue_per_watch_hour` | 箱线图 | `img/streamer_coldstart_tiers.png` | 🟡 P1 |
| T6.1 | user-streamer | 用户度分布（watch vs gift） | 直方图 | `img/user_degree_dist.png` | 🟡 P1 |
| T6.2 | user-streamer | 主播度分布（watch vs gift） | 直方图 | `img/streamer_degree_dist.png` | 🟡 P1 |
| T6.3 | user-streamer | 用户专一度分布 | 直方图 | `img/user_loyalty_dist.png` | 🔴 P0 |
| T6.4 | streamer-level | Lorenz 曲线 + Top1 用户占比 | 组合图 | `img/streamer_revenue_concentration.png` | 🟡 P1 |
| T6.5 | user-streamer | 用户多样性熵分布 | 直方图 | `img/user_diversity_entropy.png` | 🟡 P1 |
| T7.1 | click, gift | `HOUR(timestamp)` × `DOW(timestamp)` 热力图 | 热力图 | `img/hour_dow_heatmap.png` | 🔴 P0 |
| T7.2 | gift (如有日期) | 峰值日贡献分解 | 堆叠面积图 | `img/peak_contribution.png` | 🟡 P1 |
| T8.1 | sessions, gift | `gift_interval` vs `gift_amount` 散点 | 散点图 | `img/anomaly_rules.png` | 🔴 P0 |
| T8.2 | user-streamer | 边权分布对比（watch vs gift） | 直方图 | `img/edge_weight_comparison.png` | 🟡 P1 |

**总计**: 28 个图表生成任务（至少 12 个 P0 优先级）

---

> **实验计划创建时间**: 2026-01-09  
> **下一步**: 执行图表生成任务，填充所有"待执行"部分
