# 🤖 Coding Prompt: KuaiLive 全方位探索性数据分析（Comprehensive EDA）

> **Experiment ID:** `EXP-20260109-gift-allocation-02`  
> **MVP:** MVP-0.1-Enhanced  
> **Date:** 2026-01-09  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：对 KuaiLive 数据做"全方位、专业、可落地"的探索性数据分析，从用户行为、直播间供给、互动结构、时序规律、稀疏性/冷启动、异常与数据质量等多个维度挖出可行动的洞见，产出高质量图表方案（每张图必须能"一眼得到结论"）。

**验证假设**：
- H2.1: 打赏是"即时冲动"还是"随观看累积"？（首礼时间分布）
- H2.2: 高观看低付费人群是否存在？规模多大？
- H2.3: 主播供给是否存在"高吸引低变现"类型？
- H2.4: 付费是否强绑定于特定主播？（用户专一度）
- H2.5: 数据是否存在异常/作弊/污染？

**预期结果**：
- 至少 12 张 Hero 图（每张图能"一眼得到结论"）
- 28 个图表生成任务完成（12 个 P0 优先级）
- 5-8 条可行动洞见（每条=结论 + 证据图编号 + So what）
- 数据质量评分表完整
- Session 表构建完成
- 所有"待执行"部分填充完成

---

## 2. 🧪 实验设定

### 2.1 数据

```yaml
data:
  source: "KuaiLive"
  path: "data/KuaiLive/"
  files:
    - click.csv       # 4,909,515 行：user_id, live_id, streamer_id, timestamp, watch_live_time
    - gift.csv        # 72,646 行：user_id, live_id, streamer_id, timestamp, gift_price
    - user.csv        # 23,773 行：user_id, age, gender, country, device_brand, device_price, reg_timestamp, fans_num, follow_num, first_watch_live_timestamp, accu_watch_live_cnt, accu_watch_live_duration, is_live_streamer, is_photo_author, onehot_feat0-6
    - streamer.csv    # 452,622 行：streamer_id, gender, age, country, device_brand, device_price, live_operation_tag, fans_user_num, fans_group_fans_num, follow_user_num, first_live_timestamp, accu_live_cnt, accu_live_duration, accu_play_cnt, accu_play_duration, reg_timestamp, onehot_feat0-6
    - room.csv        # 11,819,965 行（需检查字段）
    - comment.csv     # 196,527 行（需检查字段）
    - like.csv        # 179,312 行（需检查字段）
    - negative.csv   # 12,705,836 行（需检查字段）
```

### 2.2 核心任务

```yaml
tasks:
  # Phase 1: 数据质量与 Session 构建
  - task: "Data Audit & Quality"
    outputs:
      - 时间范围与采样检查
      - 缺失/异常检查（watch_live_time <= 0, gift_price 异常值等）
      - Gift-Click 一致性检查（映射率）
      - 数据健康评分表（Completeness/Consistency/Validity/Timeliness）
  
  - task: "Session 构建"
    definition: "同一 user_id + live_id 的连续观看窗口"
    schema:
      - user_id, live_id, streamer_id
      - session_start, session_end, session_duration
      - click_count, gift_count, gift_amount
      - t_first_gift (相对 session_start)
      - gift_intervals (多次打赏的间隔序列)
    merge_rule: "同一 (user_id, live_id) 多个 click 且时间间隔 < 5 分钟，合并为一个 session"
  
  # Phase 2: 多维度分析
  - task: "User Behavior Analysis"
    dimensions:
      - 用户观看强度分布（DAU, 人均 session 数, 人均观看时长）
      - 付费漏斗（Click-level, Session-level, User-level 三套口径）
      - 用户分群（新/老, 轻/中/重度观看, 轻/中/重度付费, 观看-付费二维象限）
      - watch_time vs gift_prob 关系（阈值型 vs 单调）
      - 付费价位阶梯分析
  
  - task: "Supply Side Analysis"
    dimensions:
      - 主播曝光与观看（观看次数, 观看时长, 观看用户数）
      - 主播转化效率（gift_sessions / watch_sessions, revenue per watch_hour）
      - 主播二维象限（观看时长 vs 收入, 观看用户数 vs 转化率）
      - 冷启动细分（新主播/低曝光主播的表现分布）
  
  - task: "Interaction Structure Analysis"
    dimensions:
      - 二部图构建（User-Streamer，边权=watch_time/gift_count/gift_amount）
      - 度分布（用户度, 主播度，区分 watch 与 gift）
      - 用户专一度（Top1 主播占其观看/打赏的比例）
      - 主播收入集中度（Top1 用户占其收入的比例）
      - 用户观看多样性熵（Shannon entropy）
  
  - task: "Time & Seasonality Analysis"
    dimensions:
      - 7×24 热力图（watch_count, watch_time, gift_count, gift_amount, conversion_rate）
      - 黄金时段识别（多指标综合）
      - 峰值贡献分解（如有日期）
  
  - task: "Anomaly & Risk Detection"
    rules:
      - 异常用户：极高打赏但极短观看（gift_amount > P99 且 watch_time < 10s）
      - 礼物间隔极小（gift_interval < 1s）
      - 多主播瞬时扫礼（同一用户 < 1 分钟内对 > 5 个主播打赏）
      - 异常主播：收入异常集中于极少用户（Top1 用户占比 > 0.9）
      - 异常价格点（单一价格占比 > 0.8）
      - 日志污染：重复上报、watch_time 不合理
```

### 2.3 口径冻结

```yaml
metrics:
  click_level:
    gift_rate: "gift_count / click_count = 72,646 / 4,909,515 = 1.48%"
    note: "事件级转化率"
  
  session_level:
    session_definition: "同一 user_id + live_id 的连续观看窗口"
    session_conversion: "gift_sessions / total_sessions"
    first_gift_time: "t_first_gift - session_start (相对 session 开始)"
  
  user_level:
    watch_intensity: "总观看时长, 总 session 数"
    pay_intensity: "总打赏金额, 总打赏次数, 付费率"
    loyalty: "Top1 主播占其观看/打赏的比例"
  
  streamer_level:
    exposure: "总观看次数, 总观看时长, 观看用户数"
    conversion_efficiency: "gift_sessions / watch_sessions, revenue / watch_hour"
    revenue_concentration: "Top1 用户占其收入的比例"
```

---

## 3. 📊 要画的图

> ⚠️ **图表要求**：
> - 所有文字必须英文
> - 包含 legend、title、axis labels
> - 分辨率 ≥ 300 dpi
> - **figsize 规则**：单张图 `figsize=(6, 5)` 锁死，多张图按 6:5 比例扩增
> - 每张图标题必须是"结论句"（例如：'Conversion is thresholded by watch time'），不是描述句
> - 图上直接写关键数字（P50/P99、Top-share、转化率、样本量 N）

### P0 优先级图表（至少 12 张 Hero 图）

| 图号 | 图表类型 | X轴 | Y轴 | 分群/标注 | 保存路径 | 问题 |
|------|---------|-----|-----|---------|---------|------|
| **Fig 2.1** | line | Date | Sample Count | click vs gift | `gift_allocation/img/time_coverage.png` | 时间覆盖 & 缺天 |
| **Fig 2.2** | histogram | watch_live_time | Count | 负值/0占比标注 | `gift_allocation/img/watch_time_anomaly.png` | watch_time 异常分布 |
| **Fig 3.1** | histogram | t_first_gift / session_duration | Count | - | `gift_allocation/img/first_gift_time_ratio.png` | 首礼时间相对 session 的分布 |
| **Fig 3.2** | scatter | watch_time (log) | t_first_gift (log) | - | `gift_allocation/img/first_gift_vs_watch.png` | 首礼时间 vs 观看时长 |
| **Fig 3.3** | CCDF | session_duration | P(session_duration > x) | P50/P90/P99标注 | `gift_allocation/img/session_duration_ccdf.png` | Session 时长分布 |
| **Fig 3.5** | bar | Step | Conversion Rate (%) | click→session→gift_session→multi_gift | `gift_allocation/img/conversion_funnel.png` | 转化漏斗 |
| **Fig 4.1** | CCDF | total_watch_time (log) | P(watch_time > x) (log) | - | `gift_allocation/img/user_watch_time_ccdf.png` | 用户观看时长分布 |
| **Fig 4.3** | bar | Metric Type | Conversion Rate (%) | A/B/C 三套口径 | `gift_allocation/img/conversion_funnel_multi.png` | 转化漏斗（多口径） |
| **Fig 4.4** | scatter | total_watch_time (log) | total_gift_amount (log) | 四象限占比标注 | `gift_allocation/img/user_quadrant.png` | 用户二维象限 |
| **Fig 4.5** | line | watch_time_bin | gift_rate | 置信区间 | `gift_allocation/img/watch_time_vs_gift_rate.png` | watch_time 分位数分箱的 gift_rate |
| **Fig 4.8** | bar | gift_price | Count + Cumulative Share | Top 20 价位点 | `gift_allocation/img/gift_price_tiers.png` | 付费价位阶梯 |
| **Fig 5.1** | CCDF | total_watch_time (log) | P(watch_time > x) (log) | - | `gift_allocation/img/streamer_watch_time_ccdf.png` | 主播观看时长分布 |
| **Fig 5.2** | histogram | revenue_per_watch_hour (log) | Count | - | `gift_allocation/img/streamer_conversion_efficiency.png` | 主播转化效率分布 |
| **Fig 5.3** | scatter | total_watch_time (log) | total_revenue (log) | 四象限占比标注 | `gift_allocation/img/streamer_quadrant_revenue.png` | 主播二维象限（观看时长 vs 收入） |
| **Fig 5.4** | scatter | unique_viewers (log) | conversion_rate (log) | 四象限占比标注 | `gift_allocation/img/streamer_quadrant_conversion.png` | 主播二维象限（观看用户数 vs 转化率） |
| **Fig 6.3** | histogram | loyalty | Count | watch 与 gift 两套 | `gift_allocation/img/user_loyalty_dist.png` | 用户专一度分布 |
| **Fig 7.1** | heatmap | Hour × Day of Week | watch_time / gift_amount / conversion_rate | 三张并排或关键两张 | `gift_allocation/img/hour_dow_heatmap.png` | 7×24 热力图 |
| **Fig 8.1** | scatter | gift_interval | gift_amount | 可疑区域标注 | `gift_allocation/img/anomaly_rules.png` | 异常规则可视化 |

### P1 优先级图表（可选，但建议完成）

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| **Fig 3.4** | histogram | gift_interval (log-x) | Count | `gift_allocation/img/gift_interval_dist.png` |
| **Fig 4.2** | histogram | sessions_per_user | Count | `gift_allocation/img/user_sessions_dist.png` |
| **Fig 4.6** | bar | Streamer | Watch Count | Top 20 | `gift_allocation/img/high_watch_low_pay_streamers.png` |
| **Fig 4.7** | line | Days | Retention Rate (%) | 付费 vs 非付费 | `gift_allocation/img/retention_by_payment.png` |
| **Fig 4.9** | histogram | gift_interval | Count | `gift_allocation/img/gift_interval_dist.png` |
| **Fig 4.10** | histogram | days_between_gifts | Count | `gift_allocation/img/repurchase_cycle.png` |
| **Fig 5.5** | boxplot | Exposure Tier | revenue_per_watch_hour | `gift_allocation/img/streamer_coldstart_tiers.png` |
| **Fig 6.1** | histogram | Degree (log-log) | Count | watch vs gift | `gift_allocation/img/user_degree_dist.png` |
| **Fig 6.2** | histogram | Degree (log-log) | Count | watch vs gift | `gift_allocation/img/streamer_degree_dist.png` |
| **Fig 6.4** | combo | Streamer Rank | Cumulative Share + Top1 User % | Lorenz + 散点 | `gift_allocation/img/streamer_revenue_concentration.png` |
| **Fig 6.5** | histogram | entropy | Count | - | `gift_allocation/img/user_diversity_entropy.png` |
| **Fig 7.2** | stacked area | Date | Contribution | Top N 主播 | `gift_allocation/img/peak_contribution.png` |
| **Fig 8.2** | histogram | Edge Weight | Count | watch vs gift | `gift_allocation/img/edge_weight_comparison.png` |

---

## 4. 📁 参考代码

> ⚠️ **不要在这里写代码！只写路径，让 Agent 自己读取**

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/eda_kuailive.py` | `load_data()`, `compute_gini()`, 基础绘图函数 | 扩展 Session 构建、多维度分析函数 |
| `scripts/eda_kuailive.py` | `analyze_gift_basics()`, `analyze_amount_distribution()`, `analyze_user_dimension()`, `analyze_streamer_dimension()` | 复用基础统计逻辑，扩展新维度 |
| `scripts/eda_kuailive.py` | `plot_fig*()` 系列函数 | 参考绘图风格，新增 Session/User/Streamer/Interaction/Time/Anomaly 相关图表 |

**建议使用的 Python 库**：
- `pandas` - 数据处理、Session 构建、聚合
- `numpy` - 数值计算、分位数、统计
- `matplotlib`, `seaborn` - 可视化（注意 figsize 规则）
- `scipy.stats` - 统计分布拟合、Shannon entropy 计算
- `networkx` (可选) - 二部图构建与分析

**关键函数需要实现**：
- `build_sessions(click_df, gift_df)` - Session 构建（按实验报告 §10.2 Schema）
- `analyze_data_quality(click_df, gift_df)` - 数据质量检查
- `analyze_session_level(click_df, gift_df, sessions_df)` - Session 级分析
- `analyze_user_behavior(sessions_df, gift_df, user_df)` - 用户行为分析
- `analyze_supply_side(sessions_df, gift_df, streamer_df)` - 供给侧分析
- `analyze_interaction_structure(sessions_df, gift_df)` - 交互结构分析
- `analyze_temporal(click_df, gift_df)` - 时序分析
- `detect_anomalies(sessions_df, gift_df)` - 异常检测
- `plot_*()` 系列函数 - 28 个图表生成函数

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `gift_allocation/exp/exp_kuailive_eda_comprehensive_20260109.md`
- **模板**: `_backend/template/exp.md`
- **必须包含**:
  - ⚡ 核心结论速览（一句话 + 关键数字 + 5-8 条洞见）
  - 📊 实验图表（所有图 + 观察，每张图"读图结论一句话"）
  - 📝 结论（假设验证 + 设计启示 + 可行动建议）
  - **所有"待执行"部分必须填充完成**

### 5.2 图表文件
- **路径**: `gift_allocation/img/`
- **命名**: 如上表所述（如 `time_coverage.png`, `first_gift_time_ratio.png` 等）
- **要求**: 至少 12 张 P0 优先级图表，每张图能"一眼得到结论"

### 5.3 中间表
- **Session 表**: `gift_allocation/results/sessions_*.csv` 或 `gift_allocation/results/sessions_*.parquet`
- **Schema**: 见实验报告 §10.2

### 5.4 数值结果
- **格式**: JSON
- **路径**: `gift_allocation/results/eda_comprehensive_stats_20260109.json`
- **内容**:
```json
{
  "experiment_id": "EXP-20260109-gift-allocation-02",
  "date": "2026-01-09",
  "data_quality": {
    "completeness": 0-5,
    "consistency": 0-5,
    "validity": 0-5,
    "timeliness": 0-5,
    "overall_score": 0-5
  },
  "session_stats": {
    "total_sessions": N,
    "avg_session_duration": N,
    "session_duration_p50_p90_p99": [N, N, N],
    "session_conversion_rate": N,
    "immediate_gift_rate": N
  },
  "user_behavior": {
    "high_watch_low_pay_ratio": N,
    "gift_price_tiers": {...},
    "watch_time_vs_gift_rate": {...}
  },
  "supply_side": {
    "high_attract_low_revenue_ratio": N,
    "coldstart_revenue_per_watch_hour": N
  },
  "interaction": {
    "user_loyalty_p50_p90": [N, N],
    "streamer_revenue_concentration": N
  },
  "temporal": {
    "peak_hour": N,
    "peak_dow": N
  },
  "anomaly": {
    "suspicious_users_count": N,
    "suspicious_streamers_count": N
  }
}
```

---

## 6. 📤 报告抄送

完成后需同步更新：

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_allocation_roadmap.md` | MVP-0.1-Enhanced 状态 ✅ + 结论快照 | §2.1, §4.3 |
| `gift_allocation_hub.md` | 新洞见（如有重要发现）→ §4 洞见汇合<br>新设计原则（如有）→ §6 设计原则 | §4, §6 |

---

## 7. ⚠️ 注意事项

- [ ] 代码中添加 seed 固定随机性（seed=42）
- [ ] 图表文字全英文
- [ ] 处理缺失值和异常值（记录处理方法）
- [ ] Session 构建需处理边界情况（时间间隔、合并规则）
- [ ] 大数据集用采样预览（如需要）
- [ ] 保存完整日志到 `logs/eda_comprehensive_20260109.log`
- [ ] 长时间任务使用 nohup 后台运行
- [ ] **每张图必须能"一眼得到结论"**：标题是结论句，图上标注关键数字
- [ ] **避免截断导致误解**：重尾分布优先用 log-x、CCDF、rank-frequency
- [ ] **能合并就合并**：同一主题最多 1-2 张图

---

## 8. 📋 关键分析问题清单

完成 EDA 后需回答以下问题（对应实验报告中的假设）：

1. **H2.1: 打赏是"即时冲动"还是"随观看累积"？** → Fig 3.1, 3.2
2. **H2.2: 高观看低付费人群是否存在？规模多大？** → Fig 4.4, 4.6
3. **H2.3: 主播供给是否存在"高吸引低变现"类型？** → Fig 5.3, 5.4
4. **H2.4: 付费是否强绑定于特定主播？** → Fig 6.3
5. **H2.5: 数据是否存在异常/作弊/污染？** → Fig 8.1, 数据质量评分表
6. **watch_time 与 gift_prob 的关系是单调还是阈值型？** → Fig 4.5
7. **是否存在典型价位阶梯？** → Fig 4.8
8. **黄金时段是什么？（多指标综合）** → Fig 7.1
9. **冷启动主播表现如何？** → Fig 5.5
10. **用户观看多样性熵与付费金额是否相关？** → 相关性分析

---

## 9. 🔄 执行顺序建议

1. **Phase 1: 数据质量检查** (T2.1, T2.2)
   - 时间覆盖检查
   - 异常值检查
   - Gift-Click 一致性检查
   - 生成数据质量评分表

2. **Phase 2: Session 构建** (核心基础)
   - 实现 `build_sessions()` 函数
   - 验证 Session 表 Schema
   - 计算 Session 级基础统计

3. **Phase 3: Session & Funnel 分析** (T3.1-T3.5)
   - 首礼时间分析
   - Session 时长分布
   - 转化漏斗

4. **Phase 4: User Behavior 分析** (T4.1-T4.5, T4.8)
   - 用户观看强度
   - 用户二维象限
   - watch_time vs gift_rate
   - 付费价位阶梯

5. **Phase 5: Supply Side 分析** (T5.1-T5.4)
   - 主播曝光与观看
   - 主播二维象限
   - 转化效率

6. **Phase 6: Interaction Structure 分析** (T6.3)
   - 用户专一度分布

7. **Phase 7: Time & Seasonality 分析** (T7.1)
   - 7×24 热力图

8. **Phase 8: Anomaly Detection** (T8.1)
   - 异常规则可视化

9. **Phase 9: 填充报告**
   - 所有"待执行"部分
   - Executive Summary
   - 可行动建议

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先读取 `scripts/eda_kuailive.py` 理解现有代码结构
3. ✅ 理解实验报告 `exp_kuailive_eda_comprehensive_20260109.md` 的完整要求
4. ✅ 按顺序执行：数据质量检查 → Session 构建 → 多维度分析 → 图表生成
5. ✅ 复用已有函数，扩展新功能
6. ✅ 每张图必须能"一眼得到结论"（标题是结论句，标注关键数字）
7. ✅ 所有图表保存到 `gift_allocation/img/`
8. ✅ 数值结果保存为 JSON 到 `gift_allocation/results/`
9. ✅ 填充实验报告所有"待执行"部分
10. ✅ 完成后同步更新 roadmap.md 和 hub.md（如有重要发现）
-->
