# 🧠 gift_EVpred Hub
> **ID:** EXP-20260118-gift_EVpred-hub | **Status:** 🌷 收敛（识别大哥阶段）
> **Date:** 2026-01-18 | **Update:** 2026-01-18

---

| # | 💡 共识 | 证据 | 决策 |
|---|--------|------|------|
| K1 | **排序/分配 > 逐条预测**：不需要预测准每一条，把高价值放对位置即可 | RevCap vs MAE | 评估用 RevCap@K |
| K2 | **Raw Y 优于 Log(1+Y)**：log 变换压缩 whale 信号 | RevCap +39.6% | 预测目标用 raw Y |
| K3 | **RevCap ≠ Spearman**：整体排序不重要，头部精准更重要 | 负相关 | 用 RevCap 评估 |
| K4 | **历史打赏是最强信号**：pair-level 历史 > user/streamer 全局 | 系数 0.194 | 重点挖掘 pair 特征 |
| K5 | **时间窗纪律**：任何特征必须是 t_click 之前可得 | 200/200 通过 | Day-Frozen / Rolling |
| K6 | **watch_live_time 是 post-treatment**：包含打赏后停留，click 时刻不可得 | 因果分析 | 删除当前 session；可用历史先验 |
| K7 | **回归 > 分类（for RevCap）**：分类丢失金额信息，回归保留"大哥有多大" | AUC↑但RevCap↓ | RevCap 任务用回归，不用分类 |
| K8 | **Linear > Tree（高稀疏回归）**：98.5% 为 0 时，树模型预测众数 | LGB -14.5% | 稀疏回归用 Linear |

**🦾 现阶段信念**
- **Direct + raw Y (Ridge) 是当前最优方案** → RevCap@1%=52.60%，Linear 模型已到瓶颈
- **LightGBM 回归失败，分类成功但无用** → 分类丢失金额信息，AUC 高不代表 RevCap 高
- **目前专注"识别大哥"**，暂不考虑分配优化 → 先做好 EV 预测这个基础模块
- **Frozen 特征已够用**，Rolling 是后续迭代方向 → 当前不增加复杂度
- **✅ data_utils.py 已验证通过** → Expert Review 的 4 个问题经验证：3 个不存在，1 个已修复

**👣 下一步最有价值**
- 🔴 **P0-Feature**：历史观看先验特征（`user_watch_mean_7d`, `pair_watch_mean`） → 替代 watch_live_time
- 🔴 **P0-Metrics**：三层指标体系标准化（见 §10）→ 所有实验统一评估口径
- 🟡 **P1-Tail**：样本加权回归 `w_i = (1+y_i)^α`（α∈[0.3,1]）→ 提升头部捕获
- 🟡 **P1-Whale**：Whale 专属特征（历史大额打赏次数/金额） → 更好区分超级大哥
- 🟢 **P2-Model**：尝试 MLP → 神经网络可能比树模型更适合稀疏数据
- 🟢 **P3-Alloc**：分配 MVP（纯贪心 vs 凹收益贪心）→ `Δ(u→s) ≈ g'(V_s) · v_{u,s}`

> **权威数字**：Best=52.60% RevCap@1%（Linear 上界）；Baseline=37.73%；Δ=+39.4%；条件=Direct Ridge + raw Y, Day-Frozen, alpha 不敏感

| 模型/方法 | RevCap@1% | 配置 | 备注 |
|-----------|-----------|------|------|
| **Direct + raw Y (Ridge)** | **52.60%** | Ridge alpha=1.0, raw Y | ✅ Linear 上界 (B2) |
| LightGBM + raw Y | 49.49% | 强正则化 | ❌ -3.1%，Tree 失败 |
| Two-Stage + raw Y | 45.66% | LR + Ridge, raw Y | B2 变体 |
| Whale-only (P90) | 43.60% | LR + LR + Ridge | Three-Stage |
| Direct + log Y | 37.73% | Ridge, log(1+Y) | B2 baseline |
| LGB P(whale) 分类 | 36.71% | AUC=0.8533 | ❌ 分类丢失金额信息 |
| Oracle | 99.54% | 真实 top-1% | 理论上界 |

**Baseline 层级**（用于对照"是否有用"）：
| 层级 | 方法 | RevCap@1% | 说明 |
|------|------|-----------|------|
| B0 | Random | ~1% | 随机排序 |
| B1 | Popularity | 待测 | 按主播热度排序 |
| **B2** | **Direct + raw Y** | **52.68%** | **当前 ML baseline** |

**MVP 验收标准**：
- 离线：Top 1% 捕获 >50% 收入（当前 52.68% ✅）
- 在线：Revenue per DAU 正向，留存/生态指标不恶化

---

## 1) 🎯 任务定义

### 1.1 问题设定

```
【Click-level EV 预测】

输入：(user_id, streamer_id, live_id, t_click) + 进入时刻可用特征
输出：未来窗口 [t_click, t_click+W] 内礼物金额总和的期望 (EV)
标签：gift_price_label = sum(gift_price) within window W
```

| 项 | 规格 |
|---|---|
| 预测粒度 | Click-level（用户进入直播间时刻） |
| 标签窗口 | W = 1 hour |
| 预测目标 | **raw Y**（非 log(1+Y)） |
| 辅助标签 | is_gift = (gift_price_label > 0) |

### 1.2 业务目标

```
当前阶段目标：识别"大哥"（高价值打赏用户）
下游应用：分配优化（把大哥分配给合适的主播）
```

| 阶段 | 目标 | 指标 |
|------|------|------|
| **Phase 1（当前）** | 识别大哥 + 预测打赏金额 | Revenue Capture @K |
| Phase 2（后续） | User-Streamer 偏好匹配 | 待定 |
| Phase 3（远期） | 分配优化 + 长期收益 | Total Revenue + LTV |

### 1.3 特征构造策略

| 策略 | 定义 | 状态 | 适用场景 |
|------|------|------|---------|
| **A. Frozen** | 仅用 Train 窗口内历史计算统计量；Val/Test 只查表 | ✅ 当前使用 | 快速迭代、无泄漏保证 |
| B. Rolling | 对每个 click 严格用 `gift_ts < click_ts` 构造特征 | ⏳ 后续迭代 | 更精确、在线推理对齐 |

**当前选择 Frozen 的原因**：
1. 实现简单，无泄漏风险
2. 已验证 200/200 样本通过
3. 当前阶段专注模型和目标变量选择，不增加特征复杂度

---

## 2) 🌲 核心假设树

```
🌲 核心: 如何建模 EV（期望打赏）以识别高价值用户？
│
├── Q0: 基础设施正确性 ✅
│   ├── Q0.1: 数据泄漏已消除？ → ✅ Day-Frozen，200/200 验证通过
│   ├── Q0.2: 任务定义正确？ → ✅ Click-level，gift_rate ~1.5%
│   └── Q0.3: 评估指标对齐业务？ → ✅ Revenue Capture @K
│
├── Q1: 如何建模 EV？ ✅ 已收敛
│   ├── Q1.1: Direct vs Two-Stage？ → ✅ Direct 更优（RevCap 52.68% vs 45.66%）
│   ├── Q1.2: Raw Y vs Log(1+Y)？ → ✅ Raw Y 显著更优（+39.6%）
│   ├── Q1.3: Three-Stage Whale-only？ → ❌ 不如 Direct raw Y
│   └── Q1.4: 最强特征？ → ✅ pair_gift_cnt_hist
│
└── Q2: 下一步优化方向？ ⏳
    ├── Q2.1: LightGBM 替代 Linear？ → ❌ 失败（-3.1%~-14.5%），Tree 不适合稀疏回归
    ├── Q2.2: Whale-specific 特征？ → ⏳ 待验证
    └── Q2.3: Rolling 特征？ → ⏳ 后续迭代

Legend: ✅ 已验证 | ❌ 已否定 | ⏳ 待验证
```

---

## 3) 口径冻结（唯一权威）

| 项目 | 规格 |
|---|---|
| Dataset | KuaiLive (`data/KuaiLive/`) |
| Train / Val / Test | 1,629,415 / 1,717,199 / 1,409,533 (7-7-7 by days) |
| 时间范围 | 2025-05-04 ~ 2025-05-24 |
| 特征 | 31 个 Day-Frozen 特征 |
| Metric | **Revenue Capture @1%**（主指标） |
| 标签窗口 | 1 hour |
| Seed | 42 |

> 规则：任何口径变更必须写入 §8 变更日志。

---

## 4) 当前答案 & 战略推荐

### 4.1 战略推荐

- **推荐路线：Direct + raw Y + LightGBM**（理由：Linear 已到瓶颈，非线性可能进一步提升）
- 需要 Roadmap 关闭的 Gate：Gate-2（LightGBM 验证）

| Route | 一句话定位 | 当前倾向 | 关键理由 |
|---|---|---|---|
| Route A: Direct + raw Y (Linear) | 简单有效的 baseline | 🟢 已验证 | RevCap=52.68% |
| Route B: Two-Stage | 分阶段建模 | 🟡 次选 | RevCap=45.66%，不如 Direct |
| **Route C: Direct + raw Y + LightGBM** | 非线性提升 | 🔴 待验证 | 预期进一步提升 |

### 4.2 分支答案表

| 分支 | 当前答案 | 置信度 | 决策含义 | 证据 |
|---|---|---|---|---|
| Q1.1 Direct vs Two-Stage | Direct 更优 | 🟢 | 不用 Two-Stage | exp_raw_vs_log |
| Q1.2 Raw Y vs Log Y | Raw Y 显著更优 | 🟢 | 预测目标用 raw Y | exp_raw_vs_log |
| Q1.3 Three-Stage | 不如 Direct | 🟢 | 不用 Three-Stage | exp_three_stage |

---

## 5) 洞见汇合

> 只收录"会改变决策"的洞见

### 5.1 方法论洞见（EV 建模通用）

| # | 洞见 | 观察 | 解释 | 决策影响 |
|---|---|---|---|---|
| **M1** | **排序/分配 > 逐条预测** | 不需要"预测准每一条" | 把更可能产生价值的放到更好位置即可 | 评估用 RevCap@K，不用 MAE/RMSE |
| **M2** | **RevCap@K 是关键指标** | Top K% 曝光捕获的收入占比 | 打赏是重尾分布，价值集中在头部 | MVP 标准：Top 1% 捕获 >50% 收入 |
| **M3** | **时间窗纪律** | 任何特征必须是 t_click 之前可得 | ❌ 会话结束后才知道的信息（最终停留时长等） | 严格按时间切分 + 泄漏检查 |
| **M4** | **历史打赏是最强信号** | pair-level 历史 > user/streamer 全局 | 同一用户对不同主播打赏概率差很多 | 重点挖掘 pair 特征 |
| **M5** | **无常 ≠ 不能建模** | 单次打赏难预测，但条件概率稳定 | 规模上有系统性差异：有人"几乎不打赏"，有人"稳定高消费" | 模型学到系统性差异即有价值 |
| **M6** | **业界惯例：收入用 raw，参与用 log** | 电商 GMV 用 raw bid，视频推荐用 log watch_time | 业务目标决定目标变换：收入最大化 → raw；用户覆盖 → log | 打赏场景 = 收入场景 → raw Y |
| **M7** | **watch_live_time 是 post-treatment** | 包含打赏后的停留时间 | 本质是结果/中介变量，click 时刻不可得 | 当前 session 的 watch_time **必须删除** |
| **M8** | **可用替代：历史观看先验** | user/pair/streamer 历史平均观看时长 | 这些是 pre-treatment，满足 `< click_ts` 即可用 | 用历史先验代替当前 session |
| **M9** | **Partial dwell 是合理二阶段信号** | 进入后 5s/10s 的停留/互动 | 互动→打赏 的因果链路更合理 | 若允许延迟打分，可作为重排序特征 |
| **M10** | **Train 内部也需要时间一致性** | 用"整窗聚合 + last_ts 挡"会制造分布偏移 | 对 train 早期样本：pair 统计被清零，与 val/test 机制不一致 | Train 也要 rolling 或 day-snapshot |
| **M11** | **Two-Stage 不是完全没用** | Two-Stage 在 NDCG@100 可能更好 | 高稀疏下 Stage2 样本少误差大，但对"金主内部排序"可能有效 | RevCap 用 Direct，精细排序可试 Two-Stage |

### 5.2 本项目验证的洞见

| # | 洞见 | 观察 | 解释 | 决策影响 | 证据 |
|---|---|---|---|---|---|
| **E1** | **Raw Y >> Log Y** | RevCap +39.6% | Log 压缩 whale 信号（10元 vs 1000元差距变小） | 用 raw Y | exp_raw_vs_log |
| **E2** | **RevCap ≠ Spearman** | 负相关 | 整体排序 vs 头部精准是不同目标 | 用 RevCap 评估 | exp_raw_vs_log |
| **E3** | **Direct > Two-Stage** | 52.68% vs 45.66% | Two-Stage 引入额外误差 | 用 Direct | exp_raw_vs_log |
| **E4** | **Pair 历史最强** | 系数 0.194 | 历史打赏行为是最强预测信号 | 重点挖掘 pair 特征 | exp_baseline |
| **E5** | **Day-Frozen 无泄漏** | 200/200 通过 | merge_asof + allow_exact_matches=False | 可信实验基础 | exp_baseline |
| **E6** | **阈值越高 RevCap 越高** | P90 > P80 > ... > P50 | 专注"超级大哥"比覆盖全体 gifters 更有效 | raw Y 隐式实现 | exp_three_stage |
| **E7** | **Ridge alpha 不敏感** | alpha 0.001~1000 RevCap 不变 | 特征无严重共线性，Linear 已达瓶颈 | 非线性未必有效 | alpha_sweep |
| **E8** | **LightGBM 回归失败** | RevCap -3.1%~-14.5% | 98.5% 为 0，树模型预测众数 | 稀疏回归用 Linear | exp_lightgbm |
| **E9** | **LightGBM 分类成功但无用** | P(whale) AUC=0.8533 (+38%) | 分类丢失金额信息：100元 vs 10000元 whale 分数相同 | RevCap 用回归不用分类 | exp_lightgbm |
| **E10** | **AUC ≠ RevCap** | AUC 高但 RevCap 低 | AUC 衡量"区分能力"，RevCap 衡量"价值捕获" | 用 RevCap 评估 | exp_lightgbm |

**Linear 模型瓶颈确认**：Ridge alpha sweep 显示无调参空间，RevCap@1%=52.60% 是 Linear + Day-Frozen 的上界。LightGBM 无法突破此瓶颈（Tree 不适合稀疏回归）。

---

## 6) 决策空白（Decision Gaps）

| DG | 我们缺的答案 | 为什么重要 | 什么结果能关闭它 | 决策规则 |
|---|---|---|---|---|
| ~~DG1~~ | ~~LightGBM 是否优于 Linear？~~ | ~~Linear 已确认到瓶颈~~ | ❌ **已关闭** | LightGBM 失败（-3.1%），转向特征工程 |
| **DG2** | **Whale-specific 特征有用吗？** | 可能进一步区分大哥 | RevCap 提升 | If >2% 提升 → 加入；Else → 不加 |
| **DG3** | **Frozen vs Rolling 哪个对齐线上？** | 决定评估协议 | 明确线上特征更新频率 | 批处理→Frozen；实时计数器→Rolling |
| DG4 | Partial dwell (5s/10s) 值得做吗？ | 若允许延迟打分，可作为重排序特征 | 工程可行性 + RevCap 增益 | 需要产品侧确认是否允许延迟打分 |
| **DG5** | **MLP 是否优于 Linear？** | Tree 失败，NN 可能更适合稀疏数据 | RevCap 对比 | If >55% → 用 MLP；Else → 专注特征工程 |

### 6.1 Protocol Trade-off（Frozen vs Rolling）

| Protocol | 定义 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| **A. Frozen-Serving** | Train 结束时 freeze 统计；Val/Test 只查这份表 | 简单、低成本、可上线 | 同天内信息完全冻结，信号损失 | 离线批处理特征 |
| **B. Online-Rolling** | Val/Test 允许用"到该 click 为止的全部历史" | 更精确、理论上界 | 需要实时计数器，工程复杂 | 有流式特征基础设施 |

> **建议**：两套协议都跑并汇报，既评估模型效果，也评估"实时特征的工程投入是否值得"

---

## 7) 设计原则

### 7.1 已确认原则

| # | 原则 | 建议 | 适用范围 | 证据 |
|---|---|---|---|---|
| P1 | **用 raw Y，不用 log(1+Y)** | 预测目标 = raw gift_price | 收入最大化场景 | exp_raw_vs_log |
| P2 | **用 RevCap，不用 Spearman** | RevCap@1% 作为主指标 | 高价值用户识别 | exp_raw_vs_log |
| P3 | **用 Day-Frozen 特征** | `day < current_day` 的历史 | 无泄漏保证 | exp_baseline |
| P4 | **禁止当前 session 的 watch_live_time** | Post-treatment 泄漏特征 | 所有实验 | data_utils.py |
| P5 | **使用 data_utils.py** | 统一数据处理入口 | 所有实验 | - |
| **P6** | **Train/Val/Test 机制必须一致** | 若 Frozen 则全 Frozen；若 Rolling 则全 Rolling | 避免分布偏移 | expert review |
| **P7** | **Category 编码：Train 拟合，全局复用** | 用 train 的 categories 映射 val/test；未知归为 unknown | 可复现性 | expert review |

### 7.2 待验证原则

| # | 原则 | 初步建议 | 需要验证 |
|---|---|---|---|
| ~~P8~~ | ~~LightGBM 优于 Linear~~ | ~~非线性捕获能力~~ | ❌ 已否定 |
| P9 | Rolling 优于 Frozen（若线上有实时特征） | 更精确的历史 | DG3 决策后 |
| P10 | Multi-task (EV + is_gift) | 用密集信号扶起稀疏打赏 | 后续迭代 |
| P11 | MLP 可能优于 Linear | NN 可能比 Tree 更适合稀疏数据 | DG5 验证 |

### 7.3 已关闭方向

| 方向 | 否定证据 | 关闭原因 | 教训 |
|---|---|---|---|
| ~~Two-Stage for RevCap~~ | RevCap 45.66% < Direct 52.68% | Stage2 样本少误差大，乘 p(x) 放大噪声 | RevCap 用 Direct；精细排序可保留 Two-Stage |
| ~~Three-Stage Whale-only~~ | RevCap 43.60% < Direct 52.68% | 不如 Direct raw Y | raw Y 已隐式学会找 whale |
| ~~Log(1+Y) 目标~~ | RevCap -39.6% | 压缩 whale 信号（10元 vs 1000元差距变小） | 收入场景用 raw Y |
| ~~当前 session watch_live_time~~ | Post-treatment | 包含打赏后停留时间，click 时刻不可得 | 用历史观看先验替代 |
| ~~LightGBM 回归~~ | RevCap -3.1%~-14.5% | 98.5% 为 0，树模型预测众数 | **稀疏回归用 Linear，不用 Tree** |
| ~~P(whale) 分类替代回归~~ | AUC=0.8533 但 RevCap=36.71% | 分类丢失金额信息（100元=10000元） | **RevCap 任务用回归，不用分类** |

### 7.4 ✅ 已验证问题（2026-01-18）

Expert Review 识别的 4 个潜在问题，经代码分析和验证：

| 问题 | 状态 | 说明 |
|---|---|---|
| Train 内部 pair 统计被清零 | ✅ 不存在 | Day-Frozen 使用 `merge_asof(allow_exact_matches=False)`，无 last_ts 清零逻辑 |
| User/Streamer 全局聚合泄漏 | ✅ 不存在 | 三级特征都用 cumsum + merge_asof，严格 `day < click_day` |
| pair_last_gift_gap 可能为负 | ✅ 不适用 | 当前实现无 last_gift_gap 特征 |
| Category 编码各 split 独立 | ✅ **已修复** | Train 拟合 categories，Val/Test 复用映射，OOV→'unknown' |

**验证结果**：200/200 samples 通过无泄漏验证

---

## 8) 指针

| 类型 | 路径 | 说明 |
|---|---|---|
| 🗺️ Roadmap | `gift_EVpred_roadmap.md` | Decision Gates + MVP 执行 |
| 📄 Data Utils | `data_utils.py` | 统一数据处理入口 |
| 📗 Baseline | `exp/exp_baseline_day_frozen_20260118.md` | Day-Frozen 基线 |
| 📗 Three-Stage | `exp/exp_three_stage_20260118.md` | Whale-only 实验 |
| 📗 Raw vs Log | `exp/exp_raw_vs_log_20260118.md` | 当前最优方案 |
| 📗 LightGBM | `exp/exp_lightgbm_raw_y_20260118.md` | ❌ 失败分析 + 分类 vs 回归洞见 |
| 📊 Results | `results/` | 数值结果 JSON |
| 🗃️ Archive | `exp/archive_leaky/` | 有泄漏的旧实验 |

---

## 9) 变更日志

| 日期 | 变更 | 影响 |
|---|---|---|
| 2026-01-18 | 创建 Day-Frozen 无泄漏框架 | 可信实验基础 |
| 2026-01-18 | Direct vs Two-Stage baseline | 初始 baseline |
| 2026-01-18 | Three-Stage Whale-only 实验 | RevCap=43.60%，不如预期 |
| 2026-01-18 | **Raw Y vs Log Y 实验** | **RevCap=52.68%（+39.6%），当前最优** |
| 2026-01-18 | Expert Review 整合 | 新增 M7-M11 方法论洞见；明确 Frozen vs Rolling 协议 trade-off |
| 2026-01-18 | **P0-Tech 验证通过** | 3 个问题不存在（Day-Frozen 设计正确），1 个已修复（Category 编码）；200/200 验证通过 |
| 2026-01-18 | **Alpha Sweep** | Ridge alpha 0.001~1000 无影响，确认 Linear 瓶颈 52.60% |
| 2026-01-18 | **LightGBM 实验** | ❌ 回归失败（-3.1%~-14.5%），Tree 不适合稀疏数据 |
| 2026-01-18 | **分类 vs 回归洞见** | 🔑 分类丢失金额信息，AUC≠RevCap；新增 K7, K8, E8-E10 |
| 2026-01-18 | **三层指标体系** | 新增 §10 识别层/估值层/分配层指标框架 |

---

## 10) 📐 三层指标体系（2026-01-18）

> 基于专家建议，建立"识别大哥 → 预测价值 → 分配"全链路统一评估口径

### 10.1 识别层（Find Whales / 找大哥）

**主指标（North Star）**：
- **RevCap@K**（K ∈ {0.1%, 0.5%, 1%, 2%, 5%, 10%}）：画 RevCap 曲线
- **Normalized RevCap@K** = RevCap@K / Oracle@K

**诊断指标**：
- Whale Recall@K（y≥P90 的召回）
- Precision@K（TopK 里 whale 比例）
- Avg Revenue per Selected@K
- **Stability**（按天 RevCap@1% 均值±标准差）

> ⚠️ **评估泄漏警告**：用 Val 挑 best，Test 只做一次最终报告

### 10.2 估值层（Predict EV / 预测大哥期望赏金）

**Tail Calibration（头部校准）**：
- 按预测分桶（top 0.1%/0.5%/1%/5%），计算 `Sum(pred)/Sum(actual)`
- 目标：top 桶不系统性偏高/偏低

**误差指标**：
- Over/Under-estimation Ratio（分配更怕"高估"）
- Tail MAE / Pinball Loss (Q90/Q95)

### 10.3 分配层（Allocation / 把大哥分给谁）

**优化目标**：`max ∑_s g_s(V_s)`，边际增益 `Δ(u→s) ≈ g'(V_s) · v_{u,s}`

**分配指标**：
- Total Revenue（线性）: `∑_s V_s`
- Concave Revenue（凹收益）: `∑_s g_s(V_s)`
- Streamer Gini / Coverage（生态）
- Constraint Violations（约束违反率）

### 10.4 在线护栏

| 类别 | 指标 |
|------|------|
| 收益 | Revenue/DAU, ARPPU |
| 留存 | D1/D7 留存率 |
| 生态 | Gini 变化, 中小主播覆盖 |
| 风控 | 退出率, 投诉率 |

> 详见：[exp_metrics_framework_20260118.md](./exp/exp_metrics_framework_20260118.md)

---

<details>
<summary><b>附录：术语表</b></summary>

| 术语 | 定义 |
|---|---|
| EV | Expected Value，期望打赏金额 |
| RevCap@K | Revenue Capture @K%，Top-K% 预测捕获的实际收入占比 |
| Whale | 大额打赏用户（如 ≥100元） |
| Day-Frozen | 历史特征只用 `day < current_day` 的数据 |
| Rolling | 历史特征用 `gift_ts < click_ts` 的数据 |

</details>
