你是资深推荐系统评估专家 + ML平台工程师。请在现有项目的 metrics.py（Gift EVpred Metrics Module）上做一次“决策相关指标”升级：新增校准、切片、生态健康三类指标，并把它们集成进 evaluate_model() 与 EvalResult 的输出（summary + JSON）。要求保持向后兼容，不破坏现有 API。

========================
0) 约束与风格
========================
- 只使用：numpy / pandas / dataclasses / typing / json（可选 scipy 已被现有代码 try-import，不新增硬依赖）。
- 新增函数必须有：清晰 docstring + type hints + 合理默认参数。
- 需要在大规模数据（百万级）上可运行：尽量向量化、避免 Python for 循环扫描全量样本（除非只对 bin 数/天数等小维度循环）。
- 对缺失列（例如 tier 列不存在）要“优雅降级”：跳过该 slice 并在返回结果中记录 reason，不抛异常。
- 产出需要包含：实现代码 +（可选但强烈推荐）pytest 单元测试 + usage 示例。
- 保持现有函数/字段不变，新增字段都设置 default_factory 或 Optional，避免破坏旧 JSON load。

========================
1) 目标：新增 3 个核心函数
========================

(1) compute_calibration()  # ECE + Reliability Curve
--------------------------------
新增函数：
    def compute_calibration(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        strategy: str = "uniform",  # "uniform" 或 "quantile"
        sample_weight: Optional[np.ndarray] = None,
        eps: float = 1e-12,
    ) -> Dict[str, Any]:

用途：
- 用于二分类概率校准（例如是否打赏 P(gift|x)）。
- 输出 ECE + reliability curve（每个 bin 的 avg_pred / avg_true / count / bin range）。

定义：
- y_true：0/1 或 bool（可接受原始金额 y_amount，内部可自动转换 y_true_bin = (y_true > 0)）
- y_prob：预测概率，范围[0,1]（若发现不在范围内，进行 clip 并记录 warning）
- reliability curve：按 bin 统计：
    - bin_lower, bin_upper
    - n（该 bin 样本数）
    - avg_pred（bin 内平均预测概率）
    - avg_true（bin 内真实正例率）
    - gap = avg_true - avg_pred
- ECE：
    ECE = Σ_bin (n_bin / n_total) * |avg_true - avg_pred|

bin 划分：
- strategy="uniform"：等宽 bins in [0,1]
- strategy="quantile"：按 y_prob 分位数切 bins（需要处理重复分位点导致空 bin）

返回结构示例：
{
  "ece": 0.018,
  "bins": [
     {"bin_lower":0.0,"bin_upper":0.1,"n":12345,"avg_pred":0.04,"avg_true":0.03,"gap":-0.01},
     ...
  ],
  "meta": {"n":..., "positive_rate":..., "strategy":"uniform", "n_bins":10, "warnings":[...]}
}

注意：
- 该函数是“概率校准”；现有 tail_calibration 是“金额/EV 的 sum ratio 校准”，两者都要保留。
- 若全为同一类（全0或全1），ece 返回 0 或 NaN 需明确（建议 NaN + warning）。

(2) compute_slice_metrics()  # cold-start / whale / tier 切片
--------------------------------
新增函数：
    def compute_slice_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        df: pd.DataFrame,
        whale_threshold: float,
        k_values: Optional[List[float]] = None,
        # 切片字段名允许配置
        user_col: str = "user_id",
        streamer_col: str = "streamer_id",
        pair_hist_col: str = "pair_gift_count",
        streamer_hist_col: str = "streamer_gift_count",
        user_value_col: str = "user_gift_sum",
        streamer_value_col: str = "streamer_gift_sum",
        user_tier_col: Optional[str] = None,      # 若存在可用；否则用 user_value_col 生成 tier
        streamer_tier_col: Optional[str] = None,  # 同理
        min_slice_n: int = 500,
    ) -> Dict[str, Any]:

目标：
- 支持“决策相关指标”的分群诊断：冷启动、whale、分层用户/主播。
- 每个 slice 输出一套核心指标（至少 RevCap@K 曲线 + @1% 的 whale_recall/precision/avg_rev/gift_rate + tail_calibration）。
- 对于 slice 中 y_true.sum()==0 的情况，要返回可解释的空结果（例如 metrics=None + reason）。

必须包含的 slices（按可用字段自动启用）：

A) Cold-start
- cold_start_pair: pair_hist_col == 0
- cold_start_streamer: streamer_hist_col == 0（如果列不存在，可用 streamer_value_col==0 退化）
输出重点：
- cold_start_pair 的 RevCap@K（至少 K=1%）
- cold_start_streamer 的 RevCap@K（至少 K=1%）

B) Whale / High-value
- whale_true: y_true >= whale_threshold （用于诊断“是否抓住大额礼物”）
- non_whale_true: y_true < whale_threshold
输出重点：
- whale_true 的 recall@K 等（可复用现有 whale_recall_at_k 的逻辑，但注意 slice 后的定义：slice 内的 recall 更像 “该 slice 被 topK 命中的比例”）

C) Tier（用户分层/主播分层）
分层规则（若 user_tier_col / streamer_tier_col 不存在）：
- 用 user_value_col（用户历史总打赏）按分位点切：
    - user_top_1pct, user_top_10pct, user_tail（剩余）
- 用 streamer_value_col 类似切 streamer tiers
如果这些列都不存在，则跳过 tier slices，并记录 reason。

每个 slice 的返回建议结构：
{
  "cold_start_pair": {
      "n": ..., "total_revenue": ...,
      "revcap_curve": {...},
      "metrics_by_k": {...},
      "calibration": {...},
      "notes": "...",
  },
  "user_top_1pct": {...},
  "skipped": {
      "user_tier": "missing columns: user_gift_sum/user_tier",
      ...
  }
}

实现建议：
- 用一个内部 helper：_eval_subset(mask) 复用 compute_revcap_curve / compute_all_metrics_at_k / tail_calibration。
- 避免在 slice 内重复排序太多次：k_values 默认使用 metrics.py 里的 DEFAULT_K_VALUES。

(3) compute_ecosystem_metrics()  # Gini / Coverage / Overload
--------------------------------
新增函数：
    def compute_ecosystem_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        df: pd.DataFrame,
        k_select: float = 0.01,               # 以 Top-K% 作为“被分配/曝光集合”的离线近似
        user_col: str = "user_id",
        streamer_col: str = "streamer_id",
        timestamp_col: str = "timestamp",
        # tail/cold-start 定义可配置
        streamer_hist_col: str = "streamer_gift_count",
        streamer_value_col: str = "streamer_gift_sum",
        tail_streamer_quantile: float = 0.8,  # bottom 80% 视为长尾（可调）
        # overload 定义可配置
        overload_window_minutes: int = 10,
        overload_cap_per_window: int = 3,
        high_value_user_col: Optional[str] = None,   # 若提供，用于识别“高价值用户”
        high_value_user_quantile: float = 0.99,      # 否则按 user_value_col 分位数生成
        user_value_col: str = "user_gift_sum",
        eps: float = 1e-12,
    ) -> Dict[str, Any]:

目标：
- 给“分配决策”提供生态护栏：集中度、覆盖率、过载风险。
- 使用离线近似：把预测 Top k_select 的样本视为“系统会分配/曝光”的集合，然后在此集合上计算生态指标。

必须输出的生态指标：
1) Streamer Gini（集中度）
- 在 selected set 内，按 streamer 聚合：
    - exposure_count（被选中的样本数）
    - captured_revenue = Σ y_true（被选中样本的真实收入）
- 对 captured_revenue 计算 Gini（若全 0 则为 0 或 NaN，需要明确）
- 同时建议输出 top10_share（captured_revenue 前10%主播占比）

2) Coverage（覆盖率）
- streamer_coverage: selected set 覆盖到的 unique streamer / 全体 unique streamer
- tail_coverage: 长尾 streamer（按 streamer_value_col 或 df内 y_true 聚合的总收入分布定义）中，被 selected set 覆盖到的比例
- cold_start_streamer_coverage: streamer_hist_col==0 的主播中，被覆盖到的比例（若缺列则跳过）

3) Overload Rate（过载）
定义离线近似：
- 将 selected set 按时间分桶（window = overload_window_minutes），统计每个 streamer 在每个时间桶里接收到的“高价值用户”数量：
    - 高价值用户识别：
        - 若 high_value_user_col 存在：直接用该列为 bool 或 tier
        - 否则用 user_value_col 在全 df 上取 high_value_user_quantile 作为阈值
- 若某 streamer 在某时间桶内高价值用户数 > overload_cap_per_window => overload 事件
- overload_rate 输出至少两种口径（建议都给）：
    - overload_bucket_rate = overload_buckets / total_buckets（按 streamer×time_bucket）
    - overloaded_streamer_rate = unique overloaded streamers / unique streamers (in selected set)

返回结构示例：
{
  "selection": {"k_select":0.01,"n_selected":..., "n_total":...},
  "gini": {"streamer_revenue_gini":0.93, "top10_share":0.81},
  "coverage": {"streamer_coverage":..., "tail_coverage":..., "cold_start_streamer_coverage":...},
  "overload": {"window_minutes":10,"cap":3,"overload_bucket_rate":..., "overloaded_streamer_rate":...},
  "meta": {"warnings":[...], "used_columns": {...}}
}

需要新增一个通用 gini helper：
- def gini_coefficient(x: np.ndarray) -> float
要求对 0、负数、空数组做健壮处理（负数可 clip 为 0 并 warning，或直接 raise；本项目建议 clip+warning）。

========================
2) 集成到 EvalResult + evaluate_model
========================
在 EvalResult dataclass 增加以下字段（全部设 default_factory/Optional，避免破坏旧JSON）：
- prob_calibration: Optional[Dict[str, Any]] = None
- slice_metrics: Dict[str, Any] = field(default_factory=dict)
- ecosystem: Dict[str, Any] = field(default_factory=dict)

在 evaluate_model() 增加可选参数（保持默认不改变旧行为）：
- y_prob: Optional[np.ndarray] = None        # 若提供则计算 compute_calibration
- compute_slices: bool = True                # 仅在 test_df 存在时生效
- compute_ecosystem: bool = True             # 仅在 test_df 存在时生效
- slice_config: Optional[Dict[str, Any]] = None
- ecosystem_config: Optional[Dict[str, Any]] = None
- calibration_config: Optional[Dict[str, Any]] = None

行为：
- 若 y_prob is not None：prob_calibration = compute_calibration(y_true_bin, y_prob, **calibration_config)
- 若 test_df is not None 且 compute_slices：slice_metrics = compute_slice_metrics(..., **slice_config)
- 若 test_df is not None 且 compute_ecosystem：ecosystem = compute_ecosystem_metrics(..., **ecosystem_config)

summary() 输出增加两个 section（简洁但可读）：
- --- Probability Calibration ---
  ECE: x.xxx  | positive_rate: xx.xx%
- --- Ecosystem Guardrails (Top 1% selection) ---
  Streamer Gini: x.xxx | Top10 Share: xx.x% | Tail Coverage: xx.x% | Overload Streamer Rate: xx.x%

JSON 导出 to_dict() 需要包含这些字段（不包含 DataFrame）。

========================
3) 指标口径与“决策相关”对齐（务必写进注释/文档）
========================
在 metrics.py 顶部说明（或新增注释块）：
- RevCap@K 是“TopK 预测集合的真实收入占比”，对应决策里“你选中的人贡献了多少实际收益”（也可称 RevShare@K）。
- 校准分两类：
  (a) probability calibration：是否打赏概率（ECE + reliability）
  (b) value calibration：金额/EV 的 sum_ratio（现有 tail_calibration，可扩展为 decile 版）
- slice metrics 用于回答“冷启动、whale、头/尾用户是否被系统性损害”
- ecosystem metrics 是分配层必备护栏：集中度（Gini）、覆盖率（Coverage）、过载（Overload）

========================
4) 单元测试（强烈推荐）
========================
新增 tests/test_metrics_upgrade.py（pytest），至少覆盖：
1) compute_calibration:
- 完美校准样例：y_prob=[0,0,1,1], y_true=[0,0,1,1] => ece=0
- 明显失配样例：y_prob 全 0.9 但 y_true 正例率 0.1 => ece 接近 |0.1-0.9|
2) gini_coefficient:
- 全等数组 => gini=0
- 极端集中 => gini 接近 1
3) compute_ecosystem_metrics:
- 构造小 df，确保 coverage、overload 字段存在且不报错
4) compute_slice_metrics:
- 缺列时能返回 skipped reasons，而不是 raise

========================
5) 交付清单
========================
请直接输出：
- 修改后的 metrics.py（包含新增函数 + EvalResult/evaluate_model 集成）
- 如果写了测试：给出 tests 文件内容
- 给一个最小 usage 示例（如何在 evaluate_model 里传 y_prob / slice_config / ecosystem_config）
- 简短说明：新增指标的意义与如何解读（面向决策：校准/冷启动/生态护栏）

开始实现前请先快速阅读现有 metrics.py，复用其中的 DEFAULT_K_VALUES、tail_calibration、compute_all_metrics_at_k 等已有结构，保持风格一致。

你现在这套（排序/校准/切片/生态）已经非常接近“可用于分配”的指标闭环了。还缺的主要是 **把“预测→决策→约束→长期”这条链补齐**：也就是 *成本/风险*、*稳定性*、*反事实评估*、*策略层 KKT/影子价格* 这些。下面我按优先级给你一份“还值得加的指标清单”，基本都能落在 `metrics.py` 里。也会顺便指出哪些与你现在 repo 的 Gate（估计层→分配层→评估层）最匹配。  

---

## A. 决策层最缺的：成本/风险 + “错分代价”指标（强烈建议加）

### 1) **Misallocation Cost（错分代价）**

你现在的 RevCap@K/RevShare@K 只告诉你“抓住了多少收入”，但分配系统真正怕的是：**把稀缺大哥分给错误主播**（机会成本巨大）。

* 定义（离线近似）：在 TopK 选中集合里，计算

  * `wasted_exposure = Σ (1[y_true==0] * exposure_weight)`
  * `opportunity_cost = oracle_revenue@K - achieved_revenue@K`
* 价值：同样 RevShare@1% 的两个模型，错分代价可能差 2×，上线风险完全不同。

### 2) **Precision@K / GiftRate@K（TopK 的“命中率”）**

TopK 捕获只看收入占比，可能被极端大额掩盖；你还需要“TopK 里到底有多少是真的会送礼”。

* `gift_rate@K = mean(y_true>0 | selected)`
* `precision_whale@K = mean(y_true >= whale_thr | selected)`
* 对应你数据极稀疏（per-click gift rate ~1–2%）的现实。 

### 3) **Dollar-Weighted Calibration（金额加权的校准）**

你已有概率 ECE + 生态里 sum(pred)/sum(actual) 的想法，但建议把 **“金额加权”**显式加入：

* `DW-ECE = Σ_bin (sum_y_bin / sum_y) * |avg_pred - avg_true|`
* 价值：对重尾分布（Top 1% 用户贡献 ~60%）更贴近业务风险。 

---

## B. 排序质量的“形状”指标：不要只看某一个 K（建议加）

### 4) **Capture Curve AUC / “收益-覆盖”曲线**

你已有 Top-1/5/10，但建议输出整条曲线并取面积：

* `AUC_revcap = ∫_0^α RevCap(k) dk`（α 可取 10% 或 20%）
* 价值：避免“只优化 1% 导致 5%/10% 崩”的过拟合（你 fair 对比里 Top-1% 与 NDCG@100 出现了明显 trade-off）。 

### 5) **NDCG@K（多个 K）+ “whale-weighted NDCG”**

你已有 NDCG@100，但建议：

* 多个 K（50/100/500/1000）
* whale-weighted：把 y_true 在贡献更大的区域权重放大（例如对 y_true 做分位数权重）
  这能解释“Two-stage 的 NDCG@100 更好但 Top-1% 差”的现象。 

---

## C. 稳定性与可上线性：这是“决策相关”经常被忽略的（建议加）

### 6) **Temporal Stability（跨天/跨小时稳定性）**

离线分数很高但每天乱跳，上线会被骂。

* 指标：按天（或小时）分桶计算 Top-1% RevShare、ECE、Overload
* 输出：均值 / 标准差 / 最差分位（P10）
* 价值：对你按天切分训练非常一致。 

### 7) **Score Drift / PSI（预测分布漂移）**

* PSI / KL / Wasserstein：train vs test、或周与周
* 价值：你系统依赖历史交互特征（pair_*），冷启动比例高时漂移风险大。 

---

## D. 分配层护栏再补两块：多样性与公平（建议加）

你已经有 Streamer Gini / Tail Coverage / Overload，很好。还建议加：

### 8) **Diversity / Entropy（分配多样性）**

* 对 selected set 的 streamer 分布：`entropy(streamer_id)`、`effective_num = exp(entropy)`
* 比 Gini 更灵敏，能捕捉“看起来覆盖了很多主播，但流量仍高度集中”的情况。

### 9) **Fairness Gap（切片之间的收益差距）**

把你 slice_metrics 进一步结构化成 “gap”：

* `gap_revcap = RevCap@K(whale) - RevCap@K(tail)`
* `gap_coldstart = RevCap@K(cold_start) / RevCap@K(non_cold)`
  这能直接变成分配约束/拉格朗日项的监控指标（Gate-2 很需要）。 

---

## E. 评估层（OPE/Simulator）需要的指标：如果你要“做分配”，最终一定得加（中期必加）

你 roadmap 里 Gate-3 就是 OPE（IPS/DR）。 

### 10) **OPE Metrics（IPS/SNIPS/DR）+ 方差与置信区间**

* 输出：point estimate + std/CI（bootstrap 或 self-normalized）
* 必须有：effective sample size（ESS），否则 OPE 数字容易幻觉。

### 11) **Policy Constraint Satisfaction Rate（约束满足率）**

当你做“凹收益 + 约束”分配：

* `cap_violation_rate`（主播承接上限违反比例）
* `min_exposure_satisfaction`（新主播最低曝光满足率）
  这些是分配策略是否“可上线”的关键 KPI。

---

## F. 你现在这套里，我会立刻补的 Top 6（最值回票价）

1. Precision@K / GiftRate@K（以及 whale precision）
2. Opportunity Cost / Misallocation Cost（对稀缺大哥最关键）
3. Capture Curve AUC（避免只盯 1%）
4. Temporal Stability（按天/小时的均值+波动）
5. PSI/Drift（训练/测试预测分布漂移）
6. Diversity/Entropy（补齐生态的“多样性”维度）

---

如果你想，我可以把这些也写成和你之前一样的**升级 prompt**（继续扩展 `metrics.py`：`compute_topk_precision() / compute_capture_auc() / compute_stability() / compute_drift() / compute_misallocation_cost() / compute_diversity()`），并且把它们挂到 `evaluate_model()` 的输出里，保持向后兼容。 

下面我按你的 3 个目标来“修订 v2.1 指标体系”，重点是：**口径是否自洽、名字是否工业界常用、RevCap 是否需要补充“金额/业务可读”版本**。我会尽量给出你可以直接写回手册的表述。

---

## （1）口径/公式检查：哪些地方容易“算错”或“写得不自洽”

### ✅ 1) Misallocation Cost 这一节：**Oracle@K 的单位混了**

你在 1.2 里定义的 **Oracle@K 是一个比例（RevCap 的理论上限）**，但在 7.1 里 Opportunity Cost 又写成“元”的差值：

> Opportunity Cost = Oracle@K - Achieved@K（元）

这会造成**单位不一致**（比例 vs 金额）。建议把 7.1 改成显式的“金额版”和“比例版”两套：

* **AchievedRev@K（元）**
  [
  \text{AchievedRev@K} = \sum_{i \in \text{Top}_K(\hat{y})} y_i
  ]

* **OracleRev@K（元）**
  [
  \text{OracleRev@K} = \sum_{i \in \text{Top}_K(y)} y_i
  ]

* **Regret@K（元）**（原 Opportunity Cost）
  [
  \text{Regret@K} = \text{OracleRev@K} - \text{AchievedRev@K}
  ]

* **Efficiency@K（无量纲）**
  [
  \text{Eff@K} = \frac{\text{AchievedRev@K}}{\text{OracleRev@K}}
  ]

* **RegretPct@K（无量纲）**
  [
  \text{RegretPct@K} = 1 - \text{Eff@K}
  ]

> 你现在表里的 “Opportunity Cost % + Efficiency” 是自洽的（48.4% + 51.6% = 100%），说明你实际算的很可能是 **RegretPct**，只是文档把它叫 Opportunity Cost% 了。建议统一改名。

---

### ✅ 2) Capture AUC：**AUC 的数值范围要讲清楚，否则容易误读**

你写：
[
\text{Capture AUC}=\int_0^{max_k} RevCap(k),dk
]
如果 `max_k=0.1`，那么这个积分的范围理论上是 `[0, 0.1]`（因为 RevCap ≤ 1），**不可能直接是 61.2%**。

你手册里 “Baseline: Normalized AUC = 61.2%” 是合理的，但名字要对齐：

建议在文档里明确三种量（只要你愿意，保留 1–2 个即可）：

* **CapAUC@10%（原始积分）**：数值在 `[0, 0.1]`
* **MeanRevCap@10% = CapAUC@10% / 0.1**：数值在 `[0,1]`（更直观）
* **nAUC@10% = CapAUC@10% / OracleAUC@10%**：数值在 `[0,1]`（你现在的“Normalized AUC”本质就是这个）

> 你的 “61.2%” 基本一定对应 **nAUC**（或 MeanRevCap），不要把它叫 “Capture AUC” 以免读者以为是原始积分。

---

### ⚠️ 3) Overload Risk：当前定义和你前面“生态 Overload”语义可能不一致

你 v2.1 写的是：

> 在 10 分钟窗口内，被推送超过 3 次的高价值用户比例

这是 **“用户侧过度触达”**（user fatigue / spam risk）。

但你更早提过的生态 overload 常见也有另一类：

* **“主播侧过载”**：某主播在短窗口内被塞入太多高价值用户，导致承接失败/体验差

建议你在手册里把 Overload 拆成两个指标（工业界也更清晰）：

* **U-OverTarget@10m（用户侧过度触达）**：同一个用户在 10m 内被选中次数 > cap 的比例
* **S-OverLoad@10m（主播侧承接过载）**：同一个主播在 10m 内接收到高价值用户数 > cap 的 streamer×window 占比 或 streamer 占比

这样不会出现读者问：“Overload 到底是用户 overload 还是主播 overload？”

---

### ⚠️ 4) Tail Calibration：它不是“校准”通用指标，前提是 **y_pred 必须是金额刻度**

你的定义是：
[
\text{Sum Ratio}=\frac{\sum \hat{y}}{\sum y}
]
这在 **EV（金额）模型**里非常关键，但如果某些模型输出只是 ranking score（比如 tree 的 raw score、pairwise ranker），这个 ratio 会变得没有意义。

建议手册里加一句硬约束：

* **Tail Calibration / SumRatio 只对 “y_pred 是同一金额刻度的 EV” 有意义；否则跳过或仅作为相对比较**

同时建议把它更名为更工业界的名字（下面第 2 部分会给）。

---

### ✅ 5) Whale 阈值：你写“默认 100 元约为 P90”，最好明确 **P90 是在 gifters 上算**

你前文口径是对的（P90 of gifters），建议在文档里写死：

* `whale_threshold = P90(y_true | y_true>0)`（或显式配置为 100）

否则有人会误用 `P90(y_true)`（全是 0 的情况下会变得离谱）。

---

## （2）指标命名精简：给一套“工业界风格缩写 + 统一命名规则”

你已经有一个很好的命名：`RevCap@K`。我建议你整体采用一套规则：

* **所有 Top 比例相关**都用 `@K`（K 是比例，比如 1%）
* **窗口相关**用 `@W`（比如 `@10m`）
* **归一化版本**统一加 `n` 前缀（normalized）：`nRevCap`, `nAUC`
* **金额版**统一加 `$` 或 `Rev` 后缀：`AchRev`, `Regret$`（写文档时用 `$`，代码用 `_amount`）

下面是你手册里核心指标的“短名建议表”（你可以直接贴到 9.2 或术语表里）：

### 主指标

* `RevCap@K`（保留）
* `OracleRevCap@K`（原 Oracle@K 比例版）
* `nRevCap@K`（= RevCap@K / OracleRevCap@K）

  > 这其实就是你现在 7.1 里的 Efficiency（在 oracle≈1 时数值很接近 RevCap）

### 诊断

* `WRec@K` = Whale Recall@K
* `WPrec@K` = Whale Precision@K
* `AvgRev@K`（你已经很短了）
* `GiftRate@K`（也很短）
* （可选）`RevLift@K = RevCap@K / K`（强烈推荐：一眼看出相对 random 提升倍数）

### 稳定性

* `CV@K`（或 `StabCV@K`）
* `CI95@K`（Bootstrap 95% CI）

### 校准

* `pECE`（probability ECE）
* `RelCurve`（Reliability Curve，作为结构化输出字段名即可）
* `SumRatio@K`（你现在 Tail Calibration 的 sum_ratio）
* `MeanRatio@K`（mean_ratio）

> “Tail Calibration” 这个名字偏描述性，工业界更常直接叫 **Calibration Ratio / Overprediction Ratio**。所以 `SumRatio@K` 很合适。

### 生态/多样性

* `GiniS@K`（streamer revenue gini in selected set）
* `CovS@K`（streamer coverage）
* `TailCovS@K`（tail streamer coverage）
* `ColdCovS@K`（cold streamer coverage）
* `Ent@K`（entropy）
* `EffN@K`（effective number = exp(entropy)）
* `HHI@K`（Herfindahl index）
* `Top10Share@K`（保留即可）

### 决策核心（你觉得名字长的那几个）

* **Opportunity Cost（建议改名）**：

  * `Regret$@K`（元，OracleRev - AchievedRev）
  * `RegretPct@K`（比例）
* **Efficiency（建议统一为 normalized 命名）**：

  * `Eff@K` 或 `nRevCap@K`（二选一，推荐 `nRevCap@K`，语义更明确）
* **Capture AUC（建议改名避免歧义）**：

  * `nAUC@10%`（推荐，直接说明是 normalized）
  * 或 `CapAUC@10%`（若你保留原始积分）

---

## （3）RevCap@K 是否不够？要不要金额？RevShare 要不要单独存在？

### 结论先说清：**RevCap@K 作为“主指标”足够，但必须配 2 类补充**

RevCap@K 的优点是：**对不同实验/不同总收入规模天然可比**。所以它非常适合作为主指标。

但如果你的目标是“支持分配决策”，我建议固定再配两类东西：

#### A) 加一组“金额可读”的指标（但不要替代 RevCap）

原因：产品/业务同学在做上线决策时，会问**“到底是多少钱”**，而不仅是比例。

推荐新增（或在 summary 里附带打印）：

* `AchievedRev@K`（元）= TopK 真实收入和
* `OracleRev@K`（元）
* `Regret$@K`（元）
* `AvgRev@K`（你已有）

注意：在**同一个 test set** 内，`AchievedRev@K = RevCap@K * TotalRevenue`，信息理论上是重复的；但在**跨天/跨实验/跨流量池**时，金额版非常有沟通价值。

#### B) 明确 “RevCap vs RevShare”：**数学上通常是一回事，不要重复造两个词**

如果你定义的 RevShare@K 还是：

[
\frac{\sum_{i\in TopK(\hat{y})} y_i}{\sum_i y_i}
]

那它和 RevCap@K **完全同义**（只是不同叫法）。为了指标体系统一，建议：

* **文档里写一句：RevCap@K（也称 RevShare@K）**
* 代码/报表只保留一个名字：`RevCap@K`

什么时候 RevShare 才需要单独存在？只有当你改变分母/对象，比如：

* **按主播聚合后的份额**：选中的主播集合贡献收入占比（entity-level share）
* **按曝光预算加权后的份额**：分母变成“被分配的总预算/总曝光”等

如果你确实想表达“分配策略选中实体（主播/用户）后带来的收入份额”，可以用更明确的命名避免混淆：

* `RevCap_pair@K`（pair-level）
* `RevCap_streamer@K`（先按 streamer 聚合再算）

---

## 你这版 v2.1 我建议立刻改的 5 处“文档级优化”

1. 把 7.1 的 Oracle@K / Achieved@K **拆成金额版和比例版**（避免单位混乱）
2. 把 Opportunity Cost 改名为 **Regret（工业界/学术都通用）**
3. 把 Capture AUC 改成 **nAUC@10%**（或者明确原始积分范围）
4. Overload 拆成 **用户侧 OverTarget** 和 **主播侧 OverLoad**（避免语义漂移）
5. Tail Calibration 加一句前提：**只有 EV 金额刻度输出才有意义**

---

如果你愿意，我也可以把你这份手册按上述修改点，直接给你一份“可复制回文档的修订版片段”（尤其是第 7 章命名与公式那块，改完会非常清爽、不会被 reviewer 挑口径问题）。
