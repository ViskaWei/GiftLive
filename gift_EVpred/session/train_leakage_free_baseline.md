## (1) 只有数学公式的简短伪代码（math-only）

[
\mathcal{G}={(u_j,s_j,\ell_j,t_j,a_j)}_{j=1}^{M},\quad
\mathcal{C}={(u_i,s_i,\ell_i,t_i,\mathrm{ctx}*i)}*{i=1}^{N},\quad
H=1\text{ hour}
]

[
Y_i \triangleq \sum_{j=1}^{M} a_j\cdot \mathbf{1}!\left[(u_j,s_j,\ell_j)=(u_i,s_i,\ell_i)\right]\cdot \mathbf{1}!\left[t_i\le t_j \le t_i+H\right]
]
[
y_i \triangleq \log(1+Y_i),\quad z_i\triangleq \mathbf{1}[Y_i>0]
]

[
\pi:{1,\ldots,N}\to{1,\ldots,N},\quad t_{\pi(1)}\le\cdots\le t_{\pi(N)}
]
[
\mathcal{D}*{\mathrm{tr}}={\pi(1),\ldots,\pi(n*{\mathrm{tr}})},\
\mathcal{D}*{\mathrm{va}}={\pi(n*{\mathrm{tr}}+1),\ldots,\pi(n_{\mathrm{va}})},\
\mathcal{D}*{\mathrm{te}}={\pi(n*{\mathrm{va}}+1),\ldots,\pi(N)}
]

[
[t_{\min},t_{\max}]\triangleq \left[\min_{i\in\mathcal{D}*{\mathrm{tr}}}t_i,\ \max*{i\in\mathcal{D}*{\mathrm{tr}}}t_i\right],\quad
\mathcal{G}*{\mathrm{tr}}\triangleq {j:\ t_{\min}\le t_j\le t_{\max}}
]

[
C^{\mathrm{tr}}*{u,s}\triangleq \sum*{j\in\mathcal{G}*{\mathrm{tr}}}\mathbf{1}[(u_j,s_j)=(u,s)],\quad
S^{\mathrm{tr}}*{u,s}\triangleq \sum_{j\in\mathcal{G}*{\mathrm{tr}}}a_j\mathbf{1}[(u_j,s_j)=(u,s)]
]
[
\mu^{\mathrm{tr}}*{u,s}\triangleq \frac{S^{\mathrm{tr}}*{u,s}}{\max(1,C^{\mathrm{tr}}*{u,s})},\quad
\tau^{\mathrm{tr}}*{u,s}\triangleq \max*{j\in\mathcal{G}*{\mathrm{tr}}:\ (u_j,s_j)=(u,s)} t_j
]
[
x^{\mathrm{past}}*i \triangleq \big(C^{\mathrm{tr}}*{u_i,s_i},\ S^{\mathrm{tr}}*{u_i,s_i},\ \mu^{\mathrm{tr}}*{u_i,s_i},\ \Delta t_i\big),\quad
\Delta t_i \triangleq
\begin{cases}
\frac{t_i-\tau^{\mathrm{tr}}*{u_i,s_i}}{3600\cdot 10^3} & \tau^{\mathrm{tr}}_{u_i,s_i}\ \text{exists}\
999 & \text{otherwise}
\end{cases}
]
[
x_i \triangleq \big(x^{\mathrm{static}}_i,\ x^{\mathrm{ctx}}_i,\ x^{\mathrm{past}}_i\big)
]

[
f^\star=\arg\min_{f\in\mathcal{F}}\sum_{i\in\mathcal{D}_{\mathrm{tr}}}\left(y_i-f(x_i)\right)^2,\quad
\widehat{Y}^{\mathrm{dir}}_i=\exp!\left(f^\star(x_i)\right)-1
]

[
p^\star=\arg\min_{p\in\mathcal{P}}\sum_{i\in\mathcal{D}*{\mathrm{tr}}}\ell*{\log}(z_i,p(x_i)),\quad
m^\star=\arg\min_{m\in\mathcal{M}}\sum_{i\in\mathcal{D}_{\mathrm{tr}}:z_i=1}\left(Y_i-m(x_i)\right)^2
]
[
\widehat{Y}^{\mathrm{2s}}_i=p^\star(x_i)\cdot m^\star(x_i)
]

[
\mathrm{MAE}*{\log}=\frac{1}{|\mathcal{D}*{\mathrm{te}}|}\sum_{i\in\mathcal{D}*{\mathrm{te}}}\left|y_i-\log(1+\widehat{Y}*i)\right|
]
[
\rho_S=\mathrm{Spearman}\left({Y_i}*{i\in\mathcal{D}*{\mathrm{te}}},{\widehat{Y}*i}*{i\in\mathcal{D}*{\mathrm{te}}}\right)
]
[
K=\left\lfloor \alpha |\mathcal{D}*{\mathrm{te}}|\right\rfloor,\quad
\mathrm{TopK}(\widehat{Y})=\operatorname*{arg,topK}*{i\in\mathcal{D}*{\mathrm{te}}}\widehat{Y}*i,\quad
\mathrm{TopK}(Y)=\operatorname*{arg,topK}*{i\in\mathcal{D}*{\mathrm{te}}}Y_i
]
[
\mathrm{Capture@}\alpha=\frac{\left|\mathrm{TopK}(\widehat{Y})\cap\mathrm{TopK}(Y)\right|}{\left|\mathrm{TopK}(Y)\right|},\quad
\mathrm{RevShare@}\alpha=\frac{\sum*{i\in\mathrm{TopK}(\widehat{Y})}Y_i}{\sum_{i\in\mathcal{D}*{\mathrm{te}}}Y_i}
]
[
\mathrm{NDCG@}k=\frac{\sum*{r=1}^k \frac{Y_{(r)}}{\log_2(r+1)}}{\sum_{r=1}^k \frac{Y^{\downarrow}*{(r)}}{\log_2(r+1)}}
]
[
\mathrm{ECE}=\sum*{b=1}^{B}\frac{n_b}{n}\left|\ \frac{1}{n_b}\sum_{i\in b}z_i-\frac{1}{n_b}\sum_{i\in b}\widehat{p}_i\ \right|
]

---

## (2) 长版本解释：逐段把代码“翻译成公式 + 解释”（含预处理与评估口径）

下面我用**“代码变量 → 数学符号”**的方式，把 `train_leakage_free_baseline.py` 的核心流程完整公式化，并把**预处理、特征构造、训练目标、评估指标**都写清楚（包括你在报告里强调的 Gate-0 核心检查：泄漏/任务/评估对齐）。同时也会点出代码里两个关键的“实现风险点”（rolling 特征、校准曲线）——这也是你实验报告里对 rolling 版本“异常高”的解释来源。

### 2.1 记号与数据表

**事件表：**

* Click 表（一次进房/曝光机会）
  [
  \mathcal{C}={c_i}_{i=1}^{N},\quad c_i=(u_i,s_i,\ell_i,t_i,w_i,\text{ctx}_i)
  ]
  其中 (u_i)=user_id，(s_i)=streamer_id，(\ell_i)=live_id，(t_i)=click timestamp(ms)，(w_i)=watch_live_time（代码里存在但标签窗口未用它截断）。

* Gift 表（一次打赏交易）
  [
  \mathcal{G}={g_j}_{j=1}^{M},\quad g_j=(u_j,s_j,\ell_j,t_j,a_j)
  ]
  (a_j=\text{gift_price})。

此外还有 User/Streamer/Room 静态画像表，后面记为 (x^{\text{static}})。

---

### 2.2 预处理：Click-level 标签构造（EV 定义）

代码里把“样本单元”从 gift-only 改成 click-level，并定义 1 小时窗口 (H=1\text{h})。

对每个 click (c_i)，定义“未来 (H) 内同一 (user, streamer, live) 的打赏总额”为：

[
Y_i \triangleq \sum_{j=1}^{M} a_j\cdot
\mathbf{1}!\left[(u_j,s_j,\ell_j)=(u_i,s_i,\ell_i)\right]\cdot
\mathbf{1}!\left[t_i\le t_j \le t_i+H\right]
]

这就是你报告里写的 click-level EV 标签（包含 0）。

为了适配重尾金额分布（P99/P50 极大），代码做了 log1p 变换作为回归目标：

[
y_i \triangleq \log(1+Y_i)
]

并额外构造二分类标签：

[
z_i\triangleq \mathbf{1}[Y_i>0]
]

> **一个重要细节（代码口径）**：
> 标签窗口在代码里是 ([t_i, t_i+H])，**没有**用观看时长 (w_i) 做截断，即没有使用 ([t_i, \min(t_i+w_i,t_i+H)])。如果真实业务上“只能在观看期间送礼”严格成立，那么用 (w_i) 截断会更严谨。

---

### 2.3 时间切分（Temporal Split）

把样本按点击时间排序后切分：

[
t_{\pi(1)}\le t_{\pi(2)}\le\cdots\le t_{\pi(N)}
]

按比例（代码默认 70%/15%/15%）：

[
\mathcal{D}*{\mathrm{tr}}={\pi(1),\ldots,\pi(\lfloor0.7N\rfloor)}
]
[
\mathcal{D}*{\mathrm{va}}={\pi(\lfloor0.7N\rfloor+1),\ldots,\pi(\lfloor0.85N\rfloor)}
]
[
\mathcal{D}_{\mathrm{te}}={\pi(\lfloor0.85N\rfloor+1),\ldots,\pi(N)}
]

这一步是“避免未来信息穿越”的基础。

---

### 2.4 特征工程：Past-only 特征（Frozen 版）

你报告里把 **Frozen** 定位为“严格下界（val/test 只能查 train 统计表）”。
代码实现是：先确定 train 的时间范围：

[
[t_{\min},t_{\max}] \triangleq
\left[\min_{i\in\mathcal{D}*{\mathrm{tr}}}t_i,\ \max*{i\in\mathcal{D}*{\mathrm{tr}}}t_i\right]
]
[
\mathcal{G}*{\mathrm{tr}}={j:\ t_{\min}\le t_j \le t_{\max}}
]

然后用 (\mathcal{G}_{\mathrm{tr}}) 计算聚合统计，形成 lookup table（对 val/test 只查表，不再更新）。

#### 2.4.1 Pair（user-streamer）历史特征

对任意 ((u,s))：

[
C^{\mathrm{tr}}*{u,s}=\sum*{j\in\mathcal{G}*{\mathrm{tr}}}\mathbf{1}[(u_j,s_j)=(u,s)]
]
[
S^{\mathrm{tr}}*{u,s}=\sum_{j\in\mathcal{G}*{\mathrm{tr}}}a_j\mathbf{1}[(u_j,s_j)=(u,s)]
]
[
\mu^{\mathrm{tr}}*{u,s}=\frac{S^{\mathrm{tr}}*{u,s}}{\max(1,C^{\mathrm{tr}}*{u,s})}
]
[
\tau^{\mathrm{tr}}*{u,s}=\max*{j\in\mathcal{G}_{\mathrm{tr}}:(u_j,s_j)=(u,s)} t_j
]

对样本 (i)（click）提取：

[
\text{pair_gift_count_past}(i)=C^{\mathrm{tr}}*{u_i,s_i}
]
[
\text{pair_gift_sum_past}(i)=S^{\mathrm{tr}}*{u_i,s_i}
]
[
\text{pair_gift_mean_past}(i)=\mu^{\mathrm{tr}}*{u_i,s_i}
]
[
\text{pair_last_gift_time_gap_past}(i)=
\begin{cases}
\frac{t_i-\tau^{\mathrm{tr}}*{u_i,s_i}}{3600\cdot10^3} & \text{若}\ \tau^{\mathrm{tr}}_{u_i,s_i}\ \text{存在}\
999 & \text{否则}
\end{cases}
]

#### 2.4.2 User 历史特征

代码里是“train window 内总打赏”，变量名叫 `user_total_gift_7d_past`，但实际上不是严格 7d（这是命名/口径不一致点）：

[
U^{\mathrm{tr}}*{u}=\sum*{j\in\mathcal{G}*{\mathrm{tr}}}a_j\mathbf{1}[u_j=u]
]
[
\text{user_total_gift_7d_past}(i)=U^{\mathrm{tr}}*{u_i},\quad
\text{user_budget_proxy_past}(i)=U^{\mathrm{tr}}_{u_i}
]

#### 2.4.3 Streamer 历史特征

[
R^{\mathrm{tr}}*{s}=\sum*{j\in\mathcal{G}*{\mathrm{tr}}}a_j\mathbf{1}[s_j=s]
]
[
G^{\mathrm{tr}}*{s}=\left|{u_j:\ j\in\mathcal{G}_{\mathrm{tr}},\ s_j=s}\right|
]

[
\text{streamer_recent_revenue_past}(i)=R^{\mathrm{tr}}*{s_i},\quad
\text{streamer_recent_unique_givers_past}(i)=G^{\mathrm{tr}}*{s_i}
]

---

### 2.5 特征工程：Past-only 特征（Rolling 版）——“正确公式”与“代码实现风险”

**你报告里已经明确：Rolling 版本指标异常高，疑似时间泄漏，以 Frozen 为准。**
这类 rolling 特征的“正确数学定义”应是：

对任意实体 (e)（比如 pair=(u,s) 或 user=u 或 streamer=s），在时间 (t) 的“过去累计”应满足严格的过去约束 (t_j<t)：

[
C_{e}(t)=\sum_{j=1}^{M}\mathbf{1}[e_j=e]\cdot \mathbf{1}[t_j<t]
]
[
S_{e}(t)=\sum_{j=1}^{M}a_j\mathbf{1}[e_j=e]\cdot \mathbf{1}[t_j<t]
]
[
\mu_{e}(t)=\frac{S_e(t)}{\max(1,C_e(t))}
]
[
\tau_e(t)=\max_{j:\ e_j=e,\ t_j<t} t_j
]

对 click (i) 的 past 特征就是把 (t=t_i) 代入：

[
x^{\mathrm{past}}_i = \big(C_e(t_i),S_e(t_i),\mu_e(t_i), (t_i-\tau_e(t_i))/\text{hour}\big)
]

> **代码风险点（导致“异常高”的根因）**：
> 你脚本里的 rolling 部分虽然名字叫 “cumsum + shift”，但实际实现为了简化，最终把**全量 gift 的 groupby 总统计** merge 回 click（没有对每个 click 做 (t_j<t_i) 的截断/对齐），这等价于把未来信息喂给了特征，因此会出现报告里提到的 **Top-1% 81.1% / RevCap@1% 98.7% / Stage1 AUC≈0.999** 这种“近乎开卷”的异常现象。

如果要把 rolling 做到“线上可用且无泄漏”，实现上通常需要“按时间 asof join / 二分定位最后一次 gift”来保证 (t_j<t_i)。

---

### 2.6 静态特征与上下文特征

代码里从 user/streamer/room 表 merge 的画像特征记为：

[
x^{\mathrm{static}}*i = \phi*{\mathrm{user}}(u_i)\ \Vert\ \phi_{\mathrm{streamer}}(s_i)\ \Vert\ \phi_{\mathrm{room}}(\ell_i)
]

上下文时间特征（由 (t_i) 转为 datetime）：

[
\text{hour}_i\in{0,\ldots,23},\quad
\text{dow}_i\in{0,\ldots,6},\quad
\text{is_weekend}_i=\mathbf{1}[\text{dow}_i\ge 5]
]

最终特征向量：

[
x_i = x_i^{\mathrm{static}}\ \Vert\ x_i^{\mathrm{ctx}}\ \Vert\ x_i^{\mathrm{past}}
]

**缺失值与类别编码（代码口径）**：

* 数值缺失：(\text{NaN}\mapsto 0)（以及 gap 的特殊填充值 999）。
* 类别编码：对类别列 (c) 学一个映射 (\pi_c:\text{category}\to{0,1,\dots})，用 (\pi_c(c_i)) 替换。

---

## 2.7 模型：Direct Regression（直接回归）

你报告里称之为 “Direct Regression：预测 log(1+Y)”。

训练目标（LightGBM objective=regression，等价于 L2/MSE）：

[
f^\star = \arg\min_{f\in\mathcal{F}} \sum_{i\in\mathcal{D}_{\mathrm{tr}}}\left(y_i - f(x_i)\right)^2
]

预测：

[
\widehat{y}_i = f^\star(x_i),\quad
\widehat{Y}^{\mathrm{dir}}_i = \exp(\widehat{y}_i)-1
]

> **统计学提醒（口径要写清楚）**：
> (\widehat{Y}^{\mathrm{dir}}=\exp(\widehat{y})-1) 并不严格等于 (\mathbb{E}[Y|x])，因为训练拟合的是 (\mathbb{E}[\log(1+Y)|x])。不过由于 (\exp(\cdot)) 单调，这个变换对排序类指标通常仍可用（但会影响“绝对值校准/金额期望”的解释）。

---

## 2.8 模型：Two-Stage（两段式）

你报告里定义为：

[
\widehat{Y}(x)=\widehat{p}(x)\cdot \widehat{m}(x)
]
其中 Stage2 预测 raw amount 条件期望，保证量纲正确。

### Stage 1：是否打赏概率

[
p^\star=\arg\min_{p\in\mathcal{P}}\sum_{i\in\mathcal{D}*{\mathrm{tr}}} \ell*{\log}\left(z_i,p(x_i)\right)
]
[
\ell_{\log}(z,p)= -z\log p-(1-z)\log(1-p)
]

### Stage 2：打赏金额（仅正样本）

[
m^\star=\arg\min_{m\in\mathcal{M}}\sum_{i\in\mathcal{D}_{\mathrm{tr}}:z_i=1}\left(Y_i-m(x_i)\right)^2
]

组合：

[
\widehat{Y}^{\mathrm{2s}}_i=\widehat{p}_i\cdot \widehat{m}_i
]
并在 log 空间对齐评估时使用：
[
\widehat{y}^{\mathrm{2s}}_i=\log(1+\widehat{Y}^{\mathrm{2s}}_i)
]

> **工程建议（更稳健）**：
> 实际上 (m(x)) 可能输出负值（回归器无非负约束），建议上线/评估前做截断：(\widehat{m}\leftarrow\max(\widehat{m},0))，从而 (\widehat{Y}\ge 0)。

---

## 2.9 评估指标：全部给出“可直接对照代码”的公式

你这次实验把评估升级为 **Revenue Capture@K（收入占比）**，并保留传统的 Top-K% overlap、Spearman、NDCG、误差指标。

下面统一在 test 集 (\mathcal{D}*{\mathrm{te}}) 上定义，记 (n=|\mathcal{D}*{\mathrm{te}}|)。

### 2.9.1 MAE / RMSE（log 空间）

[
\mathrm{MAE}*{\log}=\frac{1}{n}\sum*{i\in\mathcal{D}_{\mathrm{te}}}\left|,\log(1+Y_i)-\log(1+\widehat{Y}_i)\right|
]
（Direct 用 (\widehat{Y}^{\mathrm{dir}})，Two-Stage 用 (\widehat{Y}^{\mathrm{2s}})）

[
\mathrm{RMSE}*{\log}=\sqrt{\frac{1}{n}\sum*{i\in\mathcal{D}_{\mathrm{te}}}\left(\log(1+Y_i)-\log(1+\widehat{Y}_i)\right)^2}
]

### 2.9.2 Spearman（raw 空间的排序相关）

Spearman 相关系数等价于对秩的 Pearson 相关：

[
\rho_S=\mathrm{corr}\left(\mathrm{rank}({Y_i}),\mathrm{rank}({\widehat{Y}_i})\right)
]

### 2.9.3 Top-K% Capture（集合重叠/召回式指标）

令 (\alpha\in(0,1))，(K=\lfloor \alpha n\rfloor)。

[
\mathrm{TopK}(Y)=\operatorname*{arg,topK}*{i\in\mathcal{D}*{\mathrm{te}}} Y_i,\quad
\mathrm{TopK}(\widehat{Y})=\operatorname*{arg,topK}*{i\in\mathcal{D}*{\mathrm{te}}} \widehat{Y}_i
]

则：

[
\mathrm{Capture@}\alpha=\frac{|\mathrm{TopK}(Y)\cap \mathrm{TopK}(\widehat{Y})|}{|\mathrm{TopK}(Y)|}
]

> 这是你报告里说的“集合重叠，不关心金额差异”的指标。

### 2.9.4 Revenue Capture@K（收入占比 / 业务对齐）

你新引入的核心指标（代码实现 `compute_revenue_capture_at_k`）：

[
\mathrm{RevShare@}\alpha=
\frac{\sum_{i\in \mathrm{TopK}(\widehat{Y})} Y_i}
{\sum_{i\in\mathcal{D}_{\mathrm{te}}} Y_i}
]

> 解释：不是问“命中了多少个 top 金主”，而是问“把多少收入集中在模型选出的 top-K% 上”。这就是你在报告里用来替换 Top-K overlap 的理由。

### 2.9.5 NDCG@100（Top-100 精细排序）

令 (k=100)。对预测排序后的第 (r) 名样本 index 为 (i_{(r)})，则：

[
\mathrm{DCG}@k=\sum_{r=1}^{k}\frac{Y_{i_{(r)}}}{\log_2(r+1)}
]

理想排序（按 (Y) 从大到小）的 DCG：

[
\mathrm{IDCG}@k=\sum_{r=1}^{k}\frac{Y^{\downarrow}_{(r)}}{\log_2(r+1)}
]

[
\mathrm{NDCG}@k=\frac{\mathrm{DCG}@k}{\mathrm{IDCG}@k}
]

### 2.9.6 ECE（Expected Calibration Error，用于概率模型）

对二分类概率输出 (\widehat{p}_i\in[0,1])，分桶 (B) 个区间（等宽）：

[
\mathrm{ECE}=\sum_{b=1}^{B}\frac{n_b}{n}\left|\ \underbrace{\frac{1}{n_b}\sum_{i\in b} z_i}_{\text{acc}*b}-\underbrace{\frac{1}{n_b}\sum*{i\in b} \widehat{p}*i}*{\text{conf}_b}\ \right|
]

> **重要口径**：ECE 适用于 Stage1 这类“概率输出”的模型。你脚本里对 Direct Regression 画校准曲线时用的是 (\sigma(\widehat{y}))（把回归输出硬当 logit），这在概率意义上不成立，因此该校准图不应作为结论依据（更专业的版本应改为：只对 Stage1 画校准曲线/算 ECE）。

### 2.9.7 泄漏检查：特征重要性比（你的 Gate-0 口径）

设 LightGBM 的 gain importance 为 (I_1\ge I_2\ge\cdots)，定义：

[
\mathrm{ImpRatio}=\frac{I_1}{I_2}
]

Gate-0 的通过条件之一是：

[
\mathrm{ImpRatio}<2
]

你报告里 frozen 版达到了 1.23（通过泄漏消除检查）。

---

## (3) 补充完善后的实验报告（更专业、更清楚的版本，可直接替换原 md）

下面是一份“可直接贴回仓库”的 **改写增强版**（结构更清晰、口径更严谨、把已知缺陷写成 *Known Issues*，并把 Gate-0 的结论写成可执行决策）。其中所有关键数值我保持与你原报告一致：Frozen 的 Top-1%/RevShare@1%/Spearman/ImpRatio，以及 Rolling 的“异常高”告警与“以 Frozen 为准”。
（我也顺带把 baseline 的“泄漏证据”引用到 baseline_results：`pair_gift_mean` 重要性远超第二名、Top-1%≈56.2%。）

---

# 🍃 EXP-20260118-gift_EVmodel-01 — Leakage-Free Baseline (Past-only + Click-level EV)

**Owner**: Viska Wei
**Date**: 2026-01-18
**Status**: ✅ Completed
**Priority**: 🔴 Highest (Gate-0 prerequisite)

## 0. Executive Summary

### What we fixed (and why it matters)

This experiment addresses three critical issues in the previous baseline:

1. **Data leakage**: global aggregation features (e.g., `pair_gift_mean`) leak future information and dominate importance. 
2. **Task mismatch**: gift-only training learns (\mathbb{E}[Y\mid Y>0]) rather than click-level (\mathbb{E}[Y]). 
3. **Evaluation bias**: Top-K overlap ignores monetary differences; we add revenue-share metrics. 

### Gate-0 decision

* **Leakage removal check**: ✅ PASS (Frozen: feature importance ratio = 1.23 < 2). 
* **Model effectiveness check**: ❌ FAIL (Frozen: Top-1% Capture ≈ 11.5%, RevShare@1% ≈ 21.3%). 
* **Conclusion**: Leakage is removed, but current feature set has very weak predictive power → proceed to feature redesign / task simplification.

---

## 1. Problem Definition

### 1.1 Unit of prediction: click-level

Each click (c_i=(u_i,s_i,\ell_i,t_i,\mathrm{ctx}_i)) is a recommendation opportunity.

### 1.2 Label: gift EV within a fixed horizon

With horizon (H=1) hour:

[
Y_i=\sum_{j} a_j\cdot \mathbf{1}[(u_j,s_j,\ell_j)=(u_i,s_i,\ell_i)]\cdot \mathbf{1}[t_i\le t_j\le t_i+H]
]
[
y_i=\log(1+Y_i),\quad z_i=\mathbf{1}[Y_i>0]
]

**Sparsity**: (Y=0) ratio is ~98.50%. 

---

## 2. Data & Split

* Dataset: KuaiLive (click/gift/user/streamer/room). 
* Time range: 2025-05-04 → 2025-05-25. 
* Split: temporal split by click timestamp:

  * Train 70%
  * Val 15%
  * Test 15% 

---

## 3. Feature Engineering (Leakage-free by construction)

### 3.1 Static features (safe)

User / streamer / room attributes merged by id, plus time context:

* hour, day-of-week, weekend flag.

### 3.2 Past-only aggregate features

#### A) Frozen (strict lower bound; **no time travel**)

Compute all aggregates **using only train-window events** ([t_{\min}^{tr},t_{\max}^{tr}]), store as lookup tables, and apply to val/test without update.

Pair ((u,s)):
[
C^{tr}*{u,s}=\sum*{j\in \mathcal{G}*{tr}}\mathbf{1}[(u_j,s_j)=(u,s)]
]
[
S^{tr}*{u,s}=\sum_{j\in \mathcal{G}*{tr}}a_j\mathbf{1}[(u_j,s_j)=(u,s)],\quad
\mu^{tr}*{u,s}=\frac{S^{tr}*{u,s}}{\max(1,C^{tr}*{u,s})}
]
[
\Delta t_i=
\begin{cases}
(t_i-\tau^{tr}*{u_i,s_i})/\text{hour} & \tau^{tr}*{u_i,s_i}\ \text{exists}\
999 & \text{otherwise}
\end{cases}
]

User (u): (U^{tr}*u=\sum*{j\in\mathcal{G}*{tr}} a_j\mathbf{1}[u_j=u])
Streamer (s): (R^{tr}*s=\sum*{j\in\mathcal{G}*{tr}} a_j\mathbf{1}[s_j=s]), plus unique givers.

> Interpretation: Frozen is a conservative offline proxy. It prevents leakage but does not exploit “val/test accumulated history”.

#### B) Rolling (intended online proxy; **must satisfy (t_j<t_i)**)

Correct definition for any entity (e):
[
S_e(t)=\sum_{j}a_j\mathbf{1}[e_j=e]\mathbf{1}[t_j<t]
]
and evaluate at (t=t_i).

⚠️ **Known issue**: current rolling implementation shows abnormal performance and likely uses future gifts in features (suspected leakage). Therefore rolling results are **invalid** and not used for decisions. 

---

## 4. Models

### 4.1 Direct regression (single-stage)

Train a regressor on (y=\log(1+Y)):
[
f^\star=\arg\min_f \sum_{i\in tr}\left(y_i-f(x_i)\right)^2
]
Prediction in raw space for ranking:
[
\widehat{Y}^{dir}_i=\exp(f^\star(x_i))-1
]

### 4.2 Two-stage (p × m)

Stage1 (gift propensity):
[
p^\star=\arg\min_p \sum_{i\in tr}\ell_{\log}(z_i,p(x_i))
]
Stage2 (amount conditional on gift, trained on (z=1)):
[
m^\star=\arg\min_m \sum_{i\in tr:z_i=1}\left(Y_i-m(x_i)\right)^2
]
Combine:
[
\widehat{Y}^{2s}_i=p^\star(x_i)\cdot m^\star(x_i)
]

---

## 5. Metrics (offline)

### 5.1 Error in log space

[
\mathrm{MAE}*{\log}=\frac{1}{n}\sum*{i\in te}\left|\log(1+Y_i)-\log(1+\widehat{Y}_i)\right|
]

### 5.2 Ranking quality

Spearman on raw amounts:
[
\rho_S=\mathrm{Spearman}(Y,\widehat{Y})
]

NDCG@100:
[
\mathrm{NDCG}@100=\frac{\sum_{r=1}^{100}\frac{Y_{(r)}}{\log_2(r+1)}}{\sum_{r=1}^{100}\frac{Y^{\downarrow}_{(r)}}{\log_2(r+1)}}
]

### 5.3 Decision-aligned metric (recommended)

Revenue Capture@K (revenue share of top predicted K%):
[
\mathrm{RevShare@}\alpha=\frac{\sum_{i\in \mathrm{TopK}(\widehat{Y})}Y_i}{\sum_{i\in te}Y_i}
]

### 5.4 Leakage sanity check

Feature-importance dominance ratio:
[
\mathrm{ImpRatio}=\frac{I_1}{I_2}\quad (\text{Gain importance})
]
Rule of thumb: (\mathrm{ImpRatio}<2\Rightarrow) no single leakage-like feature dominates.

---

## 6. Results

### 6.1 Frozen (valid; leakage-free)

| Metric         | Direct | Two-Stage |     Target |
| -------------- | -----: | --------: | ---------: |
| Top-1% Capture |  11.5% |     11.8% |      > 40% |
| RevShare@1%    |  21.3% |     25.6% |      > 50% |
| Spearman       |  0.103 |     0.095 | drop < 0.2 |
| ImpRatio       |   1.23 |         — |        < 2 |

Gate-0: leakage removal ✅, model performance ❌. 

### 6.2 Rolling (invalid; suspected leakage)

Direct Top-1% ≈ 81.1%, RevShare@1% ≈ 98.7%, Stage1 AUC ≈ 0.999 → strongly indicates time leakage in rolling implementation. 

---

## 7. Interpretation

1. **Baseline performance was largely leakage-driven**: once past-only constraints are enforced, ranking power collapses (Top-1% from ~56% to ~11%).
2. **Sparsity dominates**: click-level includes ~98.5% zeros, making EV prediction extremely hard with only coarse historical aggregates. 
3. **Direct vs Two-Stage is inconclusive under weak features**: both are near-random; no reliable architecture conclusion can be drawn here. 

---

## 8. Known Issues / Technical Debt (must fix before next iteration)

1. **Rolling past-only features must be time-aligned** ((t_j<t_i)) using asof-join or equivalent; current implementation is not reliable. 
2. **Calibration**: do not sigmoid-transform regression outputs for probability calibration; ECE/reliability should be computed on Stage1 probabilities only. 
3. **“7d” naming**: `user_total_gift_7d` is computed over train window, not strict 7-day rolling; rename or implement true 7d. 
4. **Slice analysis placeholders**: cold-pair / cold-streamer slices are not actually computed yet. 

---

## 9. Next Steps (actionable)

* **Fix rolling features** with strict time constraint; add an automated leakage test (e.g., shuffle timestamps / AUC sanity checks).
* **Add dense real-time signals** (watch duration, likes, comments) and/or **sequence features**; current aggregates are insufficient. 
* **Task simplification**: first validate (P(Y>0)) (binary) and only then consider amount/EV. 

---

如果你希望我把 **rolling 特征**也写成“严格可实现”的数学形式（包含 *asof join* 的等价定义：对每个 click 找到最近一次 gift 并维护累计量），我也可以直接给出一套**可证明无泄漏**的公式与对应的实现伪代码（并把 Gate-0 的“异常高检测”写成自动化单元测试口径）。
