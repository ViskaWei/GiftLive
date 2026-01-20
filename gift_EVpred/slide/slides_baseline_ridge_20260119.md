# Baseline v2.2 Metrics 完整评估 — 5min 汇报

> **Experiment**: EVpred-20260119-baseline-v22  \
> **Author**: Viska Wei  \
> **Date**: 2026-01-19  \
> **Language**: 中文

---

## Sample-level 抓事件强（35×），User-level 找大哥中等（12×，池子纯度 73%）

- **目的**：用 metrics v2.2 完整评估 Final Baseline，验证决策核心指标，判断是否可上线
- **X 公式**：$X := \underbrace{\text{Ridge Regression (α=1.0)}}_{\text{是什么}}\;\xrightarrow{\text{raw Y + Last-Touch 归因}}\;\underbrace{\text{预测打赏金额 EV}}_{\text{用于 TopK 筛选}}$
- **结论**：RevCap@1%=**51.4%**，WRecLift=**34.6×**（强），PSI=**0.155**（可上线）

| 类型 | 符号 | 说明 | 示例 |
|------|------|------|------|
| 🫐 输入 | $\mathcal{D}$ | 测试集 | N=1.4M, 20 Strict 特征 |
| 🫐 输入 | $\hat{y}$ | Ridge 预测值 | y_pred = [0.1, 48.3, ...] |
| 🫐 输出 | $M$ | 38 个指标 | RevCap, WRecLift, PSI, EffN... |
| 📊 指标 | RevCap@1% | 主指标 | **51.4%** |
| 📊 指标 | WRecLift@1% | 大额事件识别 | **34.6×** |
| 🍁 基线 | Random | sanity check | RevCap@1% ≈ 1% |

<details>
<summary><b>逐字稿</b></summary>

这个实验的目标是：用 v2.2 指标体系完整评估我们的 Final Baseline，验证模型是否可上线。

核心发现有三个：
1. Sample-level 抓大额事件能力强，WRecLift=34.6 倍，意思是模型识别大额打赏事件的能力是随机的 34.6 倍
2. User-level 找大哥能力中等，Lift=11.7 倍，但池子纯度高达 73%——Top 1% 用户中 73% 是真正的大哥
3. PSI=0.155，低于 0.25 阈值，分布漂移可接受，可以上线

输入是 1.4M 测试样本和 20 个 Strict 特征，输出是 38 个指标的完整评估报告。

</details>

---

## 方法 + 结果：两粒度 × 七指标 全面评估

- **算法**：Ridge Regression 预测 → 按预测值 TopK 筛选 → 多粒度评估

$$
\text{RevCap@K} = \frac{\sum_{i \in \text{Top}_K(\hat{y})} y_i}{\sum_{i=1}^{N} y_i}, \quad
\text{WRecLift@K} = \frac{\text{WhaleRecall@K}}{K}
$$

- **流程**：
```
实验流程
├── 1. 加载 Ridge 模型 (α=1.0, 20 Strict 特征)
├── 2. 预测 → y_pred = model.predict(X_test)
├── 3. 评估（核心）⭐
│   ├── L1: RevCap@1% = 51.4%
│   ├── L2: Whale 七件套 → WRecLift=34.6×, WRevCap=55.5%
│   ├── L7: 决策核心 → PSI=0.155, EffN=181
│   └── User-level → WhaleUserPrec@1%=72.6%
└── 4. 落盘 → results/metrics_v21_baseline_20260119.json
```

- **关键结果**：

| 粒度 | 指标 | 值 | 解读 |
|------|------|-----|------|
| **Sample** | WRecLift@1% | **34.6×** | 强（>20×） |
| **Sample** | WRevCap@1% | **55.5%** | 捕获 56% whale 收入 |
| **User** | WhaleUserRecLift@1% | **11.7×** | 中等（5-20×） |
| **User** | WhaleUserPrec@1% | **72.6%** | 池子纯度高 |
| **稳定性** | PSI | **0.155** | < 0.25 可上线 |

<details>
<summary><b>逐字稿</b></summary>

方法很简单：用训练好的 Ridge 模型预测，然后按预测值筛选 Top K%，在多个粒度和多个指标上评估。

关键结果分两个粒度看：

Sample-level：每个 click 是一个样本
- WRecLift=34.6 倍，意味着模型识别大额打赏事件的能力是随机的 34 倍
- WRevCap=55.5%，Top 1% 预测捕获了 56% 的大额打赏收入

User-level：每个用户是一个样本
- WhaleUserRecLift=11.7 倍，找大哥能力中等
- 但 WhaleUserPrec=72.6%，池子纯度很高——我们选出的 Top 1% 用户中，73% 确实是大哥

稳定性方面，PSI=0.155，低于 0.25 的上线阈值，分布漂移可接受。

</details>

---

## 决策：Strict Mode 可上线，User-level 需后续优化

- **结论**：「Sample-level 抓事件能力强（35×），User-level 找大哥池子纯度高（73%）但召回需优化」
    - 机制层：BaseWhaleRate 43× 差距（0.143% vs 6.21%），大额事件稀疏但用户粒度大哥比例高
    - 证据层：WRecLift=34.6×（>20× 强），WhaleUserPrec=72.6%（池子纯净）
- **决策**：✅ 采用 Strict Mode (20 特征) 作为 Final Baseline，可上线测试
    - PSI=0.155 < 0.25 阈值，分布稳定
    - RevCap@1%=51.4% > 40% 阈值，收入捕获有效
- **权衡**：
    - Δ+：51% RevCap，简单模型易维护
    - Δ-：User-level Lift 仅 11.7×，大哥识别还有空间
- **配置建议**：Whale 阈值 T=100 元（P90），K=1% 作为默认筛选比例
- **下一步**：
    - 🔴 **P0**：上线 PSI 实时监控
    - 🟡 **P1**：分配层添加 Diversity 下界约束（EffN）
    - 🟢 **P2**：实现 DW-ECE, Fairness Gap（metrics v2.3）

<details>
<summary><b>逐字稿</b></summary>

核心结论：Sample-level 抓大额事件能力强，34.6 倍提升；User-level 找大哥的池子纯度高达 73%，但召回能力还需优化。

决策很明确：采用 Strict Mode 20 特征作为 Final Baseline，可以上线测试。
理由是：PSI=0.155 低于 0.25 阈值，说明训练和测试的分布漂移可接受；RevCap@1%=51.4% 高于 40% 阈值，收入捕获有效。

权衡来看：好处是 51% 收入捕获，模型简单易维护；代价是 User-level Lift 只有 11.7 倍，大哥识别还有提升空间。

下一步三个优先级：
- P0：上线后必须有 PSI 实时监控
- P1：分配层要加 Diversity 约束，避免资源过度集中
- P2：指标体系继续完善，加上 DW-ECE 和 Fairness Gap

</details>
