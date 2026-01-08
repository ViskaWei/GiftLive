# 🤖 Coding Prompt: KuaiLive 数据探索

> **Experiment ID:** `EXP-20260108-gift-allocation-01`  
> **MVP:** MVP-0.1  
> **Date:** 2026-01-08  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：对 KuaiLive 数据集进行全面探索，理解打赏行为的分布特征，为后续建模提供数据驱动依据。

**验证假设**：
- H1.1: 打赏行为极稀疏（打赏率 < 1%）
- H1.2: 打赏金额呈重尾分布（Pareto 型）
- H1.3: 打赏存在时间延迟特性

**预期结果**：
- 产出关键数字：打赏率、P50/P90/P99 金额、用户/主播 Gini 系数
- 若打赏率 <1% 且金额 Gini >0.8 → 确认两段式建模必要性
- 若延迟分布集中在观看后期 → 确认延迟建模必要性

---

## 2. 🧪 实验设定

### 2.1 数据

```yaml
data:
  source: "KuaiLive"
  path: "data/KuaiLive/"
  files:
    - gift.csv       # 核心：打赏记录
    - click.csv      # 点击行为
    - comment.csv    # 评论行为
    - like.csv       # 点赞行为
    - room.csv       # 直播间信息
    - streamer.csv   # 主播信息
    - user.csv       # 用户信息
    - negative.csv   # 负反馈
```

### 2.2 分析维度

```yaml
analysis:
  # 1. 打赏基础统计
  gift_basics:
    - total_records
    - unique_users
    - unique_streamers
    - unique_rooms
    - gift_rate         # 打赏用户占比
  
  # 2. 金额分布
  amount_distribution:
    - mean, std
    - percentiles: [P10, P25, P50, P75, P90, P95, P99]
    - max, min
    - log_transform_stats  # log(1+Y) 分布
  
  # 3. 用户维度分析
  user_analysis:
    - gifts_per_user_distribution
    - amount_per_user_distribution
    - user_gini_coefficient  # 用户贡献集中度
    - top_k_user_contribution: [1%, 5%, 10%]
  
  # 4. 主播维度分析
  streamer_analysis:
    - gifts_per_streamer_distribution
    - amount_per_streamer_distribution
    - streamer_gini_coefficient
    - top_k_streamer_share: [1%, 5%, 10%]
  
  # 5. 时间模式
  temporal_analysis:
    - hourly_pattern
    - daily_pattern
    - gift_timing_in_session  # 打赏发生在观看的什么时间点
  
  # 6. 稀疏性分析
  sparsity:
    - user_item_matrix_density
    - cold_start_ratio_user
    - cold_start_ratio_streamer
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | histogram | Gift Amount (log scale) | Count | `experiments/gift_allocation/img/gift_amount_distribution.png` |
| Fig2 | histogram | Gift Amount | Count | `experiments/gift_allocation/img/gift_amount_distribution_raw.png` |
| Fig3 | bar | Percentile | Amount | `experiments/gift_allocation/img/gift_amount_percentiles.png` |
| Fig4 | line | User Rank (sorted by amount) | Cumulative Share (%) | `experiments/gift_allocation/img/user_lorenz_curve.png` |
| Fig5 | line | Streamer Rank (sorted by amount) | Cumulative Share (%) | `experiments/gift_allocation/img/streamer_lorenz_curve.png` |
| Fig6 | histogram | Gifts per User | Count | `experiments/gift_allocation/img/gifts_per_user_distribution.png` |
| Fig7 | histogram | Gifts per Streamer | Count | `experiments/gift_allocation/img/gifts_per_streamer_distribution.png` |
| Fig8 | bar | Hour | Gift Count / Amount | `experiments/gift_allocation/img/hourly_pattern.png` |
| Fig9 | heatmap | User | Streamer | `experiments/gift_allocation/img/user_streamer_interaction_matrix.png` |

**图表要求**：
- 所有文字必须英文
- 包含 legend、title、axis labels
- 分辨率 ≥ 300 dpi
- Lorenz 曲线需含 Gini 系数标注

---

## 4. 📁 参考代码

> ⚠️ **不要在这里写代码！只写路径，让 Agent 自己读取**

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `_backend/scripts/training/post_process.py` | 报告生成模式 | 适配 EDA 输出 |

**建议使用的 Python 库**：
- `pandas` - 数据处理
- `numpy` - 数值计算
- `matplotlib`, `seaborn` - 可视化
- `scipy.stats` - 统计分布拟合

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `experiments/gift_allocation/exp_kuailive_eda_20260108.md`
- **模板**: `_backend/template/exp.md`
- **必须包含**:
  - ⚡ 核心结论速览（打赏率、Gini 系数、关键分布特征）
  - 📊 实验图表（所有图 + 观察）
  - 📝 结论（对后续建模的启示）

### 5.2 图表文件
- **路径**: `experiments/gift_allocation/img/`
- **命名**: 如上表所述

### 5.3 数值结果
- **格式**: JSON
- **路径**: `experiments/gift_allocation/results/eda_stats_20260108.json`
- **内容**:
```json
{
  "gift_rate": 0.XXX,
  "amount_stats": {
    "mean": X, "std": X, "median": X,
    "p90": X, "p95": X, "p99": X, "max": X
  },
  "gini": {
    "user": 0.XXX,
    "streamer": 0.XXX
  },
  "top_concentration": {
    "top_1pct_users_share": 0.XXX,
    "top_5pct_users_share": 0.XXX,
    "top_1pct_streamers_share": 0.XXX,
    "top_5pct_streamers_share": 0.XXX
  },
  "sparsity": {
    "matrix_density": 0.XXXX,
    "cold_start_user_ratio": 0.XXX,
    "cold_start_streamer_ratio": 0.XXX
  }
}
```

---

## 6. 📤 报告抄送

完成后需同步更新：

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_allocation_roadmap.md` | MVP-0.1 状态 ✅ + 结论快照 | §2.1, §4.3 |
| `gift_allocation_hub.md` | 关键数字（打赏率、Gini、Top-K%贡献率） | §6.3 关键数字速查 |

---

## 7. ⚠️ 注意事项

- [ ] 代码中添加 seed 固定随机性（seed=42）
- [ ] 图表文字全英文
- [ ] 处理缺失值和异常值（记录处理方法）
- [ ] 检查 gift.csv 的字段含义（需先查看数据 schema）
- [ ] 注意数据量级，大数据集用采样预览
- [ ] 保存完整日志到 `logs/`

---

## 8. 📋 关键分析问题清单

完成 EDA 后需回答以下问题：

1. **打赏率是多少？** → 决定是否需要不均衡处理
2. **金额分布是什么形状？** → 决定是否需要 log 变换或分位数回归
3. **用户/主播 Gini 系数是多少？** → 理解金主分布的集中度
4. **Top-1% 用户贡献了多少收益？** → 评估金主重要性
5. **是否存在冷启动问题？** → 新用户/新主播占比
6. **打赏有时间规律吗？** → 时间特征是否重要

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先检查 data/KuaiLive/ 目录下的数据文件
3. ✅ 理解每个 CSV 文件的 schema（字段名、类型）
4. ✅ 按模板输出 exp.md 报告
5. ✅ 所有图表保存到 img/ 目录
6. ✅ 数值结果保存为 JSON
-->
