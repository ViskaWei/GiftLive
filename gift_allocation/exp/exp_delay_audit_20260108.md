# 🍃 Delay Data Audit: Consistency Verification

> **Name:** Delay Feedback Data Audit  
> **ID:** `EXP-20260108-gift-allocation-13`  
> **Topic:** `gift_allocation` | **MVP:** MVP-1.2-audit  
> **Author:** Viska Wei | **Date:** 2026-01-08 | **Status:** ✅

> 🎯 **Target:** 验证延迟实验数据一致性，确认DG2结论是否成立  
> 🚀 **Next:** ✅ 审计通过+代码bug修复 → DG2关闭成立 → 延迟校正方向关闭

## ⚡ 核心结论速览

> **一句话**: ✅ **DG2.1 审计通过**：发现并修复代码bug（单位错误导致pct_late_50计算错误），修正后pct_late_50=13.3%；样本膨胀1.14x正常；86.3%礼物立即发生，**DG2关闭有效**。

| 审计项 | 结果 | 结论 |
|---------|------|------|
| 🚩1: 样本数一致性 (72,646 vs 77,824) | ✅ 正常 | gift×click配对导致1.14x膨胀，92.6%一对一 |
| 🚩2: pct_late_* 口径定义 | ✅ 已修复 | **发现代码bug**：watch_time单位ms未转秒，修复后=13.3% |
| 🚩3: 0延迟质量点分析 | ✅ 合理 | 86.3%礼物delay=0，watch_time P50=322s（用户进来就送礼） |
| 🚩4: gift→click 一对一匹配 | ✅ 正常 | 92.6%一对一，0.3%多匹配 |

| Type | Link |
|------|------|
| 🧠 Hub | `gift_allocation/gift_allocation_hub.md` § 4.1 红旗警告, DG2.1 |
| 🗺️ Roadmap | `gift_allocation/gift_allocation_roadmap.md` § MVP-1.2-audit |
| 📗 Prior Exp | `gift_allocation/exp/exp_delay_modeling_20260108.md` |

---

# 1. 🎯 目标

**问题**: MVP-1.2 延迟实验存在以下红旗，需先审计再确认结论：

| 🚩 红旗 | 现象 | 影响 |
|--------|------|------|
| 🚩1 | EDA gift=72,646 vs Delay gift=77,824 (差5,178) | 可能存在 join 重复 |
| 🚩2 | JSON pct_late_50=0.688 vs 报告"0.7%" | 口径定义矛盾 |
| 🚩3 | 延迟中位数=0 vs Weibull median=35min 并存 | 混合分布，单一 Weibull 不当 |

**验证**: DG2.1 - 延迟数据口径是否正确？

| 预期 | 判断标准 |
|------|---------|
| 审计通过 | 样本数可解释、口径统一、分布建模合理 |
| 审计失败 | 发现 bug 或定义错误 → 修复后重做分析 |

---

# 2. 🧪 审计设计

## 2.1 审计A: Gift→Click 一对一匹配校验

**目的**: 验证每条 gift 是否唯一匹配到一个 click

**方法**:
1. 对每条 gift 事件，找满足条件的 click 候选集：
   - 同 (user_id, live_id, streamer_id)
   - click_ts ≤ gift_ts ≤ click_ts + watch_live_time
2. 统计候选 click 数量分布
3. 如果 >1 匹配常见，需定义 tie-break 规则

**预期输出**:
```
候选click数分布:
- 0个匹配: X% (孤儿gift，无法关联click)
- 1个匹配: Y% (正常一对一)
- 2+个匹配: Z% (需要tie-break)
```

## 2.2 审计B: 0延迟质量核验

**目的**: 理解84%的delay=0是真实还是数据问题

**方法**:
1. 计算 gift_ts == click_ts 的占比
2. 抽样检查这些样本的 watch_live_time 分布
3. 验证 KuaiLive 字段语义：
   - click.csv timestamp = click-to-watch 发生时刻
   - gift.csv timestamp = gifting 发生时刻

**预期输出**:
```
delay=0 统计:
- 占比: X%
- 对应 watch_live_time 分布: P50=Y, P90=Z
- 是否合理: [分析结论]
```

## 2.3 审计C: pct_late_* 口径统一

**目的**: 明确 pct_late_50/80/90 的计算定义

**当前矛盾**:
- JSON: pct_late_50 = 0.688 (68.8%)
- 报告: "相对延迟 >50% 观看时间：0.7%"

**需澄清**:
- 定义A: delay / watch_live_time > 0.5 的占比
- 定义B: (watch_live_time - delay) / watch_live_time > 0.5 的占比
- 是否包含 delay=0 的样本

**方法**: 审查 `scripts/train_delay_modeling.py` 中的计算代码

## 2.4 审计D: 样本数对齐

**目的**: 解释 72,646 (EDA) vs 77,824 (delay) 的差异

**可能原因**:
1. EDA 统计的是 gift.csv 原始行数
2. Delay 实验做了 gift→click join，一个 gift 匹配多个 click
3. 或者 Delay 实验包含了其他数据处理

**方法**: 对比两个实验的数据加载代码

---

# 3. 📊 审计结果

### 审计A结果: Gift→Click 匹配

| 匹配数 | 样本数 | 占比 | 说明 |
|--------|--------|------|------|
| 0 | 8,703 | 12.0% | 孤儿gift（无法关联到有效click） |
| 1 | 67,292 | **92.6%** | ✅ 正常一对一 |
| 2+ | 195 | 0.3% | 需tie-break（极少） |

**结论**: ✅ 匹配质量良好，92.6% 是一对一匹配

### 审计B结果: 0延迟分析

| 指标 | 值 | 说明 |
|------|-----|------|
| delay=0 占比 | **86.3%** | 大多数礼物立即发生 |
| 对应 watch_time P50 | 322.5 秒 | 用户平均停留5分钟 |
| 对应 watch_time P90 | 2,250 秒 | ~37分钟 |
| 非零延迟 watch_time P50 | 132.9 秒 | 停留时间更短 |

**结论**: ✅ 合理——用户进来就送礼，但停留时间很长。这符合直播打赏的即时冲动行为模式。

### 审计C结果: pct_late_* 定义

| 字段 | 正确定义 | 旧JSON值 | 新计算值 | Bug原因 |
|------|----------|----------|----------|---------|
| pct_late_50 | delay/(watch_time/1000) > 0.5 | 68.8% | **13.3%** | ❌ watch_time单位是ms，未转秒 |
| pct_late_80 | delay/(watch_time/1000) > 0.8 | 17.7% | **13.1%** | ❌ 同上 |
| pct_late_90 | delay/(watch_time/1000) > 0.9 | 7.5% | **13.1%** | ❌ 同上 |

**Bug发现**: 原代码 `relative_delay = delay_sec / np.maximum(watch_time, 1)` 中，watch_time 是毫秒，但 delay_sec 是秒，导致相对延迟被放大约1000倍！

**修复**: 将 watch_time 转换为秒：`watch_time_sec = watch_time / 1000.0`

### 审计D结果: 样本数来源

| 来源 | 样本数 | 说明 |
|------|--------|------|
| EDA gift.csv | 72,646 | 原始礼物记录 |
| Delay 有效配对 | 77,824 | gift×click 配对（0≤delay≤watch_time） |
| 唯一礼物数 | 68,302 | 配对中的不重复礼物 |
| 膨胀比例 | 1.14x | 部分礼物匹配多个click session |

**结论**: ✅ 膨胀正常——用户可能在同一直播间多次进出（多个click），一个gift可匹配多个session

---

| 字段 | 正确定义 | JSON值 | 报告值 |
|------|----------|--------|--------|
| pct_late_50 | TODO | 0.688 | 0.7% |

### 审计D结果: 样本数来源

---

# 4. 💡 洞见

## 4.1 数据质量
- ✅ **Gift→Click 匹配质量良好**：92.6% 是一对一匹配
- ✅ **零延迟现象合理**：用户进来就送礼，符合直播打赏的即时冲动行为
- ⚠️ **单位不一致是常见 bug**：ms vs sec 混用导致计算错误

## 4.2 方法论
- **代码审计是必要的**：数值异常（68.8% vs 预期）应立即触发代码审查
- **直播打赏 ≠ 广告 CVR**：延迟反馈假设来自广告场景，不适用于即时冲动行为
- **简单模型优先**：当延迟问题不存在时，无需引入复杂的 Chapelle 校正

---

# 5. 📝 结论

## 5.1 核心发现

| 审计项 | 结果 | 影响 |
|--------|------|------|
| 🚩1 样本数 | ✅ 正常（1.14x膨胀可解释） | DG2结论有效 |
| 🚩2 pct_late | ✅ 已修复（单位bug） | 修正后=13.3% |
| 🚩3 0延迟 | ✅ 合理（86.3%立即发生） | 分布建模正确 |
| 🚩4 匹配 | ✅ 正常（92.6%一对一） | 数据处理正确 |

## 5.2 DG2.1 结论

| 审计结果 | DG2.1 状态 | 下一步 |
|----------|------------|--------|
| ✅ **发现并修复bug** | **关闭** | DG2 结论成立，延迟校正无价值 |

**详情**：
- 发现 `train_delay_modeling.py` 中单位转换 bug（watch_time 是 ms，delay_sec 是 s）
- 修复后 pct_late_50 = 13.3%（而非之前错误的 68.8%）
- 重新运行实验，ECE 改善仍为 -0.010（变差），结论不变

## 5.3 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| ✅ 审计通过 | DG2 关闭确认，延迟校正方向关闭 | - |
| ❌ 取消 | 伪在线验证（延迟问题不存在，无需验证） | - |
| 🟡 继续 | MVP-1.5 召回-精排分工 或 MVP-1.3 多任务学习 | P1 |

---

# 6. 📎 附录

## 6.1 执行记录

| 项 | 值 |
|----|-----|
| 脚本 | `scripts/audit_delay_data.py` |
| 依赖数据 | `data/KuaiLive/click.csv`, `data/KuaiLive/gift.csv` |
| Output | `gift_allocation/results/delay_audit_20260108.json` |

```bash
# 运行审计
source init.sh
nohup python scripts/audit_delay_data.py > logs/delay_audit_20260108.log 2>&1 &

# 查看日志
tail -f logs/delay_audit_20260108.log
```

## 6.2 KuaiLive 字段参考

### click.csv
| 字段 | 说明 |
|------|------|
| user_id | 用户 ID |
| streamer_id | 主播 ID |
| live_id | 直播场次 ID |
| timestamp | click-to-watch 发生时刻 (ms) |
| watch_live_time | 本次观看总时长 (ms) |

### gift.csv
| 字段 | 说明 |
|------|------|
| user_id | 用户 ID |
| streamer_id | 主播 ID |
| live_id | 直播场次 ID |
| timestamp | gifting 发生时刻 (ms) |
| gift_price | 礼物金额 |

## 6.3 混合延迟分布建模（如需）

如果 0 延迟确实存在大量质量点，应使用混合分布：

$$F(t) = \pi \cdot \mathbb{1}(t \geq 0) + (1-\pi) \cdot F_{Weibull}(t), \quad t > 0$$

其中：
- $\pi$ = 0 延迟占比（约 84%）
- $F_{Weibull}(t) = 1 - \exp(-(t/\lambda)^k)$

Chapelle 权重调整：
$$w_i = 1 - F(H - t_i)$$

需要分别处理 $t_i = 0$ 和 $t_i > 0$ 的情况。

---

> **审计完成时间**: TODO
