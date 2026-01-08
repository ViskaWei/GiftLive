# Coding Prompt: MVP-1.2-audit Delay Data Audit

## 实验元数据
| 项 | 值 |
|---|---|
| MVP | MVP-1.2-audit |
| exp_id | EXP-20260108-gift-allocation-13 |
| 关闭 Gate | DG2.1 |
| 优先级 | 🔴 P0 |
| 预计时间 | 5-10 分钟 |

## 1. 审计目标

验证 MVP-1.2 延迟实验的数据一致性，确认以下红旗：

| 🚩 | 问题 | 现象 |
|---|------|------|
| 🚩1 | 样本数不一致 | EDA gift=72,646 vs Delay gift=77,824 (差5,178) |
| 🚩2 | pct_late_* 口径矛盾 | JSON pct_late_50=0.688 vs 报告"0.7%" |
| 🚩3 | 延迟分布异常 | 中位数=0 vs Weibull median=35min 并存 |

## 2. 审计设计

### 2.1 审计A: Gift→Click 一对一匹配校验

**目的**: 验证每条 gift 是否唯一匹配到一个 click

**逻辑**:
```
对每条 gift 事件 (user_id, live_id, streamer_id, gift_ts):
  找满足条件的 click 候选集:
    - 同 (user_id, live_id, streamer_id)
    - click_ts <= gift_ts <= click_ts + watch_live_time
  统计候选数量
```

**输出**: 匹配数分布 (0个/1个/2+个)

### 2.2 审计B: 0延迟质量核验

**目的**: 理解84%的delay=0是真实还是数据问题

**逻辑**:
```
统计 gift_ts == click_ts 的占比
抽样 delay=0 的样本，检查 watch_live_time 分布
判断是否合理（如果 watch_time 很长但 delay=0，说明用户一进来就送礼）
```

### 2.3 审计C: pct_late_* 口径统一

**目的**: 澄清 pct_late_50 的计算定义

**需检查的代码**: `scripts/train_delay_modeling.py`

**可能的定义**:
- A: delay / watch_live_time > 0.5 的占比
- B: (watch_live_time - delay) / watch_live_time > 0.5 的占比
- 是否排除 delay=0

### 2.4 审计D: 样本数来源追溯

**目的**: 解释 72,646 vs 77,824 的差异

**检查点**:
1. EDA 用的是 gift.csv 原始行数？
2. Delay 实验做了 gift→click join，可能一对多？
3. 是否有去重/过滤逻辑不同？

## 3. 参考代码路径

| 文件 | 用途 |
|------|------|
| `scripts/train_delay_modeling.py` | 延迟实验主代码，需审查 pct_late_* 计算 |
| `scripts/eda_kuailive.py` | EDA 代码，需确认 gift 统计逻辑 |
| `gift_allocation/results/delay_modeling_20260108.json` | 需验证的结果 |
| `gift_allocation/results/eda_stats_20260108.json` | EDA 结果 |

## 4. 输出要求

### 4.1 审计结果 JSON
保存到: `gift_allocation/results/delay_audit_20260108.json`

```json
{
  "experiment_id": "EXP-20260108-gift-allocation-13",
  "mvp": "MVP-1.2-audit",
  "timestamp": "...",
  
  "audit_a_matching": {
    "total_gifts": 72646,
    "match_0": {"count": ..., "pct": ...},
    "match_1": {"count": ..., "pct": ...},
    "match_2plus": {"count": ..., "pct": ...},
    "conclusion": "..."
  },
  
  "audit_b_zero_delay": {
    "delay_eq_zero_count": ...,
    "delay_eq_zero_pct": ...,
    "zero_delay_watch_time_p50": ...,
    "zero_delay_watch_time_p90": ...,
    "is_reasonable": true/false,
    "explanation": "..."
  },
  
  "audit_c_pct_late_definition": {
    "code_definition": "delay / watch_time > threshold",
    "pct_late_50_computed": ...,
    "pct_late_50_json": 0.688,
    "pct_late_report": "0.7%",
    "discrepancy_reason": "...",
    "correct_value": ...
  },
  
  "audit_d_sample_count": {
    "eda_gift_count": 72646,
    "delay_sample_count": 77824,
    "difference": 5178,
    "reason": "join duplication / filtering difference / ...",
    "is_bug": true/false
  },
  
  "overall_verdict": {
    "all_passed": true/false,
    "failed_audits": [],
    "dg2_conclusion_valid": true/false,
    "next_step": "..."
  }
}
```

### 4.2 实验报告
更新: `gift_allocation/exp/exp_delay_audit_20260108.md`
- 填写 §3 审计结果
- 填写 §4 洞见
- 填写 §5 结论

## 5. 决策规则

| 审计结果 | DG2.1 状态 | DG2 影响 | 下一步 |
|----------|------------|----------|--------|
| ✅ 全部通过 | 关闭 | DG2 结论成立 | → MVP-1.2-pseudo |
| ❌ 样本数有 bug | 待定 | 重做延迟分析 | 修复后重跑 |
| ❌ pct_late 定义错 | 待定 | 修正后判断 | 更新报告 |
| ❌ 匹配有问题 | 待定 | 重做延迟计算 | 修复 join 逻辑 |

## 6. 运行命令

```bash
source init.sh
nohup python scripts/audit_delay_data.py > logs/delay_audit_20260108.log 2>&1 &
echo $! > logs/delay_audit_20260108.pid

# 查看日志
tail -f logs/delay_audit_20260108.log
```

## 7. 同步更新

审计完成后，根据结论更新：
- `gift_allocation/gift_allocation_hub.md` § 4.1 红旗状态, DG2.1
- `gift_allocation/gift_allocation_roadmap.md` § MVP-1.2-audit 状态
- 如果审计失败，需更新 MVP-1.2 状态
