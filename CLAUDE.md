# GiftLive - Claude Code 配置

## 环境初始化

运行任何代码前，必须先初始化环境：
```bash
source init.sh
```

---

## 🎯 触发词规则（必须遵守）

当用户输入以下触发词时，**必须立即读取对应的 skill 文件并按其流程执行**：

| 触发词 | Skill 文件 | 功能 |
|--------|-----------|------|
| `n [描述]` 或 `N [描述]` 或 `new [描述]` 或 `新建 [描述]` | `.claude/skills/research-new-experiment.md` | 创建新实验计划 |
| `u [exp_id]` 或 `U [exp_id]` 或 `update [exp_id]` 或 `更新 [exp_id]` | `.claude/skills/research-update.md` | 完整更新实验 |
| `u hub [topic]` | `.claude/skills/research-update.md` | 重写 Hub |
| `u [关键词]` | `.claude/skills/research-update.md` | 智能追加内容 |
| `?` 或 `？` 或 `status` 或 `进度` 或 `状态` | `.claude/skills/research-status.md` | 查看项目状态 |
| `a` 或 `A` 或 `archive` 或 `归档` | `.claude/skills/research-archive.md` | 归档实验结果 |
| `a [N]` 或 `a all` | `.claude/skills/research-archive.md` | 归档第N个或全部 |
| `next` 或 `下一步` 或 `计划` | `.claude/skills/research-next-steps.md` | 查看/管理待办 |
| `next add P0/P1 [描述]` | `.claude/skills/research-next-steps.md` | 添加任务 |
| `next done [N]` | `.claude/skills/research-next-steps.md` | 完成任务 |
| `next plan` | `.claude/skills/research-next-steps.md` | AI 智能推荐 |
| `p [描述]` 或 `P [描述]` 或 `prompt [描述]` | `.claude/skills/research-coding-prompt.md` | 生成 Coding Prompt |
| `session new [topic]` | `.claude/skills/research-session.md` | 创建新会话 |
| `session list` | `.claude/skills/research-session.md` | 列出会话 |
| `card [关键词]` 或 `卡片 [关键词]` 或 `kc [关键词]` | `.claude/skills/research-card.md` | 创建知识卡片 |
| `design` 或 `设计原则` 或 `原则` | `.claude/skills/research-design-principles.md` | 提取设计原则 |
| `merge [关键词]` 或 `合并 [关键词]` | `.claude/skills/research-merge.md` | 合并实验 |
| `grow [new_topic] [parent_topic]` 或 `生长` | `.claude/skills/research-grow-topic.md` | 子节点生长 |

**执行流程**：
1. 识别触发词 → 读取对应 skill 文件
2. 按 skill 文件中的 Workflow 步骤执行
3. 输出格式遵循 skill 文件中的 Output Format

---

## 项目结构

```
experiments/
├── [topic]/                    # 各主题实验目录
│   ├── [topic]_hub.md          # 智库导航
│   ├── [topic]_roadmap.md      # 实验追踪
│   ├── exp/                    # 子实验报告目录
│   ├── prompts/                # Coding Prompt 文件
│   └── img/                    # 图表

gift_allocation/                # 专题目录（顶层）
gift_EVpred/                    # 专题目录（顶层）
KuaiLive/                       # 专题目录（顶层）

status/
├── next_steps.md               # 下一步计划
├── archive_queue.md            # 归档队列

_backend/template/              # 文档模板
```

## 模板位置

| 模板 | 路径 |
|------|------|
| Hub | `_backend/template/hub.md` |
| Roadmap | `_backend/template/roadmap.md` |
| Exp | `_backend/template/exp.md` |
| Coding Prompt | `_backend/template/prompt_coding.md` |
| Card | `_backend/template/card.md` |
| Session | `_backend/template/session.md` |

## 文件命名规范

- **Hub**: `[topic]_hub.md`
- **Roadmap**: `[topic]_roadmap.md`
- **实验报告**: `exp_[name]_[YYYYMMDD].md`
- **Coding Prompt**: `coding_prompt_[name]_[YYYYMMDD].md`
- **图表**: `[描述性名称].png` 保存在 `img/`

## 默认作者

Viska Wei

---

## 🔴 gift_EVpred 数据处理规范（强制）

> **所有 gift_EVpred 实验必须遵守以下规则，违反将导致数据泄漏！**

### 强制使用统一数据模块

```python
# ✅ 正确做法（必须）
from gift_EVpred.data_utils import prepare_dataset, get_feature_columns

train_df, val_df, test_df, lookups = prepare_dataset(
    train_days=7, val_days=7, test_days=7
)
feature_cols = get_feature_columns(train_df)
```

### 禁止使用的特征

| 特征 | 原因 | 状态 |
|------|------|------|
| `watch_live_time` | 结果泄漏（包含打赏后时间） | 🔴 禁止 |
| `watch_time_log` | 同上 | 🔴 禁止 |
| `pair_gift_mean` (非 _past) | 未来泄漏 | 🔴 禁止 |
| `user_total_gift_7d` (非 _past) | 未来泄漏 | 🔴 禁止 |

### 数据划分规则

- **7-7-7 按天划分**：Train/Val/Test 各 7 天
- **时间顺序**：Train < Val < Test，无重叠
- **使用 Frozen 特征**：Val/Test 只查 Train 期间的统计表

### 相关文件

| 文件 | 用途 |
|------|------|
| `gift_EVpred/data_utils.py` | 统一数据处理代码 |
| `gift_EVpred/DATA_PROCESSING_GUIDE.md` | 完整数据处理指南 |
| `gift_EVpred/prompts/prompt_template_evpred.md` | Coding Prompt 模板 |

### 验证清单

所有 gift_EVpred 实验必须通过：
- [ ] 使用 `prepare_dataset()` 加载数据
- [ ] 使用 `get_feature_columns()` 获取特征
- [ ] 运行 `verify_no_leakage()` 验证通过
- [ ] 特征列不包含禁止特征
