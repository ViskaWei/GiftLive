# 🎁 GiftLive

> **实验管理系统** - 基于 Physics_Informed_AI 精简版

---

## 📁 项目结构

```
GiftLive/
├── .cursorrules           # Cursor AI 规则
├── COMMANDS.md            # 命令速查手册
├── README.md              # 本文件
│
├── experiments/           # 📗 实验文档
│   └── [topic]/           # 按主题分类
│       ├── [topic]_hub.md         # 智库导航
│       ├── [topic]_roadmap.md     # 实验追踪
│       ├── exp_*.md               # 实验报告
│       ├── card/                  # 知识卡片
│       ├── sessions/              # GPT 会话
│       ├── prompts/               # Coding Prompt
│       └── img/                   # 图表
│
├── status/                # 📋 状态追踪
│   ├── next_steps.md      # 下一步计划
│   ├── archive_queue.md   # 归档队列
│   └── kanban.md          # 看板
│
├── design/                # 📐 设计原则
│   └── principles.md      # 原则汇总
│
├── reports/               # 📊 报告
│   └── drafts/            # 报告草稿
│
├── scripts/               # 🔧 脚本
├── configs/               # ⚙️ 配置文件
├── data/                  # 📦 数据
├── logs/                  # 📝 日志
├── results/               # 📈 结果
│
└── _backend/              # 🔒 后端
    └── template/          # 模板文件
        ├── exp.md         # 实验报告模板
        ├── hub.md         # 智库导航模板
        ├── roadmap.md     # 实验追踪模板
        ├── card.md        # 知识卡片模板
        ├── session.md     # 会话模板
        └── prompt_coding.md # Coding Prompt 模板
```

---

## 🚀 快速开始

### 1. 查看进度
```
?
```

### 2. 新建实验
```
n [实验描述]
```

### 3. 归档结果
```
a
```

### 4. 更新文档
```
u [experiment_id]
```

---

## 📖 命令速查

| 命令 | 作用 |
|------|------|
| `?` | 查看进度状态 |
| `n` | 新建实验计划 |
| `a` | 归档实验结果 |
| `u` | 更新文档 |
| `p` | 生成 Coding Prompt |
| `card` | 创建知识卡片 |
| `design` | 提取设计原则 |
| `next` | 管理待办 |
| `merge` | 合并实验 |
| `session` | GPT 会话归档 |

详细说明请查看 [COMMANDS.md](./COMMANDS.md)

---

## 🔄 工作流

```
规划实验 (n)
    ↓
执行实验 (python scripts/xxx.py)
    ↓
归档结果 (a)
    ↓
更新文档 (u)
    ↓
提炼知识 (card)
    ↓
规划下一步 (next)
```

---

## 📝 文档层级

| 层级 | 文件 | 职责 |
|------|------|------|
| 1 | session.md | GPT 脑暴对话 |
| 2a | hub.md | 问题树、假设、战略导航 |
| 2b | roadmap.md | MVP 列表、进度追踪 |
| 3 | exp.md | 单实验详细记录 |
| 4 | card.md | 跨实验知识卡片 |
| 5 | next_steps.md | 日常待办 |

---

## 🏷️ Experiment ID 格式

```
EXP-YYYYMMDD-topic-XX

示例：EXP-20250108-model-01
```

---

## 📚 模板位置

所有模板文件位于 `_backend/template/`

| 模板 | 用途 |
|------|------|
| `exp.md` | 实验报告 |
| `hub.md` | 智库导航 |
| `roadmap.md` | 实验追踪 |
| `card.md` | 知识卡片 |
| `session.md` | GPT 会话 |
| `prompt_coding.md` | Coding Prompt |

---

## 👤 作者

Viska Wei

---

## 📅 创建日期

2025-01-08
