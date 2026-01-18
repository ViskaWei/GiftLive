# Skill: research-design-principles

## Description
从 hub 文件中提取设计原则。触发词：design, 设计原则, 原则

## Arguments
无参数

## Workflow

### Step 1: 扫描所有 hub 文件
搜索以下位置的 `*_hub.md` 文件：
- `experiments/*/`
- `gift_allocation/`
- `KuaiLive/`

### Step 2: 检查更新时间
比较 hub 文件的最后修改时间与 `design/principles.md` 的最后同步时间

### Step 3: 提取新增设计原则
从 hub 文件的 `§6 设计原则` 章节提取：
- §6.1 已确认原则
- §6.2 待验证原则

提取格式：
| 原则 | 建议 | 适用范围 | 来源 |

### Step 4: 追加到 principles.md
更新 `design/principles.md`，追加新原则

### Step 5: 更新同步时间
记录本次同步时间戳

### Step 6: Git Commit
```bash
git add -A
git commit -m "docs: 同步设计原则"
```

## Output Format
```
📐 提取设计原则...

🔍 Step 1: 扫描 hub 文件
   找到 [N] 个 hub 文件:
   - gift_allocation_hub.md
   - kuailive_hub.md

📅 Step 2: 检查更新
   上次同步: YYYY-MM-DD
   需要更新: [N] 个文件

📝 Step 3: 提取原则
   ✅ gift_allocation_hub.md: 发现 [N] 个原则
   ✅ kuailive_hub.md: 发现 [N] 个原则

📄 Step 4: 追加到 principles.md
   新增 [N] 条原则

🕐 Step 5: 更新同步时间
   ✅ YYYY-MM-DD HH:MM

📦 Step 6: Git Commit
   ✅ 完成
```

## File Location
- 设计原则汇总: `design/principles.md`
- hub 文件: `*/*_hub.md`
