# Skill: research-status

## Description
查看项目状态，包括待办任务、归档队列和最近更新的文档。触发词：?, ？, status, 进度, 状态

## Arguments
无参数

## Workflow

### Step 1: 读取待办任务
从 `status/next_steps.md` 读取 P0/P1 任务

### Step 2: 读取归档队列
从 `status/archive_queue.md` 读取待归档项目

### Step 3: 扫描最近更新
扫描以下目录，按修改时间排序：
- `experiments/*/exp/*.md`
- `gift_allocation/exp/*.md`
- `KuaiLive/exp/*.md`

显示最近 5 个更新的文件

### Step 4: 自动 Git Commit + Push（可选）
如果有未提交的更改：
```bash
git add -A
git commit -m "chore: auto-save progress"
git push
```

## Output Format
```
📊 项目状态

📋 待办任务:
🔴 P0: [任务1]
🔴 P0: [任务2]
🟡 P1: [任务3]

📦 归档队列 ([N]个):
1. [raw_file] → [target_dir]
2. [raw_file] → [target_dir]

📝 最近更新:
- [file1.md] ([Xh ago])
- [file2.md] ([Xh ago])
- [file3.md] ([Xd ago])

📦 Git 状态:
✅ 已同步 / ⚠️ 有 [N] 个未提交更改
```

## File Locations
- 待办清单: `status/next_steps.md`
- 归档队列: `status/archive_queue.md`
- 实验目录: `experiments/`, `gift_allocation/`, `KuaiLive/`
