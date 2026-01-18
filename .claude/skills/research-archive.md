# Skill: research-archive

## Description
将原始实验结果归档为结构化 exp.md 报告。触发词：a, A, 归档, archive

## Variants
- `a` - 显示归档队列，选择要归档的项目
- `a 1` - 归档第 N 个
- `a all` - 全部归档

## Arguments
- `[N]` - 归档队列中的序号
- `all` - 归档全部

## Workflow

### Step 1: 检查归档队列
读取 `status/archive_queue.md`，显示待归档项目

### Step 2: 读取原始文件
从队列中选择的原始文件读取内容

### Step 3: 撰写实验报告
按 `_backend/template/exp.md` 模板重写：
- **设计原则**: 前 300 行是核心信息
- 填写所有必需章节

### 关键提取项
| 原始内容 | 目标章节 |
|---------|----------|
| 核心结论一句话 | §核心结论速览 |
| 假设验证结果 | §核心结论速览 |
| 最佳配置/参数 | §关键数字速查 |
| 完整数值结果 | §数值结果表 |
| 实验设计细节 | §实验设计 |
| 关键图表说明 | §实验图表 |
| 设计建议/启示 | §设计启示 |

### Step 4: 更新 roadmap.md 和 hub.md
同步关键信息到上游文档

### Step 5: 归档后检查
信息完整性检查清单：
- [ ] ⚡ 核心结论速览完整
- [ ] §1 目标清晰
- [ ] §3 实验设计完整
- [ ] §4 图表有描述
- [ ] §6 结论有数据支撑
- [ ] 上游链接正确

### Step 6: 移动原始文件
将原始文件移动到 `processed/` 目录

### Step 7: Git Commit
```bash
git add -A
git commit -m "docs: 归档实验 [exp_name]"
```

## Output Format
```
📦 归档实验结果...

📋 Step 1: 归档队列 ([N]个)
1. [raw_file1] → [target1]
2. [raw_file2] → [target2]
选择归档: [N]

📖 Step 2: 读取原始文件
   ✅ 已读取: [raw_file]

📝 Step 3: 撰写实验报告
   ✅ 已创建: [path_to_exp.md]
   - ⚡ 核心结论速览 ✅
   - §1 目标 ✅
   - §3 实验设计 ✅
   - §4 图表 ✅
   - §6 结论 ✅

📤 Step 4: 更新上游文档
   ✅ roadmap.md: §2.1, §4.3
   ✅ hub.md: §1 假设树

✅ Step 5: 完整性检查
   全部通过

📁 Step 6: 移动原始文件
   ✅ [raw_file] → processed/

📦 Step 7: Git Commit
   ✅ 完成

✅ 归档完成！
```

## Template Reference
- exp.md: `_backend/template/exp.md`

## File Locations
- 归档队列: `status/archive_queue.md`
- 原始文件: 根据队列指定
- 已处理: `processed/`
