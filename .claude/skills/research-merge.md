# Skill: research-merge

## Description
将多个相似实验合并为综合报告。触发词：merge, 合并, 整合

## Arguments
- `[topic/关键词]` - 要合并的实验主题或关键词

## Workflow

### Step 1: 解析描述
提取关键词，确定要合并的实验范围

### Step 2: 扫描目录
在 `[topic]/exp/` 中搜索匹配的实验文件

### Step 3: 提取关键信息
从每个实验中提取：
- 核心结论
- 关键数字
- 实验配置
- 洞见

### Step 4: 生成综合报告
使用模板 `_backend/template/consolidated.md`

### Step 5: 保存
输出到: `[topic]/exp_[topic]_consolidated_[YYYYMMDD].md`

### Step 6: Git Commit
```bash
git add -A
git commit -m "docs: 合并 [topic] 相关实验"
```

## Output Format
```
🔀 合并实验...

🔍 Step 1: 解析描述
   关键词: [keyword]
   Topic: [topic]

📁 Step 2: 扫描目录
   找到 [N] 个相关实验:
   - exp_xxx.md
   - exp_yyy.md
   - exp_zzz.md

📝 Step 3: 提取关键信息
   - 结论: [N] 条
   - 数字: [N] 组
   - 配置: [N] 种

📄 Step 4: 生成综合报告
   ✅ 已生成报告

💾 Step 5: 保存
   ✅ [topic]/exp_[topic]_consolidated_[YYYYMMDD].md

📦 Step 6: Git Commit
   ✅ 完成
```

## Template Reference
- `_backend/template/consolidated.md`

## Output Location
- `experiments/[topic]/exp_[topic]_consolidated_[YYYYMMDD].md`
