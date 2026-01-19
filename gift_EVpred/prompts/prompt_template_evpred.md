# 🤖 Coding Prompt Template: Gift EVpred

> **适用范围**: 所有 gift_EVpred 实验
> **版本**: 1.0
> **最后更新**: 2026-01-18

---

## ⚠️ 强制规则（必须遵守）

### 数据处理规则

> **所有 gift_EVpred 实验必须使用统一的数据处理模块，禁止自行实现数据加载和特征构建！**

```python
# ✅ 正确做法：使用统一模块
from gift_EVpred.data_utils import (
    prepare_dataset,
    get_feature_columns,
    verify_no_leakage,
    run_full_verification
)
from gift_EVpred.metrics import (
    evaluate_model,
    quick_eval,
    revenue_capture_at_k,
)

# 标准数据准备（7-7-7 按天划分）
train_df, val_df, test_df, lookups = prepare_dataset(
    train_days=7, val_days=7, test_days=7
)

# 获取特征列（自动排除泄漏特征）
feature_cols = get_feature_columns(train_df)

# ❌ 禁止做法：
# - 自行读取 click.csv 并构建特征
# - 使用 watch_live_time
# - 使用 groupby().agg() 计算 pair/user/streamer 统计
# - 自行实现评估指标（必须用 metrics.py）
```

### 评估指标规则

> **所有评估必须使用 `gift_EVpred/metrics.py`，确保指标一致性！**

```python
# ✅ 正确做法：使用统一指标模块
from gift_EVpred.metrics import evaluate_model, quick_eval

# 完整评估（推荐）
result = evaluate_model(y_true, y_pred, test_df)
print(result.summary())
result.to_json('gift_EVpred/results/exp_xxx.json')

# 快速评估（训练中）
metrics = quick_eval(y_true, y_pred, whale_threshold=100)

# ❌ 禁止做法：
# - 自行计算 RevCap、Whale Recall 等指标
# - 使用 sklearn metrics 作为主指标（如 MAE、RMSE）
```

### 禁止使用的特征

| 特征 | 原因 | 替代方案 |
|------|------|---------|
| `watch_live_time` | 结果泄漏（包含打赏后时间） | 移除 |
| `pair_gift_mean` (非 _past) | 未来泄漏 | `pair_gift_mean_past` |
| `user_total_gift_7d` (非 _past) | 未来泄漏 | `user_gift_sum_past` |

---

## 📋 Coding Prompt 模板

复制以下模板，填写 `[...]` 部分：

```markdown
# 🤖 Coding Prompt: [实验名称]

> **Experiment ID:** `EXP-[YYYYMMDD]-gift_EVpred-[##]`
> **MVP:** MVP-X.X
> **Date:** YYYY-MM-DD
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：[本实验要验证什么]

**验证假设**：H[X.X] - [假设内容]

**预期结果**：
- 若 [结果A] → [结论A]
- 若 [结果B] → [结论B]

---

## 2. 🧪 实验设定

### 2.1 数据（强制使用 data_utils）

```python
# ⚠️ 必须使用此方式加载数据
from gift_EVpred.data_utils import prepare_dataset, get_feature_columns

train_df, val_df, test_df, lookups = prepare_dataset(
    train_days=7,       # 训练集天数
    val_days=7,         # 验证集天数
    test_days=7,        # 测试集天数
    gap_days=0,         # gap 天数（可选）
    label_window_hours=1  # 标签窗口（小时）
)

feature_cols = get_feature_columns(train_df)
```

**数据信息**：
```yaml
data:
  source: "KuaiLive"
  split: "7-7-7 by days"
  train_size: ~3.4M
  val_size: ~0.7M
  test_size: ~0.7M
  gift_rate: ~1.5%
  features: 40+
```

### 2.2 模型

```yaml
model:
  name: "[模型名称]"
  params:
    param1: value1
    param2: value2
```

### 2.3 训练

```yaml
training:
  seed: 42
  early_stopping: 50
  [其他参数]
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | [type] | [...] | [...] | `gift_EVpred/img/[name].png` |

**图表要求**：
- 所有文字英文
- figsize: 单张 (6,5)，多张按 6:5 扩增
- 分辨率 >= 300 dpi

---

## 4. 📁 参考代码

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| **gift_EVpred/data_utils.py** | `prepare_dataset()`, `get_feature_columns()` | ❌ 不要修改 |
| `scripts/xxx.py` | [...] | [...] |

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `gift_EVpred/exp/exp_[name]_YYYYMMDD.md`
- **模板**: `_backend/template/exp.md`

### 5.2 图表文件
- **路径**: `gift_EVpred/img/`
- **命名**: `[descriptive_name].png`

### 5.3 数值结果
- **路径**: `gift_EVpred/results/`
- **格式**: JSON

---

## 6. ⚠️ 检查清单

### 数据处理检查（必须通过）
- [ ] 使用 `prepare_dataset()` 加载数据
- [ ] 使用 `get_feature_columns()` 获取特征
- [ ] 特征列不包含 `watch_live_time`
- [ ] 所有 gift 相关特征带 `_past` 后缀
- [ ] 运行 `verify_no_leakage()` 验证通过

### 指标评估检查（必须通过）
- [ ] 使用 `evaluate_model()` 进行完整评估
- [ ] 主指标为 `RevCap@1%`（不是 MAE/RMSE）
- [ ] 结果保存到 `gift_EVpred/results/` 目录
- [ ] 调用 `result.summary()` 输出完整报告

### 代码检查
- [ ] seed=42 固定随机性
- [ ] 图表文字全英文
- [ ] 保存日志到 `logs/`

---

## 7. 📤 报告抄送

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_EVpred_roadmap.md` | MVP 状态 + 结论快照 | §2.1, §4.3 |
| `gift_EVpred_hub.md` | 假设验证状态 + 洞见 | §1, §4 |

---

<!--
📌 Agent 执行规则：

1. ⚠️ 必须使用 gift_EVpred/data_utils.py 加载数据
2. ⚠️ 必须使用 gift_EVpred/metrics.py 评估模型
3. ❌ 禁止自行实现数据处理逻辑
4. ❌ 禁止自行实现评估指标
5. ❌ 禁止使用 watch_live_time
6. ✅ 先验证数据无泄漏再训练模型
7. ✅ 使用 evaluate_model() 输出完整评估
8. ✅ 按模板输出 exp.md 报告
-->
```

---

## 📦 data_utils.py 使用示例

### 基本用法

```python
#!/usr/bin/env python3
"""
Example: Using data_utils for gift EVpred experiment
"""
import sys
sys.path.insert(0, '/home/swei20/GiftLive')

from gift_EVpred.data_utils import (
    prepare_dataset,
    get_feature_columns,
    run_full_verification,
    load_raw_data
)
import lightgbm as lgb

# 1. 准备数据（7-7-7 划分，无泄漏）
train_df, val_df, test_df, lookups = prepare_dataset()

# 2. 获取特征列
feature_cols = get_feature_columns(train_df)
print(f"Features: {len(feature_cols)}")

# 3. 验证无泄漏（推荐）
gift, _, _, _, _ = load_raw_data()
run_full_verification(train_df, val_df, test_df, gift, feature_cols)

# 4. 准备训练数据
X_train = train_df[feature_cols]
y_train = train_df['target']  # log(1+gift)
X_val = val_df[feature_cols]
y_val = val_df['target']

# 5. 训练模型
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'seed': 42,
    'verbose': -1
}

model = lgb.train(
    params, train_data,
    num_boost_round=500,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50)]
)

# 6. 评估（必须使用 metrics 模块）
from gift_EVpred.metrics import evaluate_model

X_test = test_df[feature_cols]
y_pred = model.predict(X_test)
y_test = test_df['gift_amount']  # 原始金额（非 log）

# 完整评估
result = evaluate_model(y_test, y_pred, test_df)
print(result.summary())

# 保存结果
result.to_json('gift_EVpred/results/exp_xxx.json')
```

### 可用函数列表

#### data_utils.py（数据处理）

| 函数 | 用途 |
|------|------|
| `prepare_dataset()` | 准备完整数据集（主函数） |
| `get_feature_columns(df)` | 获取特征列（排除泄漏特征） |
| `verify_no_leakage(df, gift)` | 验证特征无泄漏 |
| `run_full_verification(...)` | 运行完整验证 |
| `load_raw_data()` | 加载原始数据 |
| `split_by_days(df, ...)` | 按天划分数据 |
| `create_frozen_lookups(gift, ts)` | 创建冻结特征查找表 |
| `apply_frozen_features(df, lookups)` | 应用冻结特征 |

#### metrics.py（指标评估）

| 函数 | 用途 |
|------|------|
| `evaluate_model(y_true, y_pred, test_df)` | 完整模型评估（**推荐**） |
| `quick_eval(y_true, y_pred)` | 快速评估（训练中使用） |
| `revenue_capture_at_k(y_true, y_pred, k)` | RevCap@K 单指标 |
| `whale_recall_at_k(...)` | Whale Recall@K |
| `whale_precision_at_k(...)` | Whale Precision@K |
| `compute_revcap_curve(...)` | 多 K 值 RevCap 曲线 |
| `compute_stability_by_day(...)` | 按天稳定性评估 |
| `EvalResult` | 结果类（支持 .summary()、.to_json()） |

---

## 🚫 常见错误

### 错误 1: 自行读取数据

```python
# ❌ 错误
click = pd.read_csv('data/KuaiLive/click.csv')
gift = pd.read_csv('data/KuaiLive/gift.csv')
# 然后自己做特征...

# ✅ 正确
from gift_EVpred.data_utils import prepare_dataset
train_df, val_df, test_df, lookups = prepare_dataset()
```

### 错误 2: 使用泄漏特征

```python
# ❌ 错误
features = ['watch_live_time', 'pair_gift_mean', ...]

# ✅ 正确
from gift_EVpred.data_utils import get_feature_columns
features = get_feature_columns(train_df)  # 自动排除泄漏特征
```

### 错误 3: 自行计算聚合特征

```python
# ❌ 错误 (会导致泄漏)
pair_stats = gift.groupby(['user_id', 'streamer_id']).agg(...)
df = df.merge(pair_stats, ...)

# ✅ 正确 (使用 data_utils 提供的 frozen 特征)
# 特征已经在 prepare_dataset() 中计算好了
```

### 错误 4: 自行实现评估指标

```python
# ❌ 错误（自己算 RevCap）
def my_revcap(y_true, y_pred, k=0.01):
    top_k = int(len(y_true) * k)
    idx = np.argsort(y_pred)[-top_k:]
    return y_true[idx].sum() / y_true.sum()

# ❌ 错误（用 sklearn 指标作为主指标）
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)  # 不是业务指标

# ✅ 正确
from gift_EVpred.metrics import evaluate_model
result = evaluate_model(y_test, y_pred, test_df)
print(f"RevCap@1%: {result.revcap_1pct:.1%}")
```

---

> **记住**: 使用 `data_utils.py` 和 `metrics.py` 是强制要求，不是建议！
