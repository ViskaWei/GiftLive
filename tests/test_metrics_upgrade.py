#!/usr/bin/env python3
"""
Unit Tests for Metrics Module v2.0 Upgrade

测试新增的决策相关指标：
1. compute_calibration() - ECE + Reliability Curve
2. gini_coefficient() - Gini 系数
3. compute_slice_metrics() - 切片指标
4. compute_ecosystem_metrics() - 生态指标

Author: Viska Wei
Date: 2026-01-19
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gift_EVpred.metrics import (
    gini_coefficient,
    compute_calibration,
    compute_slice_metrics,
    compute_ecosystem_metrics,
    evaluate_model,
    EvalResult,
    revenue_capture_at_k,
)


# =============================================================================
# Test: gini_coefficient()
# =============================================================================

class TestGiniCoefficient:
    """Gini 系数测试"""

    def test_equal_distribution(self):
        """全等分布 => Gini = 0"""
        x = np.array([100, 100, 100, 100, 100])
        assert gini_coefficient(x) == pytest.approx(0.0, abs=1e-6)

    def test_uniform_random(self):
        """均匀分布的 Gini 应该较低"""
        np.random.seed(42)
        x = np.random.uniform(90, 110, 1000)  # 几乎均匀
        gini = gini_coefficient(x)
        assert 0 <= gini <= 0.1  # 非常低的 Gini

    def test_extreme_concentration(self):
        """极端集中 => Gini 接近 1"""
        x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1000])
        gini = gini_coefficient(x)
        assert gini > 0.8  # 接近 1

    def test_moderate_inequality(self):
        """中等不平等"""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        gini = gini_coefficient(x)
        assert 0.3 < gini < 0.9

    def test_empty_array(self):
        """空数组返回 NaN"""
        assert np.isnan(gini_coefficient(np.array([])))

    def test_all_zeros(self):
        """全零返回 NaN"""
        assert np.isnan(gini_coefficient(np.array([0, 0, 0, 0])))

    def test_negative_values_clipped(self):
        """负值应被 clip 到 0"""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x = np.array([-10, 0, 10, 20, 30])
            gini = gini_coefficient(x)
            # 应该有 warning
            assert len(w) == 1
            assert "negative" in str(w[0].message).lower()
            # 值应该有效
            assert 0 <= gini <= 1

    def test_single_value(self):
        """单值数组 => Gini = 0"""
        assert gini_coefficient(np.array([100])) == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# Test: compute_calibration()
# =============================================================================

class TestComputeCalibration:
    """概率校准测试"""

    def test_perfect_calibration(self):
        """完美校准 => ECE = 0"""
        # y_prob 与 y_true 完全一致
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])

        result = compute_calibration(y_true, y_prob, n_bins=2)
        assert result['ece'] == pytest.approx(0.0, abs=0.01)

    def test_severely_miscalibrated(self):
        """严重失配 => ECE 接近 |actual - predicted|"""
        # 预测全是 0.9，但实际正例率只有 10%
        np.random.seed(42)
        n = 1000
        y_true = np.zeros(n)
        y_true[:100] = 1  # 10% positive
        y_prob = np.ones(n) * 0.9

        result = compute_calibration(y_true, y_prob, n_bins=10, strategy='uniform')
        # ECE 应该接近 |0.9 - 0.1| = 0.8
        assert result['ece'] > 0.7

    def test_binary_conversion(self):
        """金额自动转为二分类"""
        y_true = np.array([0, 0, 50, 100, 0, 200])  # 金额
        y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.1, 0.9])

        result = compute_calibration(y_true, y_prob)
        assert 'ece' in result
        assert result['meta']['positive_rate'] == pytest.approx(0.5, abs=0.01)
        assert "converted to binary" in str(result['meta']['warnings'])

    def test_prob_clipping(self):
        """超出 [0,1] 的概率被 clip"""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([-0.1, 0.5, 0.5, 1.2])

        result = compute_calibration(y_true, y_prob)
        assert "clipped" in str(result['meta']['warnings'])

    def test_all_same_class(self):
        """全同类 => ECE = NaN"""
        y_true = np.zeros(100)  # 全负
        y_prob = np.random.uniform(0, 1, 100)

        result = compute_calibration(y_true, y_prob)
        assert np.isnan(result['ece'])
        assert "same class" in str(result['meta']['warnings'])

    def test_quantile_strategy(self):
        """quantile 分桶策略"""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_prob = np.random.uniform(0, 1, 1000)

        result = compute_calibration(y_true, y_prob, n_bins=5, strategy='quantile')
        assert 'ece' in result
        assert result['meta']['strategy'] == 'quantile'

    def test_bins_structure(self):
        """bins 返回结构正确"""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_prob = np.random.uniform(0, 1, 1000)

        result = compute_calibration(y_true, y_prob, n_bins=5)
        assert len(result['bins']) > 0

        for b in result['bins']:
            assert 'bin_lower' in b
            assert 'bin_upper' in b
            assert 'n' in b
            assert 'avg_pred' in b
            assert 'avg_true' in b
            assert 'gap' in b


# =============================================================================
# Test: compute_slice_metrics()
# =============================================================================

class TestComputeSliceMetrics:
    """切片指标测试"""

    @pytest.fixture
    def sample_data(self):
        """构造测试数据"""
        np.random.seed(42)
        n = 5000

        df = pd.DataFrame({
            'user_id': np.random.randint(1, 100, n),
            'streamer_id': np.random.randint(1, 50, n),
            'pair_gift_count': np.random.choice([0, 1, 2, 3, 5, 10], n, p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1]),
            'streamer_gift_count': np.random.choice([0, 1, 5, 10, 20], n, p=[0.2, 0.2, 0.2, 0.2, 0.2]),
            'user_gift_sum': np.random.exponential(100, n),
            'streamer_gift_sum': np.random.exponential(500, n),
        })

        # 真实标签：大部分是 0，少部分有打赏
        y_true = np.zeros(n)
        gifter_mask = np.random.random(n) < 0.02
        y_true[gifter_mask] = np.random.exponential(50, gifter_mask.sum())

        # 预测分数
        y_pred = np.random.uniform(0, 1, n)

        return y_true, y_pred, df

    def test_basic_slices_exist(self, sample_data):
        """基本切片应该存在"""
        y_true, y_pred, df = sample_data
        result = compute_slice_metrics(
            y_true, y_pred, df,
            whale_threshold=100,
            min_slice_n=100,
        )

        # 应该有一些切片结果
        slices = [k for k in result.keys() if k != 'skipped']
        assert len(slices) > 0

        # 检查 skipped 字典
        assert 'skipped' in result

    def test_cold_start_pair(self, sample_data):
        """冷启动 pair 切片"""
        y_true, y_pred, df = sample_data
        result = compute_slice_metrics(
            y_true, y_pred, df,
            whale_threshold=100,
            min_slice_n=100,
        )

        if 'cold_start_pair' in result:
            cs = result['cold_start_pair']
            assert 'n' in cs
            assert 'revcap_curve' in cs
            assert cs['n'] > 0

    def test_whale_slices(self, sample_data):
        """whale 切片"""
        y_true, y_pred, df = sample_data
        # 增加一些 whale
        y_true[y_true > 0] *= 10  # 放大打赏金额

        result = compute_slice_metrics(
            y_true, y_pred, df,
            whale_threshold=100,
            min_slice_n=10,  # 降低门槛
        )

        # non_whale 应该存在
        assert 'non_whale_true' in result or 'non_whale_true' in result.get('skipped', {})

    def test_missing_columns_graceful(self):
        """缺少列时优雅降级"""
        np.random.seed(42)
        n = 1000

        # 只有基础列
        df = pd.DataFrame({
            'user_id': np.random.randint(1, 100, n),
            'streamer_id': np.random.randint(1, 50, n),
        })

        y_true = np.random.exponential(10, n)
        y_pred = np.random.uniform(0, 1, n)

        result = compute_slice_metrics(
            y_true, y_pred, df,
            whale_threshold=50,
            min_slice_n=100,
        )

        # 应该不抛异常
        assert 'skipped' in result
        # 应该记录缺列原因
        assert len(result['skipped']) > 0
        # 某些切片应该有 missing column 原因
        skipped_reasons = list(result['skipped'].values())
        assert any('missing' in r.lower() for r in skipped_reasons)

    def test_slice_metrics_structure(self, sample_data):
        """切片指标结构正确"""
        y_true, y_pred, df = sample_data
        result = compute_slice_metrics(
            y_true, y_pred, df,
            whale_threshold=100,
            min_slice_n=100,
        )

        for slice_name, slice_data in result.items():
            if slice_name == 'skipped':
                continue

            # 每个切片应该有标准字段
            assert 'n' in slice_data
            assert 'total_revenue' in slice_data
            assert 'revcap_curve' in slice_data or 'notes' in slice_data


# =============================================================================
# Test: compute_ecosystem_metrics()
# =============================================================================

class TestComputeEcosystemMetrics:
    """生态指标测试"""

    @pytest.fixture
    def sample_data(self):
        """构造测试数据"""
        np.random.seed(42)
        n = 5000

        base_ts = 1700000000000  # 毫秒时间戳

        df = pd.DataFrame({
            'user_id': np.random.randint(1, 200, n),
            'streamer_id': np.random.randint(1, 100, n),
            'timestamp': base_ts + np.random.randint(0, 3600000, n),  # 1 小时内
            'streamer_gift_count': np.random.choice([0, 1, 5, 10, 20], n),
            'streamer_gift_sum': np.random.exponential(500, n),
            'user_gift_sum': np.random.exponential(100, n),
        })

        y_true = np.zeros(n)
        gifter_mask = np.random.random(n) < 0.02
        y_true[gifter_mask] = np.random.exponential(50, gifter_mask.sum())

        y_pred = np.random.uniform(0, 1, n)

        return y_true, y_pred, df

    def test_basic_structure(self, sample_data):
        """基本结构正确"""
        y_true, y_pred, df = sample_data
        result = compute_ecosystem_metrics(y_true, y_pred, df)

        assert 'selection' in result
        assert 'gini' in result
        assert 'coverage' in result
        assert 'overload' in result
        assert 'meta' in result

    def test_selection_info(self, sample_data):
        """selection 信息正确"""
        y_true, y_pred, df = sample_data
        result = compute_ecosystem_metrics(y_true, y_pred, df, k_select=0.05)

        sel = result['selection']
        assert sel['k_select'] == 0.05
        assert sel['n_selected'] == int(len(y_true) * 0.05)
        assert sel['n_total'] == len(y_true)

    def test_gini_exists(self, sample_data):
        """Gini 指标存在"""
        y_true, y_pred, df = sample_data
        result = compute_ecosystem_metrics(y_true, y_pred, df)

        gini = result['gini']
        assert 'streamer_revenue_gini' in gini
        assert 'top10_share' in gini

    def test_coverage_exists(self, sample_data):
        """Coverage 指标存在"""
        y_true, y_pred, df = sample_data
        result = compute_ecosystem_metrics(y_true, y_pred, df)

        cov = result['coverage']
        assert 'streamer_coverage' in cov
        assert 'tail_coverage' in cov
        assert 'cold_start_streamer_coverage' in cov

    def test_overload_exists(self, sample_data):
        """Overload 指标存在"""
        y_true, y_pred, df = sample_data
        result = compute_ecosystem_metrics(y_true, y_pred, df)

        ov = result['overload']
        assert 'window_minutes' in ov
        assert 'cap' in ov
        assert 'overload_bucket_rate' in ov
        assert 'overloaded_streamer_rate' in ov

    def test_missing_columns_graceful(self):
        """缺少列时优雅降级"""
        np.random.seed(42)
        n = 1000

        # 只有最基础的列
        df = pd.DataFrame({
            'user_id': np.random.randint(1, 100, n),
        })

        y_true = np.random.exponential(10, n)
        y_pred = np.random.uniform(0, 1, n)

        # 不应该抛异常
        result = compute_ecosystem_metrics(y_true, y_pred, df)

        assert 'meta' in result
        assert len(result['meta']['warnings']) > 0


# =============================================================================
# Test: evaluate_model() 集成
# =============================================================================

class TestEvaluateModelIntegration:
    """evaluate_model 集成测试"""

    @pytest.fixture
    def sample_data(self):
        """构造完整测试数据"""
        np.random.seed(42)
        n = 3000

        base_ts = 1700000000000

        df = pd.DataFrame({
            'user_id': np.random.randint(1, 200, n),
            'streamer_id': np.random.randint(1, 100, n),
            'timestamp': base_ts + np.arange(n) * 1000,  # 递增时间戳
            'pair_gift_count': np.random.choice([0, 1, 2, 5], n),
            'streamer_gift_count': np.random.choice([0, 1, 5, 10], n),
            'user_gift_sum': np.random.exponential(100, n),
            'streamer_gift_sum': np.random.exponential(500, n),
        })

        y_true = np.zeros(n)
        gifter_mask = np.random.random(n) < 0.02
        y_true[gifter_mask] = np.random.exponential(50, gifter_mask.sum())

        y_pred = np.random.uniform(0, 1, n)
        y_prob = 1 / (1 + np.exp(-y_pred * 5 + 2))  # sigmoid

        return y_true, y_pred, y_prob, df

    def test_with_y_prob(self, sample_data):
        """带 y_prob 的完整评估"""
        y_true, y_pred, y_prob, df = sample_data

        result = evaluate_model(
            y_true, y_pred, df,
            y_prob=y_prob,
            compute_slices=True,
            compute_ecosystem=True,
        )

        # 新字段应该存在
        assert result.prob_calibration is not None
        assert 'ece' in result.prob_calibration

    def test_without_y_prob(self, sample_data):
        """不带 y_prob"""
        y_true, y_pred, _, df = sample_data

        result = evaluate_model(
            y_true, y_pred, df,
            y_prob=None,
            compute_slices=True,
            compute_ecosystem=True,
        )

        # prob_calibration 应该是 None
        assert result.prob_calibration is None
        # 其他新指标应该存在
        assert result.slice_metrics is not None
        assert result.ecosystem is not None

    def test_backward_compatibility(self, sample_data):
        """向后兼容性：老参数应该工作"""
        y_true, y_pred, _, df = sample_data

        # 只用老参数
        result = evaluate_model(
            y_true, y_pred, df,
            whale_threshold=100,
            compute_stability=False,
            compute_slices=False,
            compute_ecosystem=False,
        )

        # 基础字段应该存在
        assert result.revcap_1pct > 0 or result.revcap_1pct == 0
        assert result.whale_recall_1pct >= 0 or np.isnan(result.whale_recall_1pct)

    def test_summary_output(self, sample_data):
        """summary() 输出包含新指标"""
        y_true, y_pred, y_prob, df = sample_data

        result = evaluate_model(
            y_true, y_pred, df,
            y_prob=y_prob,
            compute_slices=True,
            compute_ecosystem=True,
        )

        summary = result.summary()

        # 应该包含新 section
        assert "Probability Calibration" in summary or result.prob_calibration is None
        assert "Ecosystem Guardrails" in summary or not result.ecosystem

    def test_to_dict_serializable(self, sample_data):
        """to_dict() 应该可序列化"""
        y_true, y_pred, y_prob, df = sample_data

        result = evaluate_model(
            y_true, y_pred, df,
            y_prob=y_prob,
            compute_slices=True,
            compute_ecosystem=True,
        )

        d = result.to_dict()

        # 应该能转为 JSON
        import json
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 0

    def test_from_json_backward_compat(self, tmp_path):
        """from_json 向后兼容旧 JSON"""
        # 模拟旧版 JSON（没有新字段）
        old_json = {
            "n_samples": 1000,
            "n_gifters": 20,
            "n_whales": 5,
            "whale_threshold": 100.0,
            "gift_rate": 0.02,
            "whale_rate": 0.005,
            "total_revenue": 500.0,
            "revcap_1pct": 0.3,
            "revcap_curve": {"1%": 0.3},
            "oracle_curve": {"1%": 0.5},
            "normalized_curve": {"1%": 0.6},
        }

        json_path = tmp_path / "old_result.json"
        import json
        with open(json_path, 'w') as f:
            json.dump(old_json, f)

        # 应该能加载
        result = EvalResult.from_json(json_path)

        assert result.n_samples == 1000
        # 新字段应该有默认值
        assert result.prob_calibration is None
        assert result.slice_metrics == {}
        assert result.ecosystem == {}


# =============================================================================
# Test: revenue_capture_at_k (保持原有测试)
# =============================================================================

class TestRevCapAtK:
    """RevCap@K 基础测试"""

    def test_perfect_prediction(self):
        """完美预测 => RevCap = Oracle"""
        y_true = np.array([0, 100, 0, 200, 0, 50, 0, 0, 0, 0])
        y_pred = y_true.copy()  # 完美预测

        revcap = revenue_capture_at_k(y_true, y_pred, k=0.3)  # Top 3
        # 应该选中 200, 100, 50
        expected = 350 / 350  # 全部
        assert revcap == pytest.approx(expected, abs=0.01)

    def test_random_prediction(self):
        """随机预测"""
        np.random.seed(42)
        y_true = np.zeros(1000)
        y_true[:50] = np.random.exponential(100, 50)

        y_pred = np.random.uniform(0, 1, 1000)

        revcap = revenue_capture_at_k(y_true, y_pred, k=0.01)
        # 应该大于 0（有概率抓到一些）
        assert revcap >= 0

    def test_zero_revenue(self):
        """零收入"""
        y_true = np.zeros(100)
        y_pred = np.random.uniform(0, 1, 100)

        revcap = revenue_capture_at_k(y_true, y_pred, k=0.01)
        assert revcap == 0.0


# =============================================================================
# 运行测试
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
