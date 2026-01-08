#!/usr/bin/env python3
"""
Simulator Calibration Tools
Experiment: MVP-0.3

Tools for calibrating the simulator against real KuaiLive data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from pathlib import Path


class SimulatorCalibrator:
    """Calibrate simulator parameters to match real data statistics."""
    
    # Target statistics from KuaiLive EDA (EXP-20260108-gift-allocation-01)
    TARGET_STATS = {
        'gift_rate': 0.05,  # Approximate (gift_rate=1.0 means only gift records)
        'amount_mean': 82.68,
        'amount_median': 2.0,
        'amount_p90': 88.0,
        'amount_p99': 1488.2,
        'amount_max': 56246.0,
        'user_gini': 0.94,
        'streamer_gini': 0.93,
        'top_1pct_user_share': 0.60,
        'top_1pct_streamer_share': 0.53,
    }
    
    def __init__(self, simulator):
        """
        Initialize calibrator.
        
        Args:
            simulator: GiftLiveSimulator instance
        """
        self.simulator = simulator
    
    def compute_stats(self, n_samples: int = 50000) -> Dict[str, float]:
        """Compute statistics from simulated data."""
        from .policies import UniformPolicy
        
        # Generate samples with uniform policy
        policy = UniformPolicy()
        logs = self.simulator.generate_logs(n_samples, policy)
        
        rewards = np.array([log.reward for log in logs])
        nonzero_rewards = rewards[rewards > 0]
        
        if len(nonzero_rewards) == 0:
            return {'error': 'No non-zero rewards'}
        
        # Compute Gini for user dimension
        user_rewards = {}
        for log in logs:
            if log.reward > 0:
                if log.user_id not in user_rewards:
                    user_rewards[log.user_id] = 0
                user_rewards[log.user_id] += log.reward
        
        user_totals = np.array(list(user_rewards.values()))
        user_gini = self._compute_gini(user_totals)
        
        # Compute Gini for streamer dimension
        streamer_rewards = {}
        for log in logs:
            if log.reward > 0:
                if log.streamer_id not in streamer_rewards:
                    streamer_rewards[log.streamer_id] = 0
                streamer_rewards[log.streamer_id] += log.reward
        
        streamer_totals = np.array(list(streamer_rewards.values()))
        streamer_gini = self._compute_gini(streamer_totals)
        
        stats = {
            'gift_rate': float(np.mean(rewards > 0)),
            'amount_mean': float(np.mean(nonzero_rewards)),
            'amount_median': float(np.median(nonzero_rewards)),
            'amount_p90': float(np.percentile(nonzero_rewards, 90)),
            'amount_p99': float(np.percentile(nonzero_rewards, 99)),
            'amount_max': float(np.max(nonzero_rewards)),
            'user_gini': float(user_gini),
            'streamer_gini': float(streamer_gini),
            'n_samples': n_samples,
            'n_nonzero': len(nonzero_rewards),
        }
        
        return stats
    
    def _compute_gini(self, values: np.ndarray) -> float:
        """Compute Gini coefficient."""
        if len(values) == 0:
            return 0.0
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)
        gini = (2 * np.sum((np.arange(1, n+1) * values)) - (n + 1) * np.sum(values)) / (n * np.sum(values))
        return gini
    
    def compare_with_target(self, n_samples: int = 50000) -> pd.DataFrame:
        """Compare simulated stats with target stats."""
        sim_stats = self.compute_stats(n_samples)
        
        comparison = []
        for key, target in self.TARGET_STATS.items():
            if key in sim_stats:
                sim_val = sim_stats[key]
                rel_error = abs(sim_val - target) / (target + 1e-10)
                comparison.append({
                    'metric': key,
                    'target': target,
                    'simulated': sim_val,
                    'rel_error': rel_error
                })
        
        return pd.DataFrame(comparison)
    
    def print_comparison(self, n_samples: int = 50000):
        """Print comparison table."""
        df = self.compare_with_target(n_samples)
        print("\n=== Simulator Calibration ===")
        print(df.to_string(index=False))
        print()
