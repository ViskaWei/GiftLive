#!/usr/bin/env python3
"""
Off-Policy Evaluation (OPE) Validation Script
Experiment ID: EXP-20260108-gift-allocation-10
MVP: MVP-3.1

Validates OPE methods (IPS, SNIPS, DR) against simulator ground truth.

Usage:
    source init.sh
    nohup python scripts/train_ope_validation.py > logs/ope_validation_20260108.log 2>&1 &
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import existing simulator
from simulator.simulator import (
    GiftLiveSimulator, SimConfig, User, AllocationPolicy, 
    RandomPolicy, GreedyPolicy, RoundRobinPolicy
)
from simulator.policies import ConcaveLogPolicy

# Paths
OUTPUT_DIR = Path("/home/swei20/GiftLive/gift_allocation")
IMG_DIR = OUTPUT_DIR / "img"
RESULTS_DIR = OUTPUT_DIR / "results"
IMG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set random seed
np.random.seed(42)

# Plot settings
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


# ===================== Propensity-Aware Policies =====================

class PropensityPolicy:
    """Base class for policies that can compute propensity scores."""
    
    def allocate_with_propensity(
        self, 
        users: List[User], 
        simulator: GiftLiveSimulator
    ) -> Tuple[List[int], List[float]]:
        """Allocate and return propensity scores.
        
        Returns:
            (allocations, propensities): Streamer IDs and their propensities
        """
        raise NotImplementedError
    
    def get_propensity(
        self,
        user: User,
        streamer_id: int,
        simulator: GiftLiveSimulator
    ) -> float:
        """Compute propensity of assigning user to streamer."""
        raise NotImplementedError


class UniformPropensityPolicy(PropensityPolicy):
    """Uniform random policy with propensity tracking."""
    
    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng(42)
    
    def allocate_with_propensity(
        self, 
        users: List[User], 
        simulator: GiftLiveSimulator
    ) -> Tuple[List[int], List[float]]:
        n_streamers = simulator.config.n_streamers
        propensity = 1.0 / n_streamers
        allocations = self.rng.integers(0, n_streamers, size=len(users)).tolist()
        propensities = [propensity] * len(users)
        return allocations, propensities
    
    def get_propensity(
        self,
        user: User,
        streamer_id: int,
        simulator: GiftLiveSimulator
    ) -> float:
        return 1.0 / simulator.config.n_streamers


class SoftmaxPropensityPolicy(PropensityPolicy):
    """Softmax policy based on expected values with propensity tracking."""
    
    def __init__(self, temperature: float = 1.0, rng: Optional[np.random.Generator] = None):
        self.temperature = temperature
        self.rng = rng or np.random.default_rng(42)
    
    def _compute_probs(
        self, 
        user: User, 
        simulator: GiftLiveSimulator
    ) -> np.ndarray:
        """Compute softmax probabilities over streamers."""
        scores = np.array([
            simulator.gift_model.expected_value(user, s)
            for s in simulator.streamer_pool.streamers
        ])
        
        # Softmax with temperature
        scores = scores / (self.temperature + 1e-10)
        scores = scores - scores.max()  # For numerical stability
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum()
        
        return probs
    
    def allocate_with_propensity(
        self, 
        users: List[User], 
        simulator: GiftLiveSimulator
    ) -> Tuple[List[int], List[float]]:
        allocations = []
        propensities = []
        
        for user in users:
            probs = self._compute_probs(user, simulator)
            action = self.rng.choice(len(probs), p=probs)
            allocations.append(action)
            propensities.append(probs[action])
        
        return allocations, propensities
    
    def get_propensity(
        self,
        user: User,
        streamer_id: int,
        simulator: GiftLiveSimulator
    ) -> float:
        probs = self._compute_probs(user, simulator)
        return probs[streamer_id]


class EpsilonGreedyPropensityPolicy(PropensityPolicy):
    """Epsilon-greedy policy with propensity tracking."""
    
    def __init__(self, epsilon: float = 0.3, rng: Optional[np.random.Generator] = None):
        self.epsilon = epsilon
        self.rng = rng or np.random.default_rng(42)
    
    def _compute_probs(
        self, 
        user: User, 
        simulator: GiftLiveSimulator
    ) -> np.ndarray:
        """Compute epsilon-greedy probabilities."""
        scores = np.array([
            simulator.gift_model.expected_value(user, s)
            for s in simulator.streamer_pool.streamers
        ])
        
        n = len(scores)
        best_action = np.argmax(scores)
        
        probs = np.full(n, self.epsilon / n)
        probs[best_action] += (1 - self.epsilon)
        
        return probs
    
    def allocate_with_propensity(
        self, 
        users: List[User], 
        simulator: GiftLiveSimulator
    ) -> Tuple[List[int], List[float]]:
        allocations = []
        propensities = []
        
        for user in users:
            probs = self._compute_probs(user, simulator)
            action = self.rng.choice(len(probs), p=probs)
            allocations.append(action)
            propensities.append(probs[action])
        
        return allocations, propensities
    
    def get_propensity(
        self,
        user: User,
        streamer_id: int,
        simulator: GiftLiveSimulator
    ) -> float:
        probs = self._compute_probs(user, simulator)
        return probs[streamer_id]


class GreedyPropensityPolicy(PropensityPolicy):
    """Deterministic greedy policy (propensity is 1 for best, 0 otherwise)."""
    
    def _get_best_action(self, user: User, simulator: GiftLiveSimulator) -> int:
        scores = np.array([
            simulator.gift_model.expected_value(user, s)
            for s in simulator.streamer_pool.streamers
        ])
        return int(np.argmax(scores))
    
    def allocate_with_propensity(
        self, 
        users: List[User], 
        simulator: GiftLiveSimulator
    ) -> Tuple[List[int], List[float]]:
        allocations = []
        propensities = []
        
        for user in users:
            action = self._get_best_action(user, simulator)
            allocations.append(action)
            propensities.append(1.0)  # Deterministic
        
        return allocations, propensities
    
    def get_propensity(
        self,
        user: User,
        streamer_id: int,
        simulator: GiftLiveSimulator
    ) -> float:
        best_action = self._get_best_action(user, simulator)
        return 1.0 if streamer_id == best_action else 0.0


class ConcavePropensityPolicy(PropensityPolicy):
    """Softmax on concave-adjusted scores."""
    
    def __init__(self, alpha: float = 0.5, temperature: float = 0.5,
                 rng: Optional[np.random.Generator] = None):
        self.alpha = alpha
        self.temperature = temperature
        self.rng = rng or np.random.default_rng(42)
    
    def _compute_probs(
        self, 
        user: User, 
        simulator: GiftLiveSimulator
    ) -> np.ndarray:
        # Get expected values
        scores = np.array([
            simulator.gift_model.expected_value(user, s)
            for s in simulator.streamer_pool.streamers
        ])
        
        # Apply concave transformation
        cum_revenues = np.array([s.cumulative_revenue for s in simulator.streamer_pool.streamers])
        marginal_utility = 1.0 / (1.0 + cum_revenues / 100.0)  # Diminishing returns
        
        concave_scores = scores * marginal_utility
        
        # Softmax
        concave_scores = concave_scores / (self.temperature + 1e-10)
        concave_scores = concave_scores - concave_scores.max()
        exp_scores = np.exp(concave_scores)
        probs = exp_scores / exp_scores.sum()
        
        return probs
    
    def allocate_with_propensity(
        self, 
        users: List[User], 
        simulator: GiftLiveSimulator
    ) -> Tuple[List[int], List[float]]:
        allocations = []
        propensities = []
        
        for user in users:
            probs = self._compute_probs(user, simulator)
            action = self.rng.choice(len(probs), p=probs)
            allocations.append(action)
            propensities.append(probs[action])
        
        return allocations, propensities
    
    def get_propensity(
        self,
        user: User,
        streamer_id: int,
        simulator: GiftLiveSimulator
    ) -> float:
        probs = self._compute_probs(user, simulator)
        return probs[streamer_id]


# ===================== Log Entry and Generation =====================

@dataclass
class LogEntry:
    """Single OPE log entry."""
    user: User
    streamer_id: int
    reward: float
    propensity: float
    expected_value: float


def generate_logs(
    simulator: GiftLiveSimulator,
    behavior_policy: PropensityPolicy,
    n_samples: int
) -> List[LogEntry]:
    """Generate logs with propensity scores."""
    logs = []
    
    for _ in range(n_samples):
        # Sample user
        user = simulator.user_pool.sample_users(1)[0]
        
        # Get allocation and propensity
        allocations, propensities = behavior_policy.allocate_with_propensity([user], simulator)
        streamer_id = allocations[0]
        propensity = propensities[0]
        
        # Simulate interaction
        streamer = simulator.streamer_pool.get_streamer(streamer_id)
        did_gift, amount = simulator.gift_model.simulate_interaction(user, streamer)
        reward = amount  # Reward is the gift amount (0 if no gift)
        
        # Expected value (for DM estimator)
        expected_value = simulator.gift_model.expected_value(user, streamer)
        
        logs.append(LogEntry(
            user=user,
            streamer_id=streamer_id,
            reward=reward,
            propensity=propensity,
            expected_value=expected_value
        ))
    
    return logs


# ===================== OPE Estimators =====================

class OPEEstimator:
    """Base class for OPE estimators."""
    
    def __init__(self, name: str):
        self.name = name
    
    def estimate(
        self, 
        logs: List[LogEntry],
        target_policy: PropensityPolicy,
        simulator: GiftLiveSimulator
    ) -> Tuple[float, Dict]:
        """Estimate policy value and return stats."""
        raise NotImplementedError


class IPSEstimator(OPEEstimator):
    """Inverse Propensity Scoring estimator."""
    
    def __init__(self, clip_M: Optional[float] = None):
        name = "IPS" if clip_M is None else f"IPS-Clip{int(clip_M)}"
        super().__init__(name)
        self.clip_M = clip_M
    
    def estimate(
        self,
        logs: List[LogEntry],
        target_policy: PropensityPolicy,
        simulator: GiftLiveSimulator
    ) -> Tuple[float, Dict]:
        weighted_rewards = []
        weights = []
        
        for log in logs:
            # Target policy propensity
            target_prop = target_policy.get_propensity(log.user, log.streamer_id, simulator)
            
            # Importance weight
            w = target_prop / (log.propensity + 1e-10)
            
            if self.clip_M is not None:
                w = min(w, self.clip_M)
            
            weights.append(w)
            weighted_rewards.append(w * log.reward)
        
        estimate = np.mean(weighted_rewards)
        weights = np.array(weights)
        
        ess = (np.sum(weights) ** 2) / (np.sum(weights ** 2) + 1e-10)
        
        stats = {
            'max_weight': float(np.max(weights)),
            'mean_weight': float(np.mean(weights)),
            'std_weight': float(np.std(weights)),
            'ess': float(ess),
            'n_samples': len(logs)
        }
        
        return estimate, stats


class SNIPSEstimator(OPEEstimator):
    """Self-Normalized IPS estimator."""
    
    def __init__(self):
        super().__init__("SNIPS")
    
    def estimate(
        self,
        logs: List[LogEntry],
        target_policy: PropensityPolicy,
        simulator: GiftLiveSimulator
    ) -> Tuple[float, Dict]:
        weighted_rewards = []
        weights = []
        
        for log in logs:
            target_prop = target_policy.get_propensity(log.user, log.streamer_id, simulator)
            w = target_prop / (log.propensity + 1e-10)
            weights.append(w)
            weighted_rewards.append(w * log.reward)
        
        weights = np.array(weights)
        weighted_rewards = np.array(weighted_rewards)
        
        estimate = np.sum(weighted_rewards) / (np.sum(weights) + 1e-10)
        ess = (np.sum(weights) ** 2) / (np.sum(weights ** 2) + 1e-10)
        
        stats = {
            'max_weight': float(np.max(weights)),
            'mean_weight': float(np.mean(weights)),
            'ess': float(ess),
            'n_samples': len(logs)
        }
        
        return estimate, stats


class DirectMethodEstimator(OPEEstimator):
    """Direct Method using expected values (simplified for efficiency)."""
    
    def __init__(self):
        super().__init__("DM")
    
    def estimate(
        self,
        logs: List[LogEntry],
        target_policy: PropensityPolicy,
        simulator: GiftLiveSimulator
    ) -> Tuple[float, Dict]:
        # Use the logged expected values directly (approximation)
        # More accurate would be to marginalize over target policy distribution
        # but this is too slow for large experiments
        predictions = [log.expected_value for log in logs]
        estimate = np.mean(predictions)
        
        stats = {
            'mean_prediction': float(np.mean(predictions)),
            'std_prediction': float(np.std(predictions)),
            'n_samples': len(logs)
        }
        
        return estimate, stats


class DoublyRobustEstimator(OPEEstimator):
    """Doubly Robust estimator (simplified for efficiency)."""
    
    def __init__(self, clip_M: Optional[float] = None):
        name = "DR" if clip_M is None else f"DR-Clip{int(clip_M)}"
        super().__init__(name)
        self.clip_M = clip_M
    
    def estimate(
        self,
        logs: List[LogEntry],
        target_policy: PropensityPolicy,
        simulator: GiftLiveSimulator
    ) -> Tuple[float, Dict]:
        dr_terms = []
        weights = []
        
        for log in logs:
            # Target propensity
            target_prop = target_policy.get_propensity(log.user, log.streamer_id, simulator)
            
            # Importance weight
            w = target_prop / (log.propensity + 1e-10)
            if self.clip_M is not None:
                w = min(w, self.clip_M)
            weights.append(w)
            
            # Q(x, a) = expected value for actual action
            q_actual = log.expected_value
            
            # Use q_actual as approximation for E_a~pi[Q(x,a)] 
            # This is a simplification; true DR would marginalize over policy
            # DR term: Q(x, a) + w * (r - Q(x, a))
            dr_term = q_actual + w * (log.reward - q_actual)
            dr_terms.append(dr_term)
        
        estimate = np.mean(dr_terms)
        weights = np.array(weights)
        ess = (np.sum(weights) ** 2) / (np.sum(weights ** 2) + 1e-10)
        
        stats = {
            'max_weight': float(np.max(weights)),
            'mean_weight': float(np.mean(weights)),
            'ess': float(ess),
            'n_samples': len(logs)
        }
        
        return estimate, stats


# ===================== Ground Truth Computation =====================

def compute_ground_truth(
    simulator: GiftLiveSimulator,
    target_policy: PropensityPolicy,
    n_samples: int = 10000
) -> float:
    """Compute ground truth policy value by on-policy sampling."""
    total_reward = 0.0
    
    for _ in range(n_samples):
        user = simulator.user_pool.sample_users(1)[0]
        allocations, _ = target_policy.allocate_with_propensity([user], simulator)
        streamer_id = allocations[0]
        
        streamer = simulator.streamer_pool.get_streamer(streamer_id)
        _, amount = simulator.gift_model.simulate_interaction(user, streamer)
        total_reward += amount
    
    return total_reward / n_samples


# ===================== Experiments =====================

def run_ope_comparison(
    simulator: GiftLiveSimulator,
    behavior_policy: PropensityPolicy,
    target_policies: Dict[str, PropensityPolicy],
    ope_methods: Dict[str, OPEEstimator],
    n_logs: int = 10000,
    n_repeats: int = 50,
    n_truth_samples: int = 20000
) -> Dict:
    """Run OPE method comparison."""
    print("\n=== OPE Comparison ===")
    print(f"Logs: {n_logs:,}, Repeats: {n_repeats}")
    
    results = {}
    
    for policy_name, target_policy in target_policies.items():
        print(f"\nTarget: {policy_name}")
        
        # Ground truth
        print("  Computing ground truth...")
        true_value = compute_ground_truth(simulator, target_policy, n_truth_samples)
        print(f"  Ground truth: {true_value:.4f}")
        
        policy_results = {'ground_truth': true_value, 'ope_methods': {}}
        
        for ope_name, ope_estimator in ope_methods.items():
            estimates = []
            all_stats = []
            
            for rep in tqdm(range(n_repeats), desc=f"  {ope_name}", leave=False):
                logs = generate_logs(simulator, behavior_policy, n_logs)
                estimate, stats = ope_estimator.estimate(logs, target_policy, simulator)
                estimates.append(estimate)
                all_stats.append(stats)
            
            estimates = np.array(estimates)
            bias = np.mean(estimates) - true_value
            variance = np.var(estimates)
            mse = bias ** 2 + variance
            rel_error = np.abs(np.mean(estimates) - true_value) / (np.abs(true_value) + 1e-10)
            
            policy_results['ope_methods'][ope_name] = {
                'mean_estimate': float(np.mean(estimates)),
                'std_estimate': float(np.std(estimates)),
                'bias': float(bias),
                'variance': float(variance),
                'mse': float(mse),
                'relative_error': float(rel_error),
                'estimates': estimates.tolist(),
                'avg_max_weight': float(np.mean([s.get('max_weight', 0) for s in all_stats])),
                'avg_ess': float(np.mean([s.get('ess', 0) for s in all_stats]))
            }
            
            print(f"    {ope_name}: RelErr={rel_error:.2%}, Bias={bias:.4f}")
        
        results[policy_name] = policy_results
    
    return results


def run_sample_size_sweep(
    simulator: GiftLiveSimulator,
    behavior_policy: PropensityPolicy,
    target_policy: PropensityPolicy,
    ope_methods: Dict[str, OPEEstimator],
    n_logs_list: List[int],
    n_repeats: int = 30
) -> Dict:
    """Sweep over sample sizes."""
    print("\n=== Sample Size Sweep ===")
    
    true_value = compute_ground_truth(simulator, target_policy, 20000)
    print(f"Ground truth: {true_value:.4f}")
    
    results = {'ground_truth': true_value, 'sweeps': {}}
    
    for ope_name, ope_estimator in ope_methods.items():
        print(f"\n{ope_name}:")
        method_results = []
        
        for n_logs in n_logs_list:
            estimates = []
            for _ in tqdm(range(n_repeats), desc=f"  N={n_logs}", leave=False):
                logs = generate_logs(simulator, behavior_policy, n_logs)
                estimate, _ = ope_estimator.estimate(logs, target_policy, simulator)
                estimates.append(estimate)
            
            estimates = np.array(estimates)
            rel_error = np.abs(np.mean(estimates) - true_value) / (np.abs(true_value) + 1e-10)
            
            method_results.append({
                'n_logs': n_logs,
                'mean_estimate': float(np.mean(estimates)),
                'std_estimate': float(np.std(estimates)),
                'relative_error': float(rel_error)
            })
            print(f"    N={n_logs}: RelErr={rel_error:.2%}")
        
        results['sweeps'][ope_name] = method_results
    
    return results


def run_epsilon_sweep(
    simulator: GiftLiveSimulator,
    target_policy: PropensityPolicy,
    ope_methods: Dict[str, OPEEstimator],
    epsilon_list: List[float],
    n_logs: int = 10000,
    n_repeats: int = 30
) -> Dict:
    """Sweep over exploration rates."""
    print("\n=== Epsilon Sweep ===")
    
    true_value = compute_ground_truth(simulator, target_policy, 20000)
    print(f"Ground truth: {true_value:.4f}")
    
    results = {'ground_truth': true_value, 'sweeps': {}}
    
    for ope_name, ope_estimator in ope_methods.items():
        print(f"\n{ope_name}:")
        method_results = []
        
        for epsilon in epsilon_list:
            behavior_policy = EpsilonGreedyPropensityPolicy(epsilon=epsilon)
            estimates = []
            
            for _ in tqdm(range(n_repeats), desc=f"  eps={epsilon:.1f}", leave=False):
                logs = generate_logs(simulator, behavior_policy, n_logs)
                estimate, _ = ope_estimator.estimate(logs, target_policy, simulator)
                estimates.append(estimate)
            
            estimates = np.array(estimates)
            rel_error = np.abs(np.mean(estimates) - true_value) / (np.abs(true_value) + 1e-10)
            
            method_results.append({
                'epsilon': epsilon,
                'mean_estimate': float(np.mean(estimates)),
                'std_estimate': float(np.std(estimates)),
                'relative_error': float(rel_error)
            })
            print(f"    eps={epsilon}: RelErr={rel_error:.2%}")
        
        results['sweeps'][ope_name] = method_results
    
    return results


# ===================== Plotting =====================

def plot_fig1_ope_comparison(results: Dict, output_path: Path):
    """Fig1: OPE method comparison - Relative Error."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    policies = list(results.keys())
    methods = list(results[policies[0]]['ope_methods'].keys())
    
    x = np.arange(len(methods))
    width = 0.8 / len(policies)
    
    for i, policy in enumerate(policies):
        rel_errors = [results[policy]['ope_methods'][m]['relative_error'] for m in methods]
        ax.bar(x + i * width - 0.4 + width/2, rel_errors, width, label=policy, alpha=0.8)
    
    ax.set_xlabel('OPE Method')
    ax.set_ylabel('Relative Error')
    ax.set_title('OPE Method Comparison: Relative Error')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(title='Target Policy')
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='10% threshold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_fig2_bias_variance(results: Dict, output_path: Path):
    """Fig2: Bias vs Variance decomposition."""
    methods = list(results[list(results.keys())[0]]['ope_methods'].keys())
    
    biases = []
    variances = []
    
    for method in methods:
        avg_bias = np.mean([abs(results[p]['ope_methods'][method]['bias']) for p in results])
        avg_var = np.mean([results[p]['ope_methods'][method]['variance'] for p in results])
        biases.append(avg_bias)
        variances.append(avg_var)
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, biases, width, label='|Bias|', color='steelblue')
    ax.bar(x + width/2, variances, width, label='Variance', color='coral')
    
    ax.set_xlabel('OPE Method')
    ax.set_ylabel('Value')
    ax.set_title('Bias-Variance Decomposition')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_fig3_sample_size_effect(results: Dict, output_path: Path):
    """Fig3: Sample size effect."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results['sweeps'])))
    
    for (method, data), color in zip(results['sweeps'].items(), colors):
        n_logs = [d['n_logs'] for d in data]
        rel_errors = [d['relative_error'] for d in data]
        ax.plot(n_logs, rel_errors, 'o-', label=method, color=color, linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Logs')
    ax.set_ylabel('Relative Error')
    ax.set_title('Effect of Sample Size on OPE Accuracy')
    ax.set_xscale('log')
    ax.legend()
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_fig4_epsilon_effect(results: Dict, output_path: Path):
    """Fig4: Epsilon effect."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results['sweeps'])))
    
    for (method, data), color in zip(results['sweeps'].items(), colors):
        epsilons = [d['epsilon'] for d in data]
        rel_errors = [d['relative_error'] for d in data]
        ax.plot(epsilons, rel_errors, 'o-', label=method, color=color, linewidth=2, markersize=8)
    
    ax.set_xlabel('Epsilon (Exploration Rate)')
    ax.set_ylabel('Relative Error')
    ax.set_title('Effect of Exploration Rate on OPE Accuracy')
    ax.legend()
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_fig5_estimate_distribution(results: Dict, output_path: Path):
    """Fig5: Distribution of estimates."""
    n_policies = len(results)
    fig, axes = plt.subplots(1, n_policies, figsize=(4 * n_policies, 5))
    if n_policies == 1:
        axes = [axes]
    
    for ax, (policy, data) in zip(axes, results.items()):
        methods = list(data['ope_methods'].keys())
        estimates_list = [data['ope_methods'][m]['estimates'] for m in methods]
        
        bp = ax.boxplot(estimates_list, labels=methods, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.axhline(y=data['ground_truth'], color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('OPE Method')
        ax.set_ylabel('Estimate')
        ax.set_title(f'Target: {policy}')
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Distribution of OPE Estimates', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_fig6_heatmap(results: Dict, output_path: Path):
    """Fig6: Heatmap of relative errors."""
    policies = list(results.keys())
    methods = list(results[policies[0]]['ope_methods'].keys())
    
    matrix = np.zeros((len(policies), len(methods)))
    for i, policy in enumerate(policies):
        for j, method in enumerate(methods):
            matrix[i, j] = results[policy]['ope_methods'][method]['relative_error']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.3)
    
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(policies)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_yticklabels(policies)
    
    for i in range(len(policies)):
        for j in range(len(methods)):
            color = 'white' if matrix[i, j] > 0.15 else 'black'
            ax.text(j, i, f'{matrix[i, j]:.1%}', ha='center', va='center', color=color)
    
    ax.set_xlabel('OPE Method')
    ax.set_ylabel('Target Policy')
    ax.set_title('Relative Error Heatmap')
    plt.colorbar(im, ax=ax, label='Relative Error')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ===================== Main =====================

def main():
    print("=" * 60)
    print("OPE Validation Experiment")
    print("EXP-20260108-gift-allocation-10 | MVP-3.1")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # Initialize simulator (smaller for faster experiments)
    print("\nInitializing simulator...")
    config = SimConfig(n_users=500, n_streamers=50, seed=42)
    simulator = GiftLiveSimulator(config)
    
    # Define policies
    behavior_policy = EpsilonGreedyPropensityPolicy(epsilon=0.3)
    
    target_policies = {
        'Greedy': GreedyPropensityPolicy(),
        'Softmax': SoftmaxPropensityPolicy(temperature=0.5),
        'Concave': ConcavePropensityPolicy(alpha=0.5, temperature=0.5),
    }
    
    # Define OPE methods
    ope_methods = {
        'IPS': IPSEstimator(clip_M=None),
        'IPS-Clip10': IPSEstimator(clip_M=10),
        'SNIPS': SNIPSEstimator(),
        'DM': DirectMethodEstimator(),
        'DR': DoublyRobustEstimator(clip_M=None),
        'DR-Clip10': DoublyRobustEstimator(clip_M=10),
    }
    
    # Experiment 1: Method comparison
    print("\n" + "=" * 40)
    print("Experiment 1: OPE Method Comparison")
    print("=" * 40)
    comparison_results = run_ope_comparison(
        simulator, behavior_policy, target_policies, ope_methods,
        n_logs=5000, n_repeats=20, n_truth_samples=10000
    )
    
    # Experiment 2: Sample size sweep
    print("\n" + "=" * 40)
    print("Experiment 2: Sample Size Sensitivity")
    print("=" * 40)
    sample_size_results = run_sample_size_sweep(
        simulator, behavior_policy, target_policies['Greedy'],
        {'IPS': IPSEstimator(), 'SNIPS': SNIPSEstimator(), 'DR': DoublyRobustEstimator()},
        n_logs_list=[500, 1000, 2000, 5000, 10000],
        n_repeats=15
    )
    
    # Experiment 3: Epsilon sweep
    print("\n" + "=" * 40)
    print("Experiment 3: Exploration Rate Sensitivity")
    print("=" * 40)
    epsilon_results = run_epsilon_sweep(
        simulator, target_policies['Greedy'],
        {'SNIPS': SNIPSEstimator(), 'DR': DoublyRobustEstimator()},
        epsilon_list=[0.1, 0.2, 0.3, 0.5, 0.7],
        n_logs=5000, n_repeats=15
    )
    
    # Generate plots
    print("\n" + "=" * 40)
    print("Generating Plots")
    print("=" * 40)
    
    plot_fig1_ope_comparison(comparison_results, IMG_DIR / "mvp31_ope_comparison.png")
    plot_fig2_bias_variance(comparison_results, IMG_DIR / "mvp31_bias_variance.png")
    plot_fig3_sample_size_effect(sample_size_results, IMG_DIR / "mvp31_sample_size_effect.png")
    plot_fig4_epsilon_effect(epsilon_results, IMG_DIR / "mvp31_epsilon_effect.png")
    plot_fig5_estimate_distribution(comparison_results, IMG_DIR / "mvp31_estimate_distribution.png")
    plot_fig6_heatmap(comparison_results, IMG_DIR / "mvp31_policy_ope_heatmap.png")
    
    # Compile results
    print("\n" + "=" * 40)
    print("Compiling Results")
    print("=" * 40)
    
    # Find best method
    avg_rel_errors = {}
    for method in ope_methods.keys():
        errors = [comparison_results[p]['ope_methods'][method]['relative_error'] 
                 for p in comparison_results]
        avg_rel_errors[method] = np.mean(errors)
    
    best_method = min(avg_rel_errors, key=avg_rel_errors.get)
    best_rel_error = avg_rel_errors[best_method]
    
    # Gate-3 decision
    gate3_result = "closed" if best_rel_error < 0.10 else "open"
    recommendation = "dr_for_policy_comparison" if gate3_result == "closed" else "use_simulator"
    
    final_results = {
        "experiment_id": "EXP-20260108-gift-allocation-10",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ope_comparison": {
            method: {
                'avg_rel_error': float(avg_rel_errors[method]),
                'avg_bias': float(np.mean([comparison_results[p]['ope_methods'][method]['bias'] 
                                          for p in comparison_results])),
                'avg_variance': float(np.mean([comparison_results[p]['ope_methods'][method]['variance'] 
                                              for p in comparison_results])),
            }
            for method in ope_methods.keys()
        },
        "best_method": best_method,
        "best_rel_error": float(best_rel_error),
        "gate3_result": gate3_result,
        "recommendation": recommendation,
        "sample_size_sensitivity": {
            method: data[-1]['relative_error']
            for method, data in sample_size_results['sweeps'].items()
        },
        "epsilon_sensitivity": {
            method: {d['epsilon']: d['relative_error'] for d in data}
            for method, data in epsilon_results['sweeps'].items()
        },
        "runtime_seconds": (datetime.now() - start_time).total_seconds()
    }
    
    # Save results
    output_file = RESULTS_DIR / "ope_validation_20260108.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Best OPE Method: {best_method}")
    print(f"Best Relative Error: {best_rel_error:.2%}")
    print(f"Gate-3 Result: {gate3_result.upper()}")
    print(f"Recommendation: {recommendation}")
    print(f"\nMethod Ranking:")
    for i, (method, error) in enumerate(sorted(avg_rel_errors.items(), key=lambda x: x[1])):
        status = "✅" if error < 0.10 else "❌"
        print(f"  {i+1}. {method}: {error:.2%} {status}")
    print(f"\nRuntime: {(datetime.now() - start_time).total_seconds():.1f}s")
    print("=" * 60)
    
    return final_results


if __name__ == "__main__":
    results = main()
