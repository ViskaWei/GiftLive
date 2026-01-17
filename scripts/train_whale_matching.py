#!/usr/bin/env python3
"""
Whale Matching Experiments
MVP-5.3: EXP-20260109-gift-allocation-53

This script runs experiments for hierarchical whale matching:
1. Algorithm Comparison: b-matching vs min-cost-flow vs greedy-swaps
2. k-value Sweep: k ∈ {1, 2, 3, 5}
3. Whale Threshold Sweep: Top 0.1% vs Top 1% vs Top 5%
4. Full Grid: algorithm × k × threshold

Outputs:
- Results JSON: gift_allocation/results/whale_matching_20260109.json
- Figures: gift_allocation/img/mvp53_*.png
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
# Gini coefficient calculation
def gini(x):
    """Calculate Gini coefficient."""
    x = np.array(x)
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.simulator import (
    SimConfig,
    GiftLiveSimulator,
    GreedyPolicy,
    WhaleMatchingPolicy,
)

# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = PROJECT_ROOT / "gift_allocation" / "results"
IMG_DIR = PROJECT_ROOT / "gift_allocation" / "img"
SEED = 42

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Simulation config (V2+ with capacity)
SIM_CONFIG = dict(
    n_users=10000,
    n_streamers=500,
    seed=SEED,
    amount_version=3,  # V2+ discrete tiers
    enable_capacity=True,
    capacity_top10=15,  # Reduced from MVP-4.2 to trigger crowding
    capacity_middle=8,
    capacity_tail=3,
    crowding_penalty_alpha=0.5,
    new_streamer_ratio=0.2,
)

# Experiment config
EXP_CONFIG = dict(
    n_rounds=50,
    users_per_round=200,
    n_simulations=30,  # Reduced for faster execution
)

# ============================================================
# Metrics Computation
# ============================================================

def compute_overload_rate(simulator: GiftLiveSimulator) -> float:
    """Compute capacity overload rate."""
    if not simulator.interaction_log:
        return 0.0
    
    overload_count = sum(
        1 for record in simulator.interaction_log
        if record.get('is_overcrowded', False)
    )
    total = len(simulator.interaction_log)
    return overload_count / total if total > 0 else 0.0


def compute_streamer_gini(simulator: GiftLiveSimulator) -> float:
    """Compute Gini coefficient for streamer revenues."""
    revenues = np.array([s.cumulative_revenue for s in simulator.streamer_pool.streamers])
    if len(revenues) == 0 or np.sum(revenues) == 0:
        return 0.0
    revenues = np.sort(revenues)
    n = len(revenues)
    index = np.arange(1, n + 1)
    gini_coef = (2 * np.sum(index * revenues)) / (n * np.sum(revenues)) - (n + 1) / n
    return float(gini_coef)


def compute_whale_distribution(
    simulator: GiftLiveSimulator,
    whale_ids: set,
    allocations: List[Tuple[int, int]]
) -> Dict[str, Any]:
    """Compute whale distribution metrics."""
    # Count whales per streamer
    whale_counts = defaultdict(int)
    for user_id, streamer_id in allocations:
        if user_id in whale_ids:
            whale_counts[streamer_id] += 1
    
    counts = list(whale_counts.values())
    
    # Compute entropy
    if counts:
        probs = np.array(counts) / sum(counts)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
    else:
        entropy = 0.0
    
    return {
        'whale_counts': dict(whale_counts),
        'n_streamers_with_whales': len(whale_counts),
        'whale_distribution_entropy': float(entropy),
        'max_whales_per_streamer': max(counts) if counts else 0,
    }


# ============================================================
# Experiment Functions
# ============================================================

def run_baseline(n_simulations: int = 30) -> Dict:
    """Run greedy baseline."""
    print("\n" + "=" * 60)
    print("Baseline: Greedy Policy")
    print("=" * 60)
    
    results = []
    
    for sim_idx in tqdm(range(n_simulations), desc="Greedy Baseline"):
        seed = SEED + sim_idx
        config = SimConfig(**{**SIM_CONFIG, 'seed': seed})
        sim = GiftLiveSimulator(config)
        
        policy = GreedyPolicy()
        metrics = sim.run_simulation(
            policy,
            n_rounds=EXP_CONFIG['n_rounds'],
            users_per_round=EXP_CONFIG['users_per_round']
        )
        
        # Add additional metrics
        metrics['overload_rate'] = compute_overload_rate(sim)
        metrics['streamer_gini'] = compute_streamer_gini(sim)
        
        results.append(metrics)
    
    # Aggregate
    summary = {
        'revenue_mean': float(np.mean([m['total_revenue'] for m in results])),
        'revenue_std': float(np.std([m['total_revenue'] for m in results])),
        'overload_rate_mean': float(np.mean([m['overload_rate'] for m in results])),
        'streamer_gini_mean': float(np.mean([m['streamer_gini'] for m in results])),
    }
    
    print(f"\nGreedy Baseline:")
    print(f"  Revenue: {summary['revenue_mean']:.2f} ± {summary['revenue_std']:.2f}")
    print(f"  Overload Rate: {summary['overload_rate_mean']:.3f}")
    print(f"  Streamer Gini: {summary['streamer_gini_mean']:.3f}")
    
    return {'baseline': summary, 'raw': results}


def run_k_sweep(
    algorithm: str = "b-matching",
    whale_threshold: str = "Top 1%",
    n_simulations: int = 30
) -> Dict:
    """Experiment: k-value sweep."""
    print(f"\n" + "=" * 60)
    print(f"k-value Sweep: {algorithm}, {whale_threshold}")
    print("=" * 60)
    
    k_values = [1, 2, 3, 5]
    results = {k: [] for k in k_values}
    
    for k in k_values:
        print(f"\n  k={k}...")
        for sim_idx in tqdm(range(n_simulations), desc=f"k={k}", leave=False):
            seed = SEED + sim_idx
            config = SimConfig(**{**SIM_CONFIG, 'seed': seed})
            sim = GiftLiveSimulator(config)
            
            policy = WhaleMatchingPolicy(
                algorithm=algorithm,
                k=k,
                whale_threshold=whale_threshold
            )
            metrics = sim.run_simulation(
                policy,
                n_rounds=EXP_CONFIG['n_rounds'],
                users_per_round=EXP_CONFIG['users_per_round']
            )
            
            # Add additional metrics
            metrics['overload_rate'] = compute_overload_rate(sim)
            metrics['streamer_gini'] = compute_streamer_gini(sim)
            
            results[k].append(metrics)
    
    # Aggregate
    summary = {}
    for k, metrics_list in results.items():
        summary[k] = {
            'revenue_mean': float(np.mean([m['total_revenue'] for m in metrics_list])),
            'revenue_std': float(np.std([m['total_revenue'] for m in metrics_list])),
            'overload_rate_mean': float(np.mean([m['overload_rate'] for m in metrics_list])),
            'streamer_gini_mean': float(np.mean([m['streamer_gini'] for m in metrics_list])),
        }
    
    print("\nk-value Summary:")
    for k, data in summary.items():
        print(f"  k={k}: Revenue={data['revenue_mean']:.2f}, "
              f"Overload={data['overload_rate_mean']:.3f}, "
              f"Gini={data['streamer_gini_mean']:.3f}")
    
    return {f'{algorithm}_{whale_threshold}': summary}


def run_algorithm_comparison(
    k: int = 2,
    whale_threshold: str = "Top 1%",
    n_simulations: int = 30
) -> Dict:
    """Experiment: Algorithm comparison."""
    print(f"\n" + "=" * 60)
    print(f"Algorithm Comparison: k={k}, {whale_threshold}")
    print("=" * 60)
    
    algorithms = ["b-matching", "greedy-swaps"]  # Skip min-cost-flow for now
    results = {alg: [] for alg in algorithms}
    
    for alg in algorithms:
        print(f"\n  {alg}...")
        for sim_idx in tqdm(range(n_simulations), desc=alg, leave=False):
            seed = SEED + sim_idx
            config = SimConfig(**{**SIM_CONFIG, 'seed': seed})
            sim = GiftLiveSimulator(config)
            
            policy = WhaleMatchingPolicy(
                algorithm=alg,
                k=k,
                whale_threshold=whale_threshold
            )
            metrics = sim.run_simulation(
                policy,
                n_rounds=EXP_CONFIG['n_rounds'],
                users_per_round=EXP_CONFIG['users_per_round']
            )
            
            metrics['overload_rate'] = compute_overload_rate(sim)
            metrics['streamer_gini'] = compute_streamer_gini(sim)
            
            results[alg].append(metrics)
    
    # Aggregate
    summary = {}
    for alg, metrics_list in results.items():
        summary[alg] = {
            'revenue_mean': float(np.mean([m['total_revenue'] for m in metrics_list])),
            'revenue_std': float(np.std([m['total_revenue'] for m in metrics_list])),
            'overload_rate_mean': float(np.mean([m['overload_rate'] for m in metrics_list])),
            'streamer_gini_mean': float(np.mean([m['streamer_gini'] for m in metrics_list])),
        }
    
    print("\nAlgorithm Summary:")
    for alg, data in summary.items():
        print(f"  {alg}: Revenue={data['revenue_mean']:.2f}, "
              f"Overload={data['overload_rate_mean']:.3f}, "
              f"Gini={data['streamer_gini_mean']:.3f}")
    
    return {'algorithm_comparison': summary}


# ============================================================
# Main
# ============================================================

def main():
    """Run all experiments."""
    print("=" * 60)
    print("Whale Matching Experiments (MVP-5.3)")
    print("=" * 60)
    
    all_results = {
        'experiment_id': 'EXP-20260109-gift-allocation-53',
        'timestamp': datetime.now().isoformat(),
    }
    
    # 1. Baseline
    baseline_results = run_baseline(n_simulations=EXP_CONFIG['n_simulations'])
    all_results.update(baseline_results)
    baseline_revenue = baseline_results['baseline']['revenue_mean']
    
    # 2. k-value sweep (b-matching, Top 1%)
    k_sweep_results = run_k_sweep(
        algorithm="b-matching",
        whale_threshold="Top 1%",
        n_simulations=EXP_CONFIG['n_simulations']
    )
    all_results.update(k_sweep_results)
    
    # 3. Algorithm comparison
    alg_comp_results = run_algorithm_comparison(
        k=2,
        whale_threshold="Top 1%",
        n_simulations=EXP_CONFIG['n_simulations']
    )
    all_results.update(alg_comp_results)
    
    # Compute deltas vs baseline
    if 'b-matching_Top 1%' in all_results:
        for k, data in all_results['b-matching_Top 1%'].items():
            data['revenue_delta_pct'] = (data['revenue_mean'] - baseline_revenue) / baseline_revenue * 100
    
    # Save results
    output_file = RESULTS_DIR / "whale_matching_20260109.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Determine best config
    best_config = None
    best_score = -np.inf
    
    if 'b-matching_Top 1%' in all_results:
        for k, data in all_results['b-matching_Top 1%'].items():
            # Score: low overload + low gini + high revenue
            score = (
                -data['overload_rate_mean'] * 10 +  # Penalize overload
                -data['streamer_gini_mean'] * 5 +   # Penalize high gini
                data['revenue_delta_pct'] / 10      # Reward revenue
            )
            if score > best_score:
                best_score = score
                best_config = {
                    'algorithm': 'b-matching',
                    'k': int(k),
                    'whale_threshold': 'Top 1%',
                    **data
                }
    
    all_results['best_config'] = best_config
    
    # Gate-5C evaluation
    if best_config:
        gate5c_pass = (
            best_config['overload_rate_mean'] < 0.10 and
            best_config['streamer_gini_mean'] < baseline_results['baseline']['streamer_gini_mean'] and
            best_config['revenue_delta_pct'] > -5.0
        )
        all_results['gate5c'] = "PASS" if gate5c_pass else "FAIL"
        
        print(f"\n{'='*60}")
        print(f"Gate-5C Evaluation: {all_results['gate5c']}")
        print(f"{'='*60}")
        print(f"Best Config: {best_config['algorithm']}, k={best_config['k']}, {best_config['whale_threshold']}")
        print(f"  Overload Rate: {best_config['overload_rate_mean']:.3f} (<0.10: {'✅' if best_config['overload_rate_mean'] < 0.10 else '❌'})")
        print(f"  Gini: {best_config['streamer_gini_mean']:.3f} (vs baseline {baseline_results['baseline']['streamer_gini_mean']:.3f})")
        print(f"  Revenue Δ: {best_config['revenue_delta_pct']:.2f}% (>-5%: {'✅' if best_config['revenue_delta_pct'] > -5.0 else '❌'})")
    
    # Save again with gate result
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Complete! Results: {output_file}")


if __name__ == "__main__":
    main()
