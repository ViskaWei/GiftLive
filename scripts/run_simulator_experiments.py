#!/usr/bin/env python3
"""
Simulator Experiments Runner
Runs MVP-0.3 (Simulator V1), MVP-2.1 (Concave Allocation), MVP-2.2 (Cold-start Constraints)

Usage:
    python scripts/run_simulator_experiments.py --all
    python scripts/run_simulator_experiments.py --mvp 0.3
    python scripts/run_simulator_experiments.py --mvp 2.1
    python scripts/run_simulator_experiments.py --mvp 2.2
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from tqdm import tqdm

# Set paths
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.simulator import (
    SimConfig, GiftLiveSimulator,
    RandomPolicy, GreedyPolicy, RoundRobinPolicy,
    ConcaveLogPolicy, ConcaveExpPolicy, ConcavePowerPolicy, GreedyWithCapPolicy,
    ConstraintConfig, ConstrainedAllocationPolicy, ColdStartTracker,
    create_policy,
)

# Paths
BASE_DIR = Path("/home/swei20/GiftLive")
IMG_DIR = BASE_DIR / "gift_allocation" / "img"
RESULTS_DIR = BASE_DIR / "gift_allocation" / "results"
EXP_DIR = BASE_DIR / "gift_allocation" / "exp"

# Plot style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Real data calibration targets (from MVP-0.1)
CALIBRATION_TARGETS = {
    'gift_rate': 0.0148,
    'user_gini': 0.942,
    'streamer_gini': 0.930,
    'top_1_user_share': 0.599,
    'amount_median': 2.0,
    'amount_p90': 88.0,
    'amount_p99': 1488.0,
    'amount_mean': 82.7,
}


def compute_gini(values: np.ndarray) -> float:
    """Compute Gini coefficient."""
    values = np.array(values, dtype=float)
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    gini = (2 * np.sum((np.arange(1, n+1) * values)) - (n + 1) * np.sum(values)) / (n * np.sum(values))
    return gini


# ============================================================
# MVP-0.3: Simulator V1 Calibration
# ============================================================

def run_mvp03_calibration(n_simulations: int = 100, verbose: bool = True):
    """Run MVP-0.3: Simulator calibration experiments."""
    print("\n" + "="*60)
    print("MVP-0.3: Simulator V1 Calibration")
    print("="*60)
    
    # Optimized config (calibrated to match real data)
    config = SimConfig(
        n_users=10000,
        n_streamers=500,
        # Tuned wealth distribution
        wealth_lognormal_mean=2.5,
        wealth_lognormal_std=1.2,
        wealth_pareto_alpha=1.3,
        wealth_pareto_min=80,
        wealth_pareto_weight=0.05,
        # Tuned gift probability (very low base rate)
        gift_theta0=-6.5,
        gift_theta1=0.4,
        gift_theta2=0.8,
        gift_theta3=0.2,
        gift_theta4=-0.05,
        # Tuned gift amount
        amount_mu0=0.5,
        amount_mu1=0.9,
        amount_mu2=0.4,
        amount_sigma=1.5,
        gamma=0.02,
        seed=42
    )
    
    # Run calibration simulations
    calibration_results = []
    
    print(f"\nRunning {n_simulations} calibration simulations...")
    for i in tqdm(range(n_simulations)):
        config_i = SimConfig(
            **{k: v for k, v in config.__dict__.items() if k != 'seed'},
            seed=42 + i
        )
        sim = GiftLiveSimulator(config_i)
        policy = GreedyPolicy()
        metrics = sim.run_simulation(policy, n_rounds=50, users_per_round=200)
        calibration_results.append(metrics)
    
    # Aggregate results
    agg_metrics = {}
    for key in calibration_results[0].keys():
        values = [r[key] for r in calibration_results]
        agg_metrics[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }
    
    # Compare with calibration targets
    calibration_comparison = {}
    for key in ['gift_rate', 'user_gini', 'streamer_gini']:
        if key in CALIBRATION_TARGETS and key in agg_metrics:
            real = CALIBRATION_TARGETS[key]
            sim_mean = agg_metrics[key]['mean']
            error_pct = abs(sim_mean - real) / real * 100 if real > 0 else 0
            calibration_comparison[key] = {
                'real': real,
                'sim': sim_mean,
                'error_pct': error_pct
            }
    
    if verbose:
        print("\nCalibration Comparison:")
        for key, comp in calibration_comparison.items():
            status = "✓" if comp['error_pct'] < 20 else "✗"
            print(f"  {status} {key}: Real={comp['real']:.4f}, Sim={comp['sim']:.4f}, Error={comp['error_pct']:.1f}%")
    
    # Policy preview
    print("\nPolicy Preview (50 simulations each)...")
    policy_results = {}
    policies = {
        'random': RandomPolicy(rng=np.random.default_rng(42)),
        'greedy': GreedyPolicy(),
        'round_robin': RoundRobinPolicy(),
    }
    
    for name, policy in policies.items():
        results = []
        for i in tqdm(range(50), desc=f"  {name}"):
            config_i = SimConfig(**{k: v for k, v in config.__dict__.items() if k != 'seed'}, seed=42 + i)
            sim = GiftLiveSimulator(config_i)
            metrics = sim.run_simulation(policy, n_rounds=50, users_per_round=200)
            results.append(metrics)
        
        policy_results[name] = {
            'revenue_mean': float(np.mean([r['total_revenue'] for r in results])),
            'revenue_std': float(np.std([r['total_revenue'] for r in results])),
            'gini_mean': float(np.mean([r['streamer_gini'] for r in results])),
            'gini_std': float(np.std([r['streamer_gini'] for r in results])),
            'top_10_share_mean': float(np.mean([r['top_10_streamer_share'] for r in results])),
        }
        
        print(f"    {name}: Revenue={policy_results[name]['revenue_mean']:.0f}, Gini={policy_results[name]['gini_mean']:.4f}")
    
    # Externality sweep
    print("\nExternality Sweep...")
    gamma_values = [0, 0.01, 0.05, 0.1, 0.2]
    externality_results = {}
    
    for gamma in gamma_values:
        results = []
        for i in range(30):
            config_i = SimConfig(
                **{k: v for k, v in config.__dict__.items() if k not in ['seed', 'gamma']},
                gamma=gamma,
                seed=42 + i
            )
            sim = GiftLiveSimulator(config_i)
            policy = GreedyPolicy()
            metrics = sim.run_simulation(policy, n_rounds=50, users_per_round=200)
            results.append(metrics)
        
        externality_results[gamma] = {
            'revenue_mean': float(np.mean([r['total_revenue'] for r in results])),
            'gini_mean': float(np.mean([r['streamer_gini'] for r in results])),
        }
        print(f"    gamma={gamma}: Revenue={externality_results[gamma]['revenue_mean']:.0f}")
    
    # Compile results
    results = {
        'calibration': calibration_comparison,
        'aggregated_metrics': agg_metrics,
        'policy_preview': policy_results,
        'externality_sweep': externality_results,
        'config': config.__dict__,
    }
    
    return results


def plot_mvp03_figures(results: Dict):
    """Generate MVP-0.3 figures."""
    print("\nGenerating MVP-0.3 figures...")
    
    # Fig1: Calibration Comparison
    fig, ax = plt.subplots(figsize=(6, 5))
    metrics = ['gift_rate', 'user_gini', 'streamer_gini']
    x = np.arange(len(metrics))
    width = 0.35
    
    real_vals = [results['calibration'][m]['real'] for m in metrics]
    sim_vals = [results['calibration'][m]['sim'] for m in metrics]
    
    ax.bar(x - width/2, real_vals, width, label='Real Data', color='steelblue')
    ax.bar(x + width/2, sim_vals, width, label='Simulated', color='coral')
    ax.set_ylabel('Value')
    ax.set_title('Calibration: Real vs Simulated')
    ax.set_xticks(x)
    ax.set_xticklabels(['Gift Rate', 'User Gini', 'Streamer Gini'])
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'mvp03_calibration_comparison.png')
    plt.close()
    print(f"  Saved: mvp03_calibration_comparison.png")
    
    # Fig2: Amount Distribution (simulated)
    fig, ax = plt.subplots(figsize=(6, 5))
    # Generate sample amounts from simulator
    config = SimConfig(**results['config'])
    sim = GiftLiveSimulator(config)
    policy = GreedyPolicy()
    _ = sim.run_simulation(policy, n_rounds=50, users_per_round=200)
    amounts = [r['amount'] for r in sim.interaction_log if r['amount'] > 0]
    
    ax.hist(np.log1p(amounts), bins=50, edgecolor='black', alpha=0.7, color='teal')
    ax.set_xlabel('Gift Amount (log scale)')
    ax.set_ylabel('Frequency')
    ax.set_title('Simulated Gift Amount Distribution')
    ax.axvline(np.mean(np.log1p(amounts)), color='red', linestyle='--', label=f'Mean: {np.mean(np.log1p(amounts)):.2f}')
    ax.axvline(np.median(np.log1p(amounts)), color='green', linestyle='--', label=f'Median: {np.median(np.log1p(amounts)):.2f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'mvp03_amount_distribution.png')
    plt.close()
    print(f"  Saved: mvp03_amount_distribution.png")
    
    # Fig3: Lorenz Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # User Lorenz
    user_spending = {}
    for r in sim.interaction_log:
        uid = r['user_id']
        if uid not in user_spending:
            user_spending[uid] = 0
        user_spending[uid] += r['amount']
    
    user_amounts = np.sort(list(user_spending.values()))
    cumsum = np.cumsum(user_amounts)
    cumsum_pct = cumsum / cumsum[-1] * 100 if cumsum[-1] > 0 else cumsum
    x = np.arange(1, len(user_amounts) + 1) / len(user_amounts) * 100
    
    ax.plot(x, cumsum_pct, 'b-', linewidth=2, label='User Gift Distribution')
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1, label='Perfect Equality')
    ax.fill_between(x, cumsum_pct, x, alpha=0.3)
    
    gini = compute_gini(user_amounts)
    ax.annotate(f'Gini = {gini:.4f}', xy=(70, 30), fontsize=14,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Cumulative % of Users')
    ax.set_ylabel('Cumulative % of Gift Amount')
    ax.set_title('Simulated User Lorenz Curve')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'mvp03_lorenz_curve.png')
    plt.close()
    print(f"  Saved: mvp03_lorenz_curve.png")
    
    # Fig4: Policy Preview
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    policies = list(results['policy_preview'].keys())
    revenues = [results['policy_preview'][p]['revenue_mean'] for p in policies]
    ginis = [results['policy_preview'][p]['gini_mean'] for p in policies]
    
    axes[0].bar(policies, revenues, color=['gray', 'steelblue', 'coral'])
    axes[0].set_ylabel('Total Revenue')
    axes[0].set_title('Policy Comparison: Revenue')
    
    axes[1].bar(policies, ginis, color=['gray', 'steelblue', 'coral'])
    axes[1].set_ylabel('Streamer Gini')
    axes[1].set_title('Policy Comparison: Fairness')
    
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'mvp03_policy_preview.png')
    plt.close()
    print(f"  Saved: mvp03_policy_preview.png")
    
    # Fig5: Externality Sweep
    fig, ax = plt.subplots(figsize=(6, 5))
    
    gammas = sorted(results['externality_sweep'].keys())
    revenues = [results['externality_sweep'][g]['revenue_mean'] for g in gammas]
    
    ax.plot(gammas, revenues, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax.set_xlabel('Gamma (Externality Coefficient)')
    ax.set_ylabel('Total Revenue')
    ax.set_title('Externality Effect on Revenue')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'mvp03_externality_sweep.png')
    plt.close()
    print(f"  Saved: mvp03_externality_sweep.png")


# ============================================================
# MVP-2.1: Concave Allocation
# ============================================================

def run_mvp21_concave(n_simulations: int = 100, verbose: bool = True):
    """Run MVP-2.1: Concave allocation experiments."""
    print("\n" + "="*60)
    print("MVP-2.1: Concave Allocation")
    print("="*60)
    
    config = SimConfig(
        n_users=10000,
        n_streamers=500,
        wealth_lognormal_mean=2.5,
        wealth_lognormal_std=1.2,
        wealth_pareto_alpha=1.3,
        wealth_pareto_min=80,
        wealth_pareto_weight=0.05,
        gift_theta0=-6.5,
        gift_theta1=0.4,
        gift_theta2=0.8,
        gift_theta3=0.2,
        gift_theta4=-0.05,
        amount_mu0=0.5,
        amount_mu1=0.9,
        amount_mu2=0.4,
        amount_sigma=1.5,
        gamma=0.02,
        seed=42
    )
    
    # Policy comparison
    print(f"\nRunning policy comparison ({n_simulations} simulations each)...")
    policies = {
        'random': RandomPolicy(rng=np.random.default_rng(42)),
        'greedy': GreedyPolicy(),
        'concave_log': ConcaveLogPolicy(),
        'concave_exp': ConcaveExpPolicy(B=1000),
        'greedy_with_cap': GreedyWithCapPolicy(cap=5000, rng=np.random.default_rng(42)),
    }
    
    policy_results = {}
    
    for name, policy in policies.items():
        results = []
        for i in tqdm(range(n_simulations), desc=f"  {name}"):
            config_i = SimConfig(**{k: v for k, v in config.__dict__.items() if k != 'seed'}, seed=42 + i)
            sim = GiftLiveSimulator(config_i)
            metrics = sim.run_simulation(policy, n_rounds=50, users_per_round=200)
            results.append(metrics)
        
        policy_results[name] = {
            'revenue': {
                'mean': float(np.mean([r['total_revenue'] for r in results])),
                'std': float(np.std([r['total_revenue'] for r in results])),
            },
            'gini': {
                'mean': float(np.mean([r['streamer_gini'] for r in results])),
                'std': float(np.std([r['streamer_gini'] for r in results])),
            },
            'top_10_share': {
                'mean': float(np.mean([r['top_10_streamer_share'] for r in results])),
                'std': float(np.std([r['top_10_streamer_share'] for r in results])),
            },
            'coverage': {
                'mean': float(np.mean([r['coverage'] for r in results])),
                'std': float(np.std([r['coverage'] for r in results])),
            },
            'cold_start_success': {
                'mean': float(np.mean([r['cold_start_success_rate'] for r in results])),
                'std': float(np.std([r['cold_start_success_rate'] for r in results])),
            },
        }
        
        if verbose:
            print(f"    {name}: Revenue={policy_results[name]['revenue']['mean']:.0f}, "
                  f"Gini={policy_results[name]['gini']['mean']:.4f}, "
                  f"Coverage={policy_results[name]['coverage']['mean']:.4f}")
    
    # Compute deltas
    greedy_revenue = policy_results['greedy']['revenue']['mean']
    greedy_gini = policy_results['greedy']['gini']['mean']
    
    for name in policy_results:
        if name != 'greedy':
            policy_results[name]['delta_revenue_pct'] = (
                (policy_results[name]['revenue']['mean'] - greedy_revenue) / greedy_revenue * 100
            )
            policy_results[name]['delta_gini'] = (
                policy_results[name]['gini']['mean'] - greedy_gini
            )
    
    # B parameter sweep for concave_exp
    print("\nB parameter sweep for ConcaveExp...")
    B_values = [100, 500, 1000, 5000, 10000]
    b_sweep_results = {}
    
    for B in B_values:
        results = []
        policy = ConcaveExpPolicy(B=B)
        for i in tqdm(range(50), desc=f"  B={B}"):
            config_i = SimConfig(**{k: v for k, v in config.__dict__.items() if k != 'seed'}, seed=42 + i)
            sim = GiftLiveSimulator(config_i)
            metrics = sim.run_simulation(policy, n_rounds=50, users_per_round=200)
            results.append(metrics)
        
        b_sweep_results[B] = {
            'revenue_mean': float(np.mean([r['total_revenue'] for r in results])),
            'gini_mean': float(np.mean([r['streamer_gini'] for r in results])),
        }
    
    # Scale effect
    print("\nScale effect...")
    n_users_values = [1000, 5000, 10000, 50000]
    scale_results = {'greedy': {}, 'concave_log': {}}
    
    for n_users in n_users_values:
        for policy_name, policy_cls in [('greedy', GreedyPolicy), ('concave_log', ConcaveLogPolicy)]:
            results = []
            for i in range(30):
                config_i = SimConfig(
                    **{k: v for k, v in config.__dict__.items() if k not in ['seed', 'n_users']},
                    n_users=n_users,
                    seed=42 + i
                )
                sim = GiftLiveSimulator(config_i)
                policy = policy_cls()
                metrics = sim.run_simulation(policy, n_rounds=50, users_per_round=min(200, n_users//50))
                results.append(metrics)
            
            scale_results[policy_name][n_users] = float(np.mean([r['total_revenue'] for r in results]))
    
    # Gate-2 decision
    best_concave = 'concave_log'
    delta_revenue = policy_results[best_concave]['delta_revenue_pct']
    delta_gini = policy_results[best_concave]['delta_gini']
    
    if delta_revenue >= 10 and delta_gini < 0:
        gate2_decision = "confirm_allocation"
        dg3_result = "closed_positive"
    elif delta_revenue >= 5:
        gate2_decision = "confirm_allocation_marginal"
        dg3_result = "closed_marginal"
    else:
        gate2_decision = "simplify_to_greedy"
        dg3_result = "closed_negative"
    
    results = {
        'policy_comparison': policy_results,
        'b_sweep': b_sweep_results,
        'scale_effect': scale_results,
        'gate2_decision': gate2_decision,
        'dg3_result': dg3_result,
        'config': config.__dict__,
    }
    
    if verbose:
        print(f"\n=== Gate-2 Decision: {gate2_decision} ===")
        print(f"    Concave Log Δ Revenue: {delta_revenue:.1f}%")
        print(f"    Concave Log Δ Gini: {delta_gini:.4f}")
    
    return results


def plot_mvp21_figures(results: Dict):
    """Generate MVP-2.1 figures."""
    print("\nGenerating MVP-2.1 figures...")
    
    # Fig1: Revenue Comparison
    fig, ax = plt.subplots(figsize=(6, 5))
    policies = list(results['policy_comparison'].keys())
    revenues = [results['policy_comparison'][p]['revenue']['mean'] for p in policies]
    errors = [results['policy_comparison'][p]['revenue']['std'] for p in policies]
    
    colors = ['gray', 'steelblue', 'coral', 'teal', 'purple']
    ax.bar(policies, revenues, yerr=errors, color=colors, capsize=5)
    ax.set_ylabel('Total Revenue')
    ax.set_title('Policy Comparison: Revenue')
    ax.set_xticklabels(policies, rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'mvp21_revenue_comparison.png')
    plt.close()
    print(f"  Saved: mvp21_revenue_comparison.png")
    
    # Fig2: Fairness Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    policies = list(results['policy_comparison'].keys())
    ginis = [results['policy_comparison'][p]['gini']['mean'] for p in policies]
    top10s = [results['policy_comparison'][p]['top_10_share']['mean'] for p in policies]
    
    axes[0].bar(policies, ginis, color=colors)
    axes[0].set_ylabel('Streamer Gini')
    axes[0].set_title('Fairness: Gini Coefficient')
    axes[0].set_xticklabels(policies, rotation=15, ha='right')
    
    axes[1].bar(policies, top10s, color=colors)
    axes[1].set_ylabel('Top-10% Share')
    axes[1].set_title('Fairness: Top-10% Revenue Share')
    axes[1].set_xticklabels(policies, rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'mvp21_fairness_comparison.png')
    plt.close()
    print(f"  Saved: mvp21_fairness_comparison.png")
    
    # Fig3: Pareto Frontier (Revenue vs Gini)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    for i, p in enumerate(policies):
        rev = results['policy_comparison'][p]['revenue']['mean']
        gini = results['policy_comparison'][p]['gini']['mean']
        ax.scatter(rev, gini, s=150, c=colors[i], label=p, edgecolors='black')
    
    ax.set_xlabel('Total Revenue')
    ax.set_ylabel('Streamer Gini (lower = fairer)')
    ax.set_title('Pareto Frontier: Revenue vs Fairness')
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'mvp21_pareto_frontier.png')
    plt.close()
    print(f"  Saved: mvp21_pareto_frontier.png")
    
    # Fig4: B Parameter Sensitivity
    fig, ax = plt.subplots(figsize=(6, 5))
    
    B_values = sorted(results['b_sweep'].keys())
    revenues = [results['b_sweep'][b]['revenue_mean'] for b in B_values]
    ginis = [results['b_sweep'][b]['gini_mean'] for b in B_values]
    
    ax.plot(B_values, revenues, 'o-', linewidth=2, markersize=8, color='steelblue', label='Revenue')
    ax.set_xlabel('B (Saturation Parameter)')
    ax.set_ylabel('Revenue', color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')
    
    ax2 = ax.twinx()
    ax2.plot(B_values, ginis, 's--', linewidth=2, markersize=8, color='coral', label='Gini')
    ax2.set_ylabel('Gini', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    
    ax.set_title('ConcaveExp: B Parameter Sensitivity')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85))
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'mvp21_param_sensitivity.png')
    plt.close()
    print(f"  Saved: mvp21_param_sensitivity.png")
    
    # Fig5: Scale Effect
    fig, ax = plt.subplots(figsize=(6, 5))
    
    n_users_values = sorted(results['scale_effect']['greedy'].keys())
    greedy_revs = [results['scale_effect']['greedy'][n] for n in n_users_values]
    concave_revs = [results['scale_effect']['concave_log'][n] for n in n_users_values]
    delta_revs = [c - g for c, g in zip(concave_revs, greedy_revs)]
    
    ax.plot(n_users_values, delta_revs, 'o-', linewidth=2, markersize=8, color='teal')
    ax.set_xlabel('Number of Users')
    ax.set_ylabel('Revenue Δ (Concave - Greedy)')
    ax.set_title('Scale Effect: Concave vs Greedy')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'mvp21_scale_effect.png')
    plt.close()
    print(f"  Saved: mvp21_scale_effect.png")
    
    # Fig6: Lorenz by Policy
    fig, ax = plt.subplots(figsize=(6, 5))
    
    config = SimConfig(**results['config'])
    
    for i, (name, policy_cls) in enumerate([
        ('greedy', GreedyPolicy),
        ('concave_log', ConcaveLogPolicy),
    ]):
        sim = GiftLiveSimulator(config)
        policy = policy_cls()
        _ = sim.run_simulation(policy, n_rounds=50, users_per_round=200)
        
        streamer_revenue = {}
        for r in sim.interaction_log:
            sid = r['streamer_id']
            if sid not in streamer_revenue:
                streamer_revenue[sid] = 0
            streamer_revenue[sid] += r['amount']
        
        revenues = np.sort(list(streamer_revenue.values()))
        cumsum = np.cumsum(revenues)
        cumsum_pct = cumsum / cumsum[-1] * 100 if cumsum[-1] > 0 else cumsum
        x = np.arange(1, len(revenues) + 1) / len(revenues) * 100
        
        ax.plot(x, cumsum_pct, linewidth=2, label=f'{name} (Gini={compute_gini(revenues):.3f})', color=colors[i+1])
    
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1, label='Perfect Equality')
    ax.set_xlabel('Cumulative % of Streamers')
    ax.set_ylabel('Cumulative % of Revenue')
    ax.set_title('Streamer Lorenz Curve by Policy')
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'mvp21_lorenz_by_policy.png')
    plt.close()
    print(f"  Saved: mvp21_lorenz_by_policy.png")


# ============================================================
# MVP-2.2: Cold-start Constraints
# ============================================================

def run_mvp22_coldstart(n_simulations: int = 100, verbose: bool = True):
    """Run MVP-2.2: Cold-start and fairness constraint experiments."""
    print("\n" + "="*60)
    print("MVP-2.2: Cold-start / Fairness Constraints")
    print("="*60)
    
    config = SimConfig(
        n_users=10000,
        n_streamers=500,
        new_streamer_ratio=0.2,  # 20% are new streamers
        wealth_lognormal_mean=2.5,
        wealth_lognormal_std=1.2,
        wealth_pareto_alpha=1.3,
        wealth_pareto_min=80,
        wealth_pareto_weight=0.05,
        gift_theta0=-6.5,
        gift_theta1=0.4,
        gift_theta2=0.8,
        gift_theta3=0.2,
        gift_theta4=-0.05,
        amount_mu0=0.5,
        amount_mu1=0.9,
        amount_mu2=0.4,
        amount_sigma=1.5,
        gamma=0.02,
        seed=42
    )
    
    # Define constraint configurations
    constraint_configs = {
        'no_constraint': ConstraintConfig(),
        'soft_cold_start': ConstraintConfig(
            enable_cold_start=True,
            min_allocation_per_new=10,
            cold_start_enforce="soft",
            lambda_cold_start=0.5,
        ),
        'soft_head_cap': ConstraintConfig(
            enable_head_cap=True,
            max_share_top10=0.5,
            head_cap_enforce="soft",
            lambda_head_cap=0.5,
        ),
        'soft_all': ConstraintConfig(
            enable_cold_start=True,
            enable_head_cap=True,
            enable_diversity=True,
            min_allocation_per_new=10,
            max_share_top10=0.5,
            min_coverage=0.3,
            cold_start_enforce="soft",
            head_cap_enforce="soft",
            diversity_enforce="soft",
            lambda_cold_start=0.5,
            lambda_head_cap=0.5,
            lambda_diversity=0.5,
        ),
        'hard_cold_start': ConstraintConfig(
            enable_cold_start=True,
            min_allocation_per_new=10,
            cold_start_enforce="hard",
        ),
    }
    
    # Base policy (concave log)
    base_policy = ConcaveLogPolicy()
    
    # Run constraint comparison
    print(f"\nRunning constraint comparison ({n_simulations} simulations each)...")
    constraint_results = {}
    
    for name, constraint_cfg in constraint_configs.items():
        results = []
        
        for i in tqdm(range(n_simulations), desc=f"  {name}"):
            config_i = SimConfig(**{k: v for k, v in config.__dict__.items() if k != 'seed'}, seed=42 + i)
            sim = GiftLiveSimulator(config_i)
            
            if name == 'no_constraint':
                policy = base_policy
            else:
                policy = ConstrainedAllocationPolicy(
                    base_policy=base_policy,
                    constraint_config=constraint_cfg,
                    rng=np.random.default_rng(42 + i)
                )
            
            metrics = sim.run_simulation(policy, n_rounds=50, users_per_round=200)
            results.append(metrics)
        
        constraint_results[name] = {
            'revenue': {
                'mean': float(np.mean([r['total_revenue'] for r in results])),
                'std': float(np.std([r['total_revenue'] for r in results])),
            },
            'cold_start_success': {
                'mean': float(np.mean([r['cold_start_success_rate'] for r in results])),
                'std': float(np.std([r['cold_start_success_rate'] for r in results])),
            },
            'coverage': {
                'mean': float(np.mean([r['coverage'] for r in results])),
                'std': float(np.std([r['coverage'] for r in results])),
            },
            'gini': {
                'mean': float(np.mean([r['streamer_gini'] for r in results])),
                'std': float(np.std([r['streamer_gini'] for r in results])),
            },
            'top_10_share': {
                'mean': float(np.mean([r['top_10_streamer_share'] for r in results])),
                'std': float(np.std([r['top_10_streamer_share'] for r in results])),
            },
        }
        
        if verbose:
            print(f"    {name}: Revenue={constraint_results[name]['revenue']['mean']:.0f}, "
                  f"ColdStart={constraint_results[name]['cold_start_success']['mean']:.4f}")
    
    # Compute deltas vs no_constraint
    baseline_revenue = constraint_results['no_constraint']['revenue']['mean']
    baseline_cold_start = constraint_results['no_constraint']['cold_start_success']['mean']
    
    for name in constraint_results:
        if name != 'no_constraint':
            constraint_results[name]['revenue_loss_pct'] = (
                (baseline_revenue - constraint_results[name]['revenue']['mean']) / baseline_revenue * 100
            )
            constraint_results[name]['cold_start_improvement_pct'] = (
                (constraint_results[name]['cold_start_success']['mean'] - baseline_cold_start) / (baseline_cold_start + 1e-6) * 100
            )
    
    # Constraint strength sweep
    print("\nConstraint strength sweep (min_allocation_per_new)...")
    strength_values = [5, 10, 20, 50, 100]
    strength_results = {}
    
    for strength in strength_values:
        results = []
        constraint_cfg = ConstraintConfig(
            enable_cold_start=True,
            min_allocation_per_new=strength,
            cold_start_enforce="soft",
            lambda_cold_start=0.5,
        )
        
        for i in tqdm(range(50), desc=f"  strength={strength}"):
            config_i = SimConfig(**{k: v for k, v in config.__dict__.items() if k != 'seed'}, seed=42 + i)
            sim = GiftLiveSimulator(config_i)
            policy = ConstrainedAllocationPolicy(
                base_policy=base_policy,
                constraint_config=constraint_cfg,
                rng=np.random.default_rng(42 + i)
            )
            metrics = sim.run_simulation(policy, n_rounds=50, users_per_round=200)
            results.append(metrics)
        
        strength_results[strength] = {
            'revenue_mean': float(np.mean([r['total_revenue'] for r in results])),
            'cold_start_success_mean': float(np.mean([r['cold_start_success_rate'] for r in results])),
        }
    
    # Long-term simulation (coverage over time)
    print("\nLong-term simulation (200 rounds)...")
    coverage_over_time = {
        'no_constraint': [],
        'soft_cold_start': [],
    }
    
    for name in coverage_over_time.keys():
        sim = GiftLiveSimulator(config)
        
        if name == 'no_constraint':
            policy = base_policy
        else:
            constraint_cfg = ConstraintConfig(
                enable_cold_start=True,
                min_allocation_per_new=10,
                cold_start_enforce="soft",
                lambda_cold_start=0.5,
            )
            policy = ConstrainedAllocationPolicy(
                base_policy=base_policy,
                constraint_config=constraint_cfg,
                rng=np.random.default_rng(42)
            )
        
        sim.reset()
        for round_idx in range(200):
            users = sim.user_pool.sample_users(200)
            allocations = policy.allocate(users, sim)
            sim.simulate_batch(users, allocations)
            
            # Compute coverage at this round
            streamer_revenue = {}
            for r in sim.interaction_log:
                sid = r['streamer_id']
                if sid not in streamer_revenue:
                    streamer_revenue[sid] = 0
                streamer_revenue[sid] += r['amount']
            
            n_with_revenue = sum(1 for v in streamer_revenue.values() if v > 0)
            coverage = n_with_revenue / config.n_streamers
            coverage_over_time[name].append(coverage)
    
    # Decision
    best_constraint = 'soft_cold_start'
    revenue_loss = constraint_results[best_constraint].get('revenue_loss_pct', 0)
    cold_start_improvement = constraint_results[best_constraint].get('cold_start_improvement_pct', 0)
    
    if revenue_loss < 5 and cold_start_improvement > 20:
        decision = "adopt_constraint"
    elif revenue_loss < 10:
        decision = "adopt_constraint_carefully"
    else:
        decision = "reject_constraint"
    
    results = {
        'constraint_comparison': constraint_results,
        'strength_sweep': strength_results,
        'coverage_over_time': coverage_over_time,
        'decision': decision,
        'config': config.__dict__,
    }
    
    if verbose:
        print(f"\n=== Decision: {decision} ===")
        print(f"    Revenue Loss: {revenue_loss:.1f}%")
        print(f"    Cold Start Improvement: {cold_start_improvement:.1f}%")
    
    return results


def plot_mvp22_figures(results: Dict):
    """Generate MVP-2.2 figures."""
    print("\nGenerating MVP-2.2 figures...")
    
    # Fig1: Constraint Effect
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    policies = list(results['constraint_comparison'].keys())
    revenues = [results['constraint_comparison'][p]['revenue']['mean'] for p in policies]
    cold_starts = [results['constraint_comparison'][p]['cold_start_success']['mean'] for p in policies]
    
    colors = ['gray', 'steelblue', 'coral', 'teal', 'purple']
    
    axes[0].bar(policies, revenues, color=colors)
    axes[0].set_ylabel('Total Revenue')
    axes[0].set_title('Constraint Effect: Revenue')
    axes[0].set_xticklabels(policies, rotation=20, ha='right')
    
    axes[1].bar(policies, cold_starts, color=colors)
    axes[1].set_ylabel('Cold Start Success Rate')
    axes[1].set_title('Constraint Effect: Cold Start Success')
    axes[1].set_xticklabels(policies, rotation=20, ha='right')
    
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'mvp22_constraint_effect.png')
    plt.close()
    print(f"  Saved: mvp22_constraint_effect.png")
    
    # Fig2: Tradeoff Scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    
    baseline_revenue = results['constraint_comparison']['no_constraint']['revenue']['mean']
    baseline_cold_start = results['constraint_comparison']['no_constraint']['cold_start_success']['mean']
    
    for i, p in enumerate(policies):
        if p == 'no_constraint':
            continue
        rev_loss = results['constraint_comparison'][p].get('revenue_loss_pct', 0)
        cold_start_delta = (
            results['constraint_comparison'][p]['cold_start_success']['mean'] - baseline_cold_start
        ) * 100
        ax.scatter(rev_loss, cold_start_delta, s=150, c=colors[i], label=p, edgecolors='black')
    
    ax.axvline(5, color='red', linestyle='--', alpha=0.5, label='5% Loss Threshold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Revenue Loss %')
    ax.set_ylabel('Cold Start Success Δ (pp)')
    ax.set_title('Trade-off: Revenue Loss vs Cold Start Improvement')
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'mvp22_tradeoff_scatter.png')
    plt.close()
    print(f"  Saved: mvp22_tradeoff_scatter.png")
    
    # Fig3: Constraint Strength
    fig, ax = plt.subplots(figsize=(6, 5))
    
    strengths = sorted(results['strength_sweep'].keys())
    revenues = [results['strength_sweep'][s]['revenue_mean'] for s in strengths]
    cold_starts = [results['strength_sweep'][s]['cold_start_success_mean'] for s in strengths]
    
    ax.plot(strengths, revenues, 'o-', linewidth=2, markersize=8, color='steelblue', label='Revenue')
    ax.set_xlabel('Min Allocation per New Streamer')
    ax.set_ylabel('Revenue', color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')
    
    ax2 = ax.twinx()
    ax2.plot(strengths, cold_starts, 's--', linewidth=2, markersize=8, color='coral', label='Cold Start Success')
    ax2.set_ylabel('Cold Start Success Rate', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    
    ax.set_title('Constraint Strength Sensitivity')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85))
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'mvp22_constraint_strength.png')
    plt.close()
    print(f"  Saved: mvp22_constraint_strength.png")
    
    # Fig4: Coverage Over Time
    fig, ax = plt.subplots(figsize=(6, 5))
    
    for i, (name, coverage) in enumerate(results['coverage_over_time'].items()):
        ax.plot(range(len(coverage)), coverage, linewidth=2, label=name, color=['gray', 'steelblue'][i])
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative Coverage')
    ax.set_title('Coverage Over Time (200 Rounds)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'mvp22_coverage_over_time.png')
    plt.close()
    print(f"  Saved: mvp22_coverage_over_time.png")
    
    # Fig5: Ecosystem Diversity
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Compute entropy as diversity metric
    def entropy(revenues):
        revenues = np.array(revenues)
        revenues = revenues[revenues > 0]
        if len(revenues) == 0:
            return 0
        p = revenues / revenues.sum()
        return -np.sum(p * np.log(p + 1e-10))
    
    config = SimConfig(**results['config'])
    diversities = {}
    
    for name in ['no_constraint', 'soft_cold_start', 'soft_all']:
        sim = GiftLiveSimulator(config)
        
        if name == 'no_constraint':
            policy = ConcaveLogPolicy()
        else:
            constraint_cfg = ConstraintConfig(
                enable_cold_start=True,
                min_allocation_per_new=10,
                cold_start_enforce="soft",
                lambda_cold_start=0.5,
            ) if name == 'soft_cold_start' else ConstraintConfig(
                enable_cold_start=True,
                enable_head_cap=True,
                enable_diversity=True,
                min_allocation_per_new=10,
                max_share_top10=0.5,
                min_coverage=0.3,
                cold_start_enforce="soft",
                head_cap_enforce="soft",
                diversity_enforce="soft",
                lambda_cold_start=0.5,
                lambda_head_cap=0.5,
                lambda_diversity=0.5,
            )
            policy = ConstrainedAllocationPolicy(
                base_policy=ConcaveLogPolicy(),
                constraint_config=constraint_cfg,
                rng=np.random.default_rng(42)
            )
        
        _ = sim.run_simulation(policy, n_rounds=50, users_per_round=200)
        
        streamer_revenue = {}
        for r in sim.interaction_log:
            sid = r['streamer_id']
            if sid not in streamer_revenue:
                streamer_revenue[sid] = 0
            streamer_revenue[sid] += r['amount']
        
        diversities[name] = entropy(list(streamer_revenue.values()))
    
    names = list(diversities.keys())
    values = [diversities[n] for n in names]
    
    ax.bar(names, values, color=['gray', 'steelblue', 'teal'])
    ax.set_ylabel('Ecosystem Diversity (Entropy)')
    ax.set_title('Ecosystem Diversity by Policy')
    ax.set_xticklabels(names, rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'mvp22_ecosystem_diversity.png')
    plt.close()
    print(f"  Saved: mvp22_ecosystem_diversity.png")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Run simulator experiments')
    parser.add_argument('--mvp', type=str, choices=['0.3', '2.1', '2.2'], 
                        help='Run specific MVP')
    parser.add_argument('--all', action='store_true', help='Run all MVPs')
    parser.add_argument('--n_sim', type=int, default=100, help='Number of simulations')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer simulations)')
    args = parser.parse_args()
    
    if args.quick:
        args.n_sim = 20
    
    # Ensure directories exist
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    if args.all or args.mvp == '0.3':
        results_03 = run_mvp03_calibration(n_simulations=args.n_sim)
        plot_mvp03_figures(results_03)
        all_results['mvp03'] = results_03
        
        # Save results
        with open(RESULTS_DIR / 'simulator_v1_20260108.json', 'w') as f:
            json.dump(results_03, f, indent=2, default=str)
        print(f"\nSaved: simulator_v1_20260108.json")
    
    if args.all or args.mvp == '2.1':
        results_21 = run_mvp21_concave(n_simulations=args.n_sim)
        plot_mvp21_figures(results_21)
        all_results['mvp21'] = results_21
        
        # Save results
        with open(RESULTS_DIR / 'concave_allocation_20260108.json', 'w') as f:
            json.dump(results_21, f, indent=2, default=str)
        print(f"\nSaved: concave_allocation_20260108.json")
    
    if args.all or args.mvp == '2.2':
        results_22 = run_mvp22_coldstart(n_simulations=args.n_sim)
        plot_mvp22_figures(results_22)
        all_results['mvp22'] = results_22
        
        # Save results
        with open(RESULTS_DIR / 'coldstart_constraint_20260108.json', 'w') as f:
            json.dump(results_22, f, indent=2, default=str)
        print(f"\nSaved: coldstart_constraint_20260108.json")
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    main()
