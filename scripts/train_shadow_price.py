#!/usr/bin/env python3
"""
Shadow Price Allocation Experiments
MVP-5.2: EXP-20260109-gift-allocation-16

This script runs four experiments:
1. Policy Comparison: greedy vs greedy_with_rules vs shadow_price_core vs shadow_price_all
2. Learning Rate Sweep: eta ∈ {0.001, 0.01, 0.05, 0.1, 0.2}
3. Constraint Ablation: progressively add constraints
4. Lambda Convergence: track dual variable convergence

Outputs:
- Results JSON: gift_allocation/results/shadow_price_20260109.json
- Figures: gift_allocation/img/mvp52_*.png
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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.simulator import (
    SimConfig,
    GiftLiveSimulator,
    GreedyPolicy,
    ShadowPriceConfig,
    ShadowPriceAllocator,
    GreedyWithRulesPolicy,
    run_shadow_price_simulation,
)

# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = PROJECT_ROOT / "gift_allocation" / "results"
IMG_DIR = PROJECT_ROOT / "gift_allocation" / "img"
SEED = 42

# Simulation config
SIM_CONFIG = dict(
    n_users=10000,
    n_streamers=500,
    seed=SEED,
    enable_capacity=True,
    capacity_top10=100,
    capacity_middle=50,
    capacity_tail=20,
    new_streamer_ratio=0.2,
)

# Experiment config
EXP_CONFIG = dict(
    n_rounds=50,
    users_per_round=200,
    n_simulations=50,  # Reduced for faster testing
)

# ============================================================
# Experiment Functions
# ============================================================

def run_policy_comparison(n_simulations: int = 50) -> Dict:
    """Experiment 1: Compare different allocation policies."""
    print("\n" + "=" * 60)
    print("Experiment 1: Policy Comparison")
    print("=" * 60)
    
    results = {
        'greedy': [],
        'greedy_with_rules': [],
        'shadow_price_core': [],
        'shadow_price_all': [],
    }
    
    for sim_idx in tqdm(range(n_simulations), desc="Simulations"):
        seed = SEED + sim_idx
        config = SimConfig(**{**SIM_CONFIG, 'seed': seed})
        sim = GiftLiveSimulator(config)
        
        # 1. Greedy baseline
        sim.reset()
        greedy = GreedyPolicy()
        m = sim.run_simulation(greedy, n_rounds=EXP_CONFIG['n_rounds'], 
                               users_per_round=EXP_CONFIG['users_per_round'])
        results['greedy'].append(m)
        
        # 2. Greedy with rules
        sim.reset()
        greedy_rules = GreedyWithRulesPolicy(lambda_cold_start=0.5, freq_penalty=0.3)
        m = sim.run_simulation(greedy_rules, n_rounds=EXP_CONFIG['n_rounds'], 
                               users_per_round=EXP_CONFIG['users_per_round'])
        results['greedy_with_rules'].append(m)
        
        # 3. Shadow Price (core: capacity + cold_start + head_cap)
        sim.reset()
        sp_config = ShadowPriceConfig(
            enable_capacity=True,
            enable_cold_start=True,
            enable_head_cap=True,
            enable_whale_spread=False,
            enable_frequency=False,
            eta=0.05
        )
        sp_core = ShadowPriceAllocator(sp_config)
        m = run_shadow_price_simulation(sim, sp_core, n_rounds=EXP_CONFIG['n_rounds'],
                                        users_per_round=EXP_CONFIG['users_per_round'])
        results['shadow_price_core'].append(m)
        
        # 4. Shadow Price (all constraints)
        sim.reset()
        sp_config_all = ShadowPriceConfig(
            enable_capacity=True,
            enable_cold_start=True,
            enable_head_cap=True,
            enable_whale_spread=True,
            enable_frequency=True,
            eta=0.05
        )
        sp_all = ShadowPriceAllocator(sp_config_all)
        m = run_shadow_price_simulation(sim, sp_all, n_rounds=EXP_CONFIG['n_rounds'],
                                        users_per_round=EXP_CONFIG['users_per_round'])
        results['shadow_price_all'].append(m)
    
    # Aggregate results
    summary = {}
    for policy, metrics_list in results.items():
        summary[policy] = {
            'revenue': {
                'mean': float(np.mean([m['total_revenue'] for m in metrics_list])),
                'std': float(np.std([m['total_revenue'] for m in metrics_list])),
            },
            'gini': {
                'mean': float(np.mean([m['streamer_gini'] for m in metrics_list])),
                'std': float(np.std([m['streamer_gini'] for m in metrics_list])),
            },
            'cold_start_success': {
                'mean': float(np.mean([m['cold_start_success_rate'] for m in metrics_list])),
                'std': float(np.std([m['cold_start_success_rate'] for m in metrics_list])),
            },
            'top_10_share': {
                'mean': float(np.mean([m['top_10_streamer_share'] for m in metrics_list])),
                'std': float(np.std([m['top_10_streamer_share'] for m in metrics_list])),
            },
        }
        
        # Add constraint metrics for shadow price policies
        if 'capacity_satisfy_rate' in metrics_list[0]:
            summary[policy]['capacity_satisfy_rate'] = {
                'mean': float(np.mean([m['capacity_satisfy_rate'] for m in metrics_list])),
                'std': float(np.std([m['capacity_satisfy_rate'] for m in metrics_list])),
            }
    
    print("\nPolicy Comparison Summary:")
    greedy_rev = summary['greedy']['revenue']['mean']
    for policy, data in summary.items():
        rev = data['revenue']['mean']
        delta = (rev - greedy_rev) / greedy_rev * 100
        print(f"  {policy}: Revenue={rev:.2f} (Δ{delta:+.2f}%), "
              f"ColdStart={data['cold_start_success']['mean']:.3f}")
    
    return {'policy_comparison': summary}


def run_lr_sweep(n_simulations: int = 30) -> Dict:
    """Experiment 2: Learning rate sensitivity analysis."""
    print("\n" + "=" * 60)
    print("Experiment 2: Learning Rate Sweep")
    print("=" * 60)
    
    eta_values = [0.001, 0.01, 0.05, 0.1, 0.2]
    results = {str(eta): [] for eta in eta_values}
    
    for eta in tqdm(eta_values, desc="LR Sweep"):
        for sim_idx in range(n_simulations):
            seed = SEED + sim_idx
            config = SimConfig(**{**SIM_CONFIG, 'seed': seed})
            sim = GiftLiveSimulator(config)
            
            sp_config = ShadowPriceConfig(
                enable_capacity=True,
                enable_cold_start=True,
                enable_head_cap=True,
                enable_whale_spread=False,
                enable_frequency=False,
                eta=eta
            )
            allocator = ShadowPriceAllocator(sp_config)
            m = run_shadow_price_simulation(sim, allocator, n_rounds=EXP_CONFIG['n_rounds'],
                                            users_per_round=EXP_CONFIG['users_per_round'])
            results[str(eta)].append(m)
    
    # Aggregate
    summary = {}
    for eta, metrics_list in results.items():
        summary[eta] = {
            'revenue_mean': float(np.mean([m['total_revenue'] for m in metrics_list])),
            'revenue_std': float(np.std([m['total_revenue'] for m in metrics_list])),
            'cold_start_mean': float(np.mean([m['cold_start_success_rate'] for m in metrics_list])),
            'capacity_satisfy_mean': float(np.mean([m.get('capacity_satisfy_rate', 1.0) for m in metrics_list])),
            'lambda_stability': float(np.mean([
                np.mean(list(m.get('lambda_stability', {}).values())) 
                for m in metrics_list if 'lambda_stability' in m
            ])) if metrics_list else 0,
        }
    
    print("\nLR Sweep Summary:")
    for eta, data in summary.items():
        print(f"  η={eta}: Revenue={data['revenue_mean']:.2f}, "
              f"ColdStart={data['cold_start_mean']:.3f}, "
              f"Stability={data['lambda_stability']:.4f}")
    
    return {'lr_sweep': summary}


def run_constraint_ablation(n_simulations: int = 30) -> Dict:
    """Experiment 3: Constraint ablation study."""
    print("\n" + "=" * 60)
    print("Experiment 3: Constraint Ablation")
    print("=" * 60)
    
    constraint_combos = [
        (['capacity'], 'capacity_only'),
        (['capacity', 'cold_start'], '+cold_start'),
        (['capacity', 'cold_start', 'head_cap'], '+head_cap'),
        (['capacity', 'cold_start', 'head_cap', 'whale_spread'], '+whale_spread'),
        (['capacity', 'cold_start', 'head_cap', 'whale_spread', 'frequency'], '+frequency'),
    ]
    
    results = {name: [] for _, name in constraint_combos}
    
    for constraints, name in tqdm(constraint_combos, desc="Ablation"):
        for sim_idx in range(n_simulations):
            seed = SEED + sim_idx
            config = SimConfig(**{**SIM_CONFIG, 'seed': seed})
            sim = GiftLiveSimulator(config)
            
            allocator = ShadowPriceAllocator(
                ShadowPriceConfig(eta=0.05),
                constraints=constraints
            )
            m = run_shadow_price_simulation(sim, allocator, n_rounds=EXP_CONFIG['n_rounds'],
                                            users_per_round=EXP_CONFIG['users_per_round'])
            results[name].append(m)
    
    # Get greedy baseline for delta calculation
    config = SimConfig(**{**SIM_CONFIG, 'seed': SEED})
    sim = GiftLiveSimulator(config)
    greedy_revs = []
    for sim_idx in range(n_simulations):
        seed = SEED + sim_idx
        config = SimConfig(**{**SIM_CONFIG, 'seed': seed})
        sim = GiftLiveSimulator(config)
        sim.reset()
        greedy = GreedyPolicy()
        m = sim.run_simulation(greedy, n_rounds=EXP_CONFIG['n_rounds'], 
                               users_per_round=EXP_CONFIG['users_per_round'])
        greedy_revs.append(m['total_revenue'])
    greedy_mean = float(np.mean(greedy_revs))
    
    # Aggregate
    summary = {}
    for name, metrics_list in results.items():
        rev_mean = float(np.mean([m['total_revenue'] for m in metrics_list]))
        summary[name] = {
            'revenue_mean': rev_mean,
            'revenue_std': float(np.std([m['total_revenue'] for m in metrics_list])),
            'delta_vs_greedy_pct': (rev_mean - greedy_mean) / greedy_mean * 100,
            'cold_start_mean': float(np.mean([m['cold_start_success_rate'] for m in metrics_list])),
            'capacity_satisfy_mean': float(np.mean([m.get('capacity_satisfy_rate', 1.0) for m in metrics_list])),
        }
    
    print("\nConstraint Ablation Summary:")
    for name, data in summary.items():
        print(f"  {name}: Revenue={data['revenue_mean']:.2f} "
              f"(Δ{data['delta_vs_greedy_pct']:+.2f}%), "
              f"ColdStart={data['cold_start_mean']:.3f}")
    
    return {'constraint_ablation': summary, 'greedy_baseline_revenue': greedy_mean}


def run_lambda_convergence(n_simulations: int = 10) -> Dict:
    """Experiment 4: Lambda convergence analysis."""
    print("\n" + "=" * 60)
    print("Experiment 4: Lambda Convergence")
    print("=" * 60)
    
    n_rounds = 100  # Longer run for convergence
    
    lambda_histories = []
    
    for sim_idx in tqdm(range(n_simulations), desc="Lambda Convergence"):
        seed = SEED + sim_idx
        config = SimConfig(**{**SIM_CONFIG, 'seed': seed})
        sim = GiftLiveSimulator(config)
        
        sp_config = ShadowPriceConfig(
            enable_capacity=True,
            enable_cold_start=True,
            enable_head_cap=True,
            enable_whale_spread=True,
            enable_frequency=True,
            eta=0.05
        )
        allocator = ShadowPriceAllocator(sp_config)
        m = run_shadow_price_simulation(sim, allocator, n_rounds=n_rounds,
                                        users_per_round=EXP_CONFIG['users_per_round'],
                                        track_lambda=True)
        
        if 'lambda_history' in m:
            lambda_histories.append(m['lambda_history'])
    
    # Average lambda values across simulations
    avg_lambda = defaultdict(list)
    if lambda_histories:
        n_rounds_actual = len(lambda_histories[0])
        for round_idx in range(n_rounds_actual):
            for constraint in lambda_histories[0][0].keys():
                values = [
                    h[round_idx][constraint] 
                    for h in lambda_histories 
                    if round_idx < len(h)
                ]
                avg_lambda[constraint].append(float(np.mean(values)))
    
    # Compute stability (last 50 rounds)
    stability = {}
    for constraint, values in avg_lambda.items():
        if len(values) >= 50:
            last_50 = values[-50:]
            mean_val = np.mean(last_50)
            std_val = np.std(last_50)
            stability[constraint] = float(std_val / (mean_val + 1e-8))
        else:
            stability[constraint] = 0.0
    
    # Get final values
    final_lambdas = {}
    for constraint, values in avg_lambda.items():
        final_lambdas[constraint] = values[-1] if values else 0.0
    
    print("\nLambda Convergence Summary:")
    print("  Final lambda values:")
    for c, v in final_lambdas.items():
        print(f"    {c}: {v:.4f} (stability: {stability.get(c, 0):.4f})")
    
    return {
        'lambda_convergence': {
            'history': {k: v for k, v in avg_lambda.items()},
            'final_lambdas': final_lambdas,
            'stability': stability,
        }
    }


# ============================================================
# Visualization Functions
# ============================================================

def plot_revenue_comparison(results: Dict, output_path: Path):
    """Fig 1: Bar chart comparing revenue across policies."""
    plt.figure(figsize=(8, 6))
    
    policy_comparison = results.get('policy_comparison', {})
    if not policy_comparison:
        print("No policy comparison data for Fig 1")
        return
    
    policies = list(policy_comparison.keys())
    greedy_rev = policy_comparison['greedy']['revenue']['mean']
    
    revenues = [policy_comparison[p]['revenue']['mean'] for p in policies]
    stds = [policy_comparison[p]['revenue']['std'] for p in policies]
    deltas = [(r - greedy_rev) / greedy_rev * 100 for r in revenues]
    
    x = np.arange(len(policies))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Revenue bars
    bars1 = ax1.bar(x - width/2, revenues, width, yerr=stds, label='Revenue', 
                    color='steelblue', capsize=3)
    ax1.set_xlabel('Policy')
    ax1.set_ylabel('Total Revenue', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # Delta percentage on secondary axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, deltas, width, label='Δ vs Greedy (%)', 
                    color='coral', alpha=0.7)
    ax2.set_ylabel('Δ vs Greedy (%)', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.replace('_', '\n') for p in policies], rotation=0)
    ax1.set_title('Revenue Comparison Across Policies')
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_constraint_satisfaction(results: Dict, output_path: Path):
    """Fig 2: Heatmap of constraint satisfaction rates."""
    plt.figure(figsize=(10, 6))
    
    policy_comparison = results.get('policy_comparison', {})
    if not policy_comparison:
        print("No policy comparison data for Fig 2")
        return
    
    policies = list(policy_comparison.keys())
    constraints = ['cold_start_success', 'top_10_share', 'capacity_satisfy_rate']
    
    # Build matrix
    data = []
    for c in constraints:
        row = []
        for p in policies:
            if c in policy_comparison[p]:
                row.append(policy_comparison[p][c]['mean'])
            elif c == 'capacity_satisfy_rate':
                row.append(1.0)  # Default for greedy
            else:
                row.append(0.0)
        data.append(row)
    
    data = np.array(data)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=[p.replace('_', '\n') for p in policies],
                yticklabels=['Cold Start\nSuccess', 'Top 10%\nShare (lower=better)', 
                            'Capacity\nSatisfy'],
                ax=ax, vmin=0, vmax=1)
    
    ax.set_title('Constraint Satisfaction by Policy')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_lambda_convergence(results: Dict, output_path: Path):
    """Fig 3: Line plot of lambda convergence over rounds."""
    plt.figure(figsize=(10, 6))
    
    conv_data = results.get('lambda_convergence', {})
    history = conv_data.get('history', {})
    
    if not history:
        print("No lambda convergence data for Fig 3")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(history)))
    
    for (constraint, values), color in zip(history.items(), colors):
        rounds = np.arange(len(values))
        ax.plot(rounds, values, label=constraint, linewidth=2, color=color)
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Lambda Value')
    ax.set_title('Dual Variable (Lambda) Convergence')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_lr_sensitivity(results: Dict, output_path: Path):
    """Fig 4: Learning rate sensitivity plot."""
    plt.figure(figsize=(10, 6))
    
    lr_data = results.get('lr_sweep', {})
    if not lr_data:
        print("No LR sweep data for Fig 4")
        return
    
    etas = sorted([float(e) for e in lr_data.keys()])
    revenues = [lr_data[str(e)]['revenue_mean'] for e in etas]
    cold_starts = [lr_data[str(e)]['cold_start_mean'] for e in etas]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'steelblue'
    ax1.plot(etas, revenues, 'o-', color=color1, linewidth=2, markersize=8, label='Revenue')
    ax1.set_xlabel('Learning Rate (η)')
    ax1.set_ylabel('Revenue', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log')
    
    ax2 = ax1.twinx()
    color2 = 'coral'
    ax2.plot(etas, cold_starts, 's--', color=color2, linewidth=2, markersize=8, 
             label='Cold Start Rate')
    ax2.set_ylabel('Cold Start Success Rate', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_title('Learning Rate Sensitivity Analysis')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_constraint_ablation(results: Dict, output_path: Path):
    """Fig 5: Stacked bar chart for constraint ablation."""
    plt.figure(figsize=(10, 6))
    
    ablation_data = results.get('constraint_ablation', {})
    if not ablation_data:
        print("No ablation data for Fig 5")
        return
    
    combos = list(ablation_data.keys())
    deltas = [ablation_data[c]['delta_vs_greedy_pct'] for c in combos]
    
    # Color by positive/negative
    colors = ['green' if d >= 0 else 'red' for d in deltas]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(combos, deltas, color=colors, alpha=0.7, edgecolor='black')
    
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax.axhline(y=5, color='green', linestyle='--', linewidth=1, alpha=0.7, 
               label='Gate-5B Target (+5%)')
    
    ax.set_xlabel('Constraint Combination')
    ax.set_ylabel('Revenue Δ vs Greedy (%)')
    ax.set_title('Constraint Ablation: Revenue Impact')
    ax.legend()
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_pareto_frontier(results: Dict, output_path: Path):
    """Fig 6: Scatter plot of revenue vs constraint satisfaction (Pareto)."""
    plt.figure(figsize=(10, 6))
    
    policy_comparison = results.get('policy_comparison', {})
    if not policy_comparison:
        print("No policy comparison data for Fig 6")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['blue', 'orange', 'green', 'red']
    markers = ['o', 's', '^', 'D']
    
    for i, (policy, data) in enumerate(policy_comparison.items()):
        revenue = data['revenue']['mean']
        
        # Average constraint satisfaction
        cs_rate = data['cold_start_success']['mean']
        cap_rate = data.get('capacity_satisfy_rate', {}).get('mean', 0.9)
        head_ok = 1.0 - data['top_10_share']['mean']  # Lower share is better
        
        avg_satisfy = (cs_rate + cap_rate + head_ok) / 3
        
        ax.scatter(revenue, avg_satisfy, s=200, c=colors[i], marker=markers[i],
                  label=policy.replace('_', ' '), edgecolors='black', linewidths=1)
    
    ax.set_xlabel('Total Revenue')
    ax.set_ylabel('Avg Constraint Satisfaction')
    ax.set_title('Revenue vs Constraint Satisfaction (Pareto Frontier)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# Gate Evaluation
# ============================================================

def evaluate_gate5b(results: Dict) -> Tuple[str, Dict]:
    """Evaluate Gate-5B: Shadow price ≥ Greedy+5% AND constraint satisfy >90%."""
    policy_comparison = results.get('policy_comparison', {})
    
    if not policy_comparison:
        return 'FAIL', {'reason': 'No policy comparison data'}
    
    greedy_rev = policy_comparison['greedy']['revenue']['mean']
    target_rev = greedy_rev * 1.05
    
    # Check shadow_price_core (our main candidate)
    sp_core = policy_comparison.get('shadow_price_core', {})
    sp_all = policy_comparison.get('shadow_price_all', {})
    
    best_candidate = None
    best_rev = 0
    best_data = {}
    
    for name, data in [('shadow_price_core', sp_core), ('shadow_price_all', sp_all)]:
        rev = data.get('revenue', {}).get('mean', 0)
        cs_rate = data.get('cold_start_success', {}).get('mean', 0)
        cap_rate = data.get('capacity_satisfy_rate', {}).get('mean', 0.9)
        
        if rev > best_rev:
            best_rev = rev
            best_candidate = name
            best_data = {
                'revenue': rev,
                'revenue_vs_greedy_pct': (rev - greedy_rev) / greedy_rev * 100,
                'cold_start_success': cs_rate,
                'capacity_satisfy_rate': cap_rate,
            }
    
    # Gate criteria
    revenue_pass = best_rev >= target_rev
    constraint_pass = (
        best_data.get('cold_start_success', 0) >= 0.30 and
        best_data.get('capacity_satisfy_rate', 0) >= 0.90
    )
    
    gate_result = 'PASS' if (revenue_pass and constraint_pass) else 'FAIL'
    
    evaluation = {
        'best_candidate': best_candidate,
        'revenue_pass': revenue_pass,
        'revenue_target': target_rev,
        'revenue_actual': best_rev,
        'constraint_pass': constraint_pass,
        **best_data,
    }
    
    return gate_result, evaluation


# ============================================================
# Main
# ============================================================

def main():
    """Run all experiments and generate outputs."""
    print("=" * 70)
    print("Shadow Price Allocation Experiments")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Ensure output directories exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    results = {}
    
    # Experiment 1: Policy comparison
    results.update(run_policy_comparison(n_simulations=EXP_CONFIG['n_simulations']))
    
    # Experiment 2: Learning rate sweep
    results.update(run_lr_sweep(n_simulations=30))
    
    # Experiment 3: Constraint ablation
    results.update(run_constraint_ablation(n_simulations=30))
    
    # Experiment 4: Lambda convergence
    results.update(run_lambda_convergence(n_simulations=10))
    
    # Evaluate Gate-5B
    gate_result, gate_eval = evaluate_gate5b(results)
    results['gate5b'] = gate_result
    results['gate5b_evaluation'] = gate_eval
    
    print("\n" + "=" * 60)
    print("GATE-5B EVALUATION")
    print("=" * 60)
    print(f"Result: {gate_result}")
    for k, v in gate_eval.items():
        print(f"  {k}: {v}")
    
    # Save results
    output_path = RESULTS_DIR / "shadow_price_20260109.json"
    
    # Convert any non-serializable items
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    serializable_results = make_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nSaved results to: {output_path}")
    
    # Generate figures
    print("\nGenerating figures...")
    plot_revenue_comparison(results, IMG_DIR / "mvp52_revenue_comparison.png")
    plot_constraint_satisfaction(results, IMG_DIR / "mvp52_constraint_satisfaction.png")
    plot_lambda_convergence(results, IMG_DIR / "mvp52_lambda_convergence.png")
    plot_lr_sensitivity(results, IMG_DIR / "mvp52_lr_sensitivity.png")
    plot_constraint_ablation(results, IMG_DIR / "mvp52_constraint_ablation.png")
    plot_pareto_frontier(results, IMG_DIR / "mvp52_pareto_frontier.png")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
