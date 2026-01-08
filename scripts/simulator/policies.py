#!/usr/bin/env python3
"""
Allocation Policies for GiftLive Simulator
MVP-0.3, MVP-2.1, MVP-2.2

This module contains various allocation policies:
- Random, Greedy, RoundRobin (baselines)
- ConcaveLog, ConcaveExp (concave returns)
- ConstrainedPolicy (with cold-start and fairness constraints)
"""

import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass

# Import from simulator
from .simulator import AllocationPolicy, User, Streamer, GiftLiveSimulator


# ============================================================
# Concave Allocation Policies (MVP-2.1)
# ============================================================

class ConcaveLogPolicy(AllocationPolicy):
    """Concave allocation with log utility: g(V) = log(1 + V)
    
    Allocation rule: argmax_s g'(V_s) * v_{u,s}
    where g'(V) = 1 / (1 + V)
    """
    
    def allocate(self, users: List[User], simulator: GiftLiveSimulator) -> List[int]:
        ev_matrix = simulator.get_expected_values(users)
        
        # Get current cumulative revenues
        cum_revenues = np.array([s.cumulative_revenue for s in simulator.streamer_pool.streamers])
        
        # Marginal utility: g'(V) = 1 / (1 + V)
        marginal_utility = 1.0 / (1.0 + cum_revenues)
        
        # Score: g'(V_s) * v_{u,s}
        scores = ev_matrix * marginal_utility[np.newaxis, :]
        
        return np.argmax(scores, axis=1).tolist()


class ConcaveExpPolicy(AllocationPolicy):
    """Concave allocation with saturation utility: g(V) = B * (1 - exp(-V/B))
    
    Allocation rule: argmax_s g'(V_s) * v_{u,s}
    where g'(V) = exp(-V/B)
    """
    
    def __init__(self, B: float = 1000.0):
        self.B = B
    
    def allocate(self, users: List[User], simulator: GiftLiveSimulator) -> List[int]:
        ev_matrix = simulator.get_expected_values(users)
        
        # Get current cumulative revenues
        cum_revenues = np.array([s.cumulative_revenue for s in simulator.streamer_pool.streamers])
        
        # Marginal utility: g'(V) = exp(-V/B)
        marginal_utility = np.exp(-cum_revenues / self.B)
        
        # Score: g'(V_s) * v_{u,s}
        scores = ev_matrix * marginal_utility[np.newaxis, :]
        
        return np.argmax(scores, axis=1).tolist()


class ConcavePowerPolicy(AllocationPolicy):
    """Concave allocation with power utility: g(V) = V^alpha, alpha < 1
    
    Allocation rule: argmax_s g'(V_s) * v_{u,s}
    where g'(V) = alpha * V^(alpha-1)
    """
    
    def __init__(self, alpha: float = 0.5):
        assert 0 < alpha < 1, "Alpha must be in (0, 1)"
        self.alpha = alpha
    
    def allocate(self, users: List[User], simulator: GiftLiveSimulator) -> List[int]:
        ev_matrix = simulator.get_expected_values(users)
        
        # Get current cumulative revenues
        cum_revenues = np.array([s.cumulative_revenue for s in simulator.streamer_pool.streamers])
        
        # Marginal utility: g'(V) = alpha * V^(alpha-1)
        # Add small epsilon to avoid division by zero
        marginal_utility = self.alpha * np.power(cum_revenues + 1e-6, self.alpha - 1)
        
        # Score: g'(V_s) * v_{u,s}
        scores = ev_matrix * marginal_utility[np.newaxis, :]
        
        return np.argmax(scores, axis=1).tolist()


class GreedyWithCapPolicy(AllocationPolicy):
    """Greedy allocation with head cap: switch to random if V_s > cap."""
    
    def __init__(self, cap: float = 10000.0, rng: Optional[np.random.Generator] = None):
        self.cap = cap
        self.rng = rng or np.random.default_rng(42)
    
    def allocate(self, users: List[User], simulator: GiftLiveSimulator) -> List[int]:
        ev_matrix = simulator.get_expected_values(users)
        cum_revenues = np.array([s.cumulative_revenue for s in simulator.streamer_pool.streamers])
        
        allocations = []
        for i in range(len(users)):
            # Mask out streamers over cap
            valid_mask = cum_revenues < self.cap
            
            if valid_mask.any():
                masked_ev = ev_matrix[i].copy()
                masked_ev[~valid_mask] = -np.inf
                allocations.append(int(np.argmax(masked_ev)))
            else:
                # All over cap, random allocation
                allocations.append(int(self.rng.integers(0, len(cum_revenues))))
        
        return allocations


# ============================================================
# Constrained Allocation Policies (MVP-2.2)
# ============================================================

@dataclass
class ConstraintConfig:
    """Configuration for allocation constraints."""
    # Cold start constraint
    enable_cold_start: bool = False
    min_allocation_per_new: int = 10
    new_streamer_window_days: int = 7
    cold_start_enforce: str = "soft"  # "soft" or "hard"
    
    # Head cap constraint
    enable_head_cap: bool = False
    max_share_top10: float = 0.5
    head_cap_enforce: str = "soft"
    
    # Diversity constraint
    enable_diversity: bool = False
    min_coverage: float = 0.3
    diversity_enforce: str = "soft"
    
    # Lagrangian optimization
    lambda_cold_start: float = 0.1
    lambda_head_cap: float = 0.1
    lambda_diversity: float = 0.1
    lambda_lr: float = 0.01


class ConstrainedAllocationPolicy(AllocationPolicy):
    """Allocation policy with cold-start and fairness constraints.
    
    Uses Lagrangian relaxation for soft constraints.
    """
    
    def __init__(
        self,
        base_policy: AllocationPolicy,
        constraint_config: ConstraintConfig,
        rng: Optional[np.random.Generator] = None
    ):
        self.base_policy = base_policy
        self.config = constraint_config
        self.rng = rng or np.random.default_rng(42)
        
        # Lagrangian multipliers
        self.lambda_cold_start = constraint_config.lambda_cold_start
        self.lambda_head_cap = constraint_config.lambda_head_cap
        self.lambda_diversity = constraint_config.lambda_diversity
        
        # Tracking
        self.new_streamer_allocations: Dict[int, int] = {}  # streamer_id -> allocation count
        self.violation_history: List[Dict] = []
    
    def allocate(self, users: List[User], simulator: GiftLiveSimulator) -> List[int]:
        n_users = len(users)
        n_streamers = simulator.config.n_streamers
        
        # Get expected values
        ev_matrix = simulator.get_expected_values(users)
        
        # Get cumulative revenues
        cum_revenues = np.array([s.cumulative_revenue for s in simulator.streamer_pool.streamers])
        
        # Get new streamer mask
        new_streamer_mask = np.array([s.is_new for s in simulator.streamer_pool.streamers])
        
        # Compute base scores (from base policy logic)
        if isinstance(self.base_policy, ConcaveLogPolicy):
            marginal_utility = 1.0 / (1.0 + cum_revenues)
            scores = ev_matrix * marginal_utility[np.newaxis, :]
        elif isinstance(self.base_policy, ConcaveExpPolicy):
            marginal_utility = np.exp(-cum_revenues / self.base_policy.B)
            scores = ev_matrix * marginal_utility[np.newaxis, :]
        else:
            # Default to greedy
            scores = ev_matrix.copy()
        
        # Apply constraints
        if self.config.enable_cold_start and self.config.cold_start_enforce == "hard":
            # Hard constraint: reserve allocations for new streamers
            return self._allocate_with_hard_cold_start(users, simulator, scores, new_streamer_mask)
        
        # Soft constraints: Lagrangian penalty
        if self.config.enable_cold_start and self.config.cold_start_enforce == "soft":
            # Boost new streamers by lambda
            cold_start_bonus = self.lambda_cold_start * new_streamer_mask.astype(float)
            scores = scores + cold_start_bonus[np.newaxis, :]
        
        if self.config.enable_head_cap and self.config.head_cap_enforce == "soft":
            # Penalty for high-revenue streamers
            revenue_threshold = np.percentile(cum_revenues[cum_revenues > 0], 90) if (cum_revenues > 0).any() else 1e10
            head_penalty = self.lambda_head_cap * np.maximum(0, cum_revenues - revenue_threshold)
            scores = scores - head_penalty[np.newaxis, :]
        
        if self.config.enable_diversity and self.config.diversity_enforce == "soft":
            # Bonus for low-coverage streamers
            allocation_counts = np.zeros(n_streamers)
            for sid in self.new_streamer_allocations:
                if sid < n_streamers:
                    allocation_counts[sid] = self.new_streamer_allocations[sid]
            diversity_bonus = self.lambda_diversity / (1 + allocation_counts)
            scores = scores + diversity_bonus[np.newaxis, :]
        
        # Allocate
        allocations = np.argmax(scores, axis=1).tolist()
        
        # Update tracking
        for sid in allocations:
            if sid not in self.new_streamer_allocations:
                self.new_streamer_allocations[sid] = 0
            self.new_streamer_allocations[sid] += 1
        
        return allocations
    
    def _allocate_with_hard_cold_start(
        self,
        users: List[User],
        simulator: GiftLiveSimulator,
        scores: np.ndarray,
        new_streamer_mask: np.ndarray
    ) -> List[int]:
        """Hard cold-start constraint: reserve some users for new streamers."""
        n_users = len(users)
        n_streamers = simulator.config.n_streamers
        
        # Calculate how many allocations each new streamer still needs
        new_streamer_ids = np.where(new_streamer_mask)[0]
        reserve_needed = {}
        for sid in new_streamer_ids:
            current_alloc = self.new_streamer_allocations.get(sid, 0)
            needed = max(0, self.config.min_allocation_per_new - current_alloc)
            if needed > 0:
                reserve_needed[sid] = needed
        
        total_reserved = sum(reserve_needed.values())
        
        allocations = []
        reserved_used = {sid: 0 for sid in reserve_needed}
        
        for i in range(n_users):
            # First, try to satisfy reservation
            if reserved_used and i < total_reserved:
                # Find a new streamer that still needs allocation
                for sid, needed in reserve_needed.items():
                    if reserved_used[sid] < needed:
                        allocations.append(sid)
                        reserved_used[sid] += 1
                        break
                else:
                    # No reservation left, use scores
                    allocations.append(int(np.argmax(scores[i])))
            else:
                # Use scores
                allocations.append(int(np.argmax(scores[i])))
        
        # Update tracking
        for sid in allocations:
            if sid not in self.new_streamer_allocations:
                self.new_streamer_allocations[sid] = 0
            self.new_streamer_allocations[sid] += 1
        
        return allocations
    
    def update_dual_variables(self, metrics: Dict):
        """Update Lagrangian multipliers based on constraint violations."""
        violations = {}
        
        if self.config.enable_cold_start:
            # Violation: cold_start_success_rate < target
            target_rate = 0.3  # Target 30% success rate for new streamers
            actual_rate = metrics.get('cold_start_success_rate', 0)
            violation = max(0, target_rate - actual_rate)
            violations['cold_start'] = violation
            self.lambda_cold_start = max(0, self.lambda_cold_start + self.config.lambda_lr * violation)
        
        if self.config.enable_head_cap:
            # Violation: top_10_share > max_share
            actual_share = metrics.get('top_10_streamer_share', 0)
            violation = max(0, actual_share - self.config.max_share_top10)
            violations['head_cap'] = violation
            self.lambda_head_cap = max(0, self.lambda_head_cap + self.config.lambda_lr * violation)
        
        if self.config.enable_diversity:
            # Violation: coverage < min_coverage
            actual_coverage = metrics.get('coverage', 0)
            violation = max(0, self.config.min_coverage - actual_coverage)
            violations['diversity'] = violation
            self.lambda_diversity = max(0, self.lambda_diversity + self.config.lambda_lr * violation)
        
        self.violation_history.append(violations)
        return violations
    
    def reset(self):
        """Reset tracking state."""
        self.new_streamer_allocations = {}
        self.violation_history = []
        self.lambda_cold_start = self.config.lambda_cold_start
        self.lambda_head_cap = self.config.lambda_head_cap
        self.lambda_diversity = self.config.lambda_diversity


class ColdStartTracker:
    """Track cold-start streamer performance."""
    
    def __init__(self, simulator: GiftLiveSimulator):
        self.simulator = simulator
        self.new_streamer_ids = set(
            s.streamer_id for s in simulator.streamer_pool.streamers if s.is_new
        )
        self.allocation_counts: Dict[int, int] = {sid: 0 for sid in self.new_streamer_ids}
        self.gift_counts: Dict[int, int] = {sid: 0 for sid in self.new_streamer_ids}
        self.revenue: Dict[int, float] = {sid: 0.0 for sid in self.new_streamer_ids}
    
    def update(self, records: List[Dict]):
        """Update tracking from interaction records."""
        for r in records:
            sid = r['streamer_id']
            if sid in self.new_streamer_ids:
                self.allocation_counts[sid] += 1
                if r['did_gift']:
                    self.gift_counts[sid] += 1
                    self.revenue[sid] += r['amount']
    
    def get_metrics(self) -> Dict:
        """Compute cold-start metrics."""
        n_new = len(self.new_streamer_ids)
        n_with_gift = sum(1 for sid in self.new_streamer_ids if self.gift_counts[sid] > 0)
        total_revenue = sum(self.revenue.values())
        total_allocations = sum(self.allocation_counts.values())
        
        return {
            'n_new_streamers': n_new,
            'n_new_with_gift': n_with_gift,
            'success_rate': n_with_gift / n_new if n_new > 0 else 0,
            'total_new_revenue': total_revenue,
            'total_new_allocations': total_allocations,
            'avg_allocations_per_new': total_allocations / n_new if n_new > 0 else 0,
        }


# ============================================================
# Policy Factory
# ============================================================

def create_policy(name: str, **kwargs) -> AllocationPolicy:
    """Factory function to create allocation policies."""
    from .simulator import RandomPolicy, GreedyPolicy, RoundRobinPolicy
    
    policies = {
        'random': lambda: RandomPolicy(rng=kwargs.get('rng')),
        'greedy': lambda: GreedyPolicy(),
        'round_robin': lambda: RoundRobinPolicy(),
        'concave_log': lambda: ConcaveLogPolicy(),
        'concave_exp': lambda: ConcaveExpPolicy(B=kwargs.get('B', 1000.0)),
        'concave_power': lambda: ConcavePowerPolicy(alpha=kwargs.get('alpha', 0.5)),
        'greedy_with_cap': lambda: GreedyWithCapPolicy(
            cap=kwargs.get('cap', 10000.0), 
            rng=kwargs.get('rng')
        ),
    }
    
    if name not in policies:
        raise ValueError(f"Unknown policy: {name}. Available: {list(policies.keys())}")
    
    return policies[name]()


if __name__ == "__main__":
    from .simulator import GiftLiveSimulator, SimConfig
    
    print("Testing allocation policies...")
    
    config = SimConfig(n_users=1000, n_streamers=100, seed=42)
    sim = GiftLiveSimulator(config)
    
    policies = {
        'greedy': GreedyPolicy(),
        'concave_log': ConcaveLogPolicy(),
        'concave_exp': ConcaveExpPolicy(B=500),
    }
    
    for name, policy in policies.items():
        sim.reset()
        metrics = sim.run_simulation(policy, n_rounds=20, users_per_round=50)
        print(f"\n{name}:")
        print(f"  Revenue: {metrics['total_revenue']:.2f}")
        print(f"  Gini: {metrics['streamer_gini']:.4f}")
        print(f"  Top-10 Share: {metrics['top_10_streamer_share']:.4f}")
