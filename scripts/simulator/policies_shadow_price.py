#!/usr/bin/env python3
"""
Shadow Price Allocation Policies with Primal-Dual Framework
MVP-5.2: Unified constraint handling via Lagrangian relaxation

This module implements:
- Abstract Constraint interface
- 5 concrete constraint types (capacity, cold_start, head_cap, whale_spread, frequency)
- ShadowPriceAllocator with online Primal-Dual updates
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

# Import from simulator
from .simulator import AllocationPolicy, User, Streamer, GiftLiveSimulator


# ============================================================
# Constraint Interface
# ============================================================

class Constraint(ABC):
    """Abstract base class for allocation constraints.
    
    Each constraint computes:
    - violation: how much the constraint is violated (for dual update)
    - penalty: marginal cost of allocating user u to streamer s
    """
    
    def __init__(self, name: str, lambda_init: float = 0.1, lambda_max: float = 10.0):
        self.name = name
        self.lambda_value = lambda_init
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        self.history: List[Dict] = []
    
    @abstractmethod
    def compute_penalty(self, user: User, streamer: Streamer, state: Dict) -> float:
        """Compute marginal penalty for allocating user to streamer.
        
        Args:
            user: User being allocated
            streamer: Candidate streamer
            state: Current allocation state (counts, etc.)
            
        Returns:
            Penalty value (higher = less preferred)
        """
        pass
    
    @abstractmethod
    def compute_violation(self, state: Dict) -> float:
        """Compute constraint violation for dual variable update.
        
        Args:
            state: Current allocation state
            
        Returns:
            Violation value (positive = violated, negative = slack)
        """
        pass
    
    def update_lambda(self, violation: float, eta: float):
        """Update Lagrangian multiplier via subgradient descent.
        
        λ_c^{t+1} = [λ_c^t + η * violation]_+
        """
        new_lambda = self.lambda_value + eta * violation
        self.lambda_value = np.clip(new_lambda, 0, self.lambda_max)
        self.history.append({
            'lambda': self.lambda_value,
            'violation': violation
        })
    
    def reset(self):
        """Reset lambda to initial value."""
        self.lambda_value = self.lambda_init
        self.history = []


# ============================================================
# Constraint Implementations
# ============================================================

class CapacityConstraint(Constraint):
    """C1: Streamer concurrent capacity constraint.
    
    Penalizes allocations to streamers that are already at/over capacity.
    """
    
    def __init__(self, lambda_init: float = 0.1, lambda_max: float = 5.0):
        super().__init__("capacity", lambda_init, lambda_max)
    
    def compute_penalty(self, user: User, streamer: Streamer, state: Dict) -> float:
        """Returns 1.0 if streamer is at/over capacity, 0.0 otherwise."""
        current_load = state.get('current_load', {}).get(streamer.streamer_id, 0)
        if current_load >= streamer.capacity:
            return 1.0
        return 0.0
    
    def compute_violation(self, state: Dict) -> float:
        """Compute average capacity violation across all streamers."""
        current_load = state.get('current_load', {})
        capacities = state.get('capacities', {})
        
        if not current_load:
            return 0.0
        
        violations = []
        for sid, load in current_load.items():
            cap = capacities.get(sid, 50)
            if load > cap:
                violations.append((load - cap) / cap)
            else:
                violations.append(0.0)
        
        return np.mean(violations) if violations else 0.0


class ColdStartConstraint(Constraint):
    """C2: Cold-start coverage constraint (bonus for new streamers).
    
    Gives bonus (negative penalty) to allocations to new streamers.
    """
    
    def __init__(
        self, 
        min_alloc_per_new: int = 10,
        lambda_init: float = 0.5, 
        lambda_max: float = 3.0
    ):
        super().__init__("cold_start", lambda_init, lambda_max)
        self.min_alloc_per_new = min_alloc_per_new
    
    def compute_penalty(self, user: User, streamer: Streamer, state: Dict) -> float:
        """Returns negative penalty (bonus) for new streamers that need more allocations."""
        if not streamer.is_new:
            return 0.0
        
        # Check current allocations to this new streamer
        new_alloc_counts = state.get('new_alloc_counts', {})
        current_alloc = new_alloc_counts.get(streamer.streamer_id, 0)
        
        if current_alloc < self.min_alloc_per_new:
            # Bonus: still needs allocation
            need_ratio = (self.min_alloc_per_new - current_alloc) / self.min_alloc_per_new
            return -need_ratio  # Negative = bonus
        return 0.0
    
    def compute_violation(self, state: Dict) -> float:
        """Compute fraction of new streamers below minimum allocation."""
        new_alloc_counts = state.get('new_alloc_counts', {})
        new_streamer_ids = state.get('new_streamer_ids', set())
        
        if not new_streamer_ids:
            return 0.0
        
        n_below_min = sum(
            1 for sid in new_streamer_ids 
            if new_alloc_counts.get(sid, 0) < self.min_alloc_per_new
        )
        
        # Violation is positive when below target
        return n_below_min / len(new_streamer_ids)


class HeadCapConstraint(Constraint):
    """C3: Head (top 10%) revenue share cap.
    
    Penalizes allocations to top-10% revenue streamers when share exceeds cap.
    """
    
    def __init__(
        self, 
        max_share: float = 0.5,
        lambda_init: float = 0.1, 
        lambda_max: float = 3.0
    ):
        super().__init__("head_cap", lambda_init, lambda_max)
        self.max_share = max_share
    
    def compute_penalty(self, user: User, streamer: Streamer, state: Dict) -> float:
        """Penalty proportional to current share if streamer is in top 10%."""
        top10_ids = state.get('top10_streamer_ids', set())
        
        if streamer.streamer_id not in top10_ids:
            return 0.0
        
        current_share = state.get('top10_share', 0.0)
        if current_share > self.max_share:
            return current_share
        return current_share * 0.5  # Smaller penalty when under cap
    
    def compute_violation(self, state: Dict) -> float:
        """Violation when top10 share exceeds max_share."""
        current_share = state.get('top10_share', 0.0)
        return max(0, current_share - self.max_share)


class WhaleSpreadConstraint(Constraint):
    """C4: Whale (high-value user) distribution constraint.
    
    Limits how many whale users can be allocated to the same streamer.
    """
    
    def __init__(
        self, 
        max_whale_per_streamer: int = 2,
        whale_threshold: float = 500.0,  # Wealth threshold
        lambda_init: float = 0.2, 
        lambda_max: float = 3.0
    ):
        super().__init__("whale_spread", lambda_init, lambda_max)
        self.max_whale_per_streamer = max_whale_per_streamer
        self.whale_threshold = whale_threshold
    
    def is_whale(self, user: User) -> bool:
        """Check if user is a whale based on wealth."""
        return user.wealth >= self.whale_threshold
    
    def compute_penalty(self, user: User, streamer: Streamer, state: Dict) -> float:
        """Penalty when allocating whale to streamer that already has many whales."""
        if not self.is_whale(user):
            return 0.0
        
        whale_counts = state.get('whale_counts', {})
        current_whales = whale_counts.get(streamer.streamer_id, 0)
        
        if current_whales >= self.max_whale_per_streamer:
            return current_whales / self.max_whale_per_streamer
        return 0.0
    
    def compute_violation(self, state: Dict) -> float:
        """Fraction of streamers exceeding whale limit."""
        whale_counts = state.get('whale_counts', {})
        
        if not whale_counts:
            return 0.0
        
        n_over = sum(1 for count in whale_counts.values() if count > self.max_whale_per_streamer)
        return n_over / len(whale_counts)


class FrequencyConstraint(Constraint):
    """C5: User-streamer frequency constraint.
    
    Limits repeat allocations of the same user to the same streamer.
    """
    
    def __init__(
        self, 
        max_freq_per_pair: int = 3,
        lambda_init: float = 0.3, 
        lambda_max: float = 5.0
    ):
        super().__init__("frequency", lambda_init, lambda_max)
        self.max_freq_per_pair = max_freq_per_pair
    
    def compute_penalty(self, user: User, streamer: Streamer, state: Dict) -> float:
        """Binary penalty if frequency exceeds max."""
        freq_counts = state.get('freq_counts', {})
        pair_key = (user.user_id, streamer.streamer_id)
        current_freq = freq_counts.get(pair_key, 0)
        
        if current_freq >= self.max_freq_per_pair:
            return 1.0
        return 0.0
    
    def compute_violation(self, state: Dict) -> float:
        """Fraction of pairs exceeding frequency limit."""
        freq_counts = state.get('freq_counts', {})
        
        if not freq_counts:
            return 0.0
        
        n_over = sum(1 for freq in freq_counts.values() if freq > self.max_freq_per_pair)
        return n_over / len(freq_counts) if freq_counts else 0.0


# ============================================================
# Shadow Price Allocator
# ============================================================

@dataclass
class ShadowPriceConfig:
    """Configuration for Shadow Price Allocator."""
    # Learning rate for dual variable updates
    eta: float = 0.05
    
    # Constraint enables
    enable_capacity: bool = True
    enable_cold_start: bool = True
    enable_head_cap: bool = True
    enable_whale_spread: bool = True
    enable_frequency: bool = True
    
    # Constraint parameters
    min_alloc_per_new: int = 10
    max_share_top10: float = 0.5
    max_whale_per_streamer: int = 2
    whale_threshold: float = 500.0
    max_freq_per_pair: int = 3
    
    # Lambda initial values
    lambda_capacity: float = 0.1
    lambda_cold_start: float = 0.5
    lambda_head_cap: float = 0.1
    lambda_whale_spread: float = 0.2
    lambda_frequency: float = 0.3


class ShadowPriceAllocator(AllocationPolicy):
    """Primal-Dual Shadow Price Allocator.
    
    Implements:
        s*(u) = argmax_s [EV(u,s) - Σ_c λ_c * Δg_c(u→s)]
    
    Where:
        - EV(u,s): Expected value of allocating user u to streamer s
        - λ_c: Lagrangian multiplier (shadow price) for constraint c
        - Δg_c(u→s): Marginal constraint penalty for allocation
    
    After each round, dual variables are updated:
        λ_c^{t+1} = [λ_c^t + η * violation_c]_+
    """
    
    def __init__(
        self, 
        config: Optional[ShadowPriceConfig] = None,
        constraints: Optional[List[str]] = None,
        rng: Optional[np.random.Generator] = None
    ):
        self.config = config or ShadowPriceConfig()
        self.rng = rng or np.random.default_rng(42)
        
        # Initialize constraints
        self.constraints: Dict[str, Constraint] = {}
        self._init_constraints(constraints)
        
        # State tracking
        self.state: Dict = {}
        self._reset_state()
        
        # Lambda history for convergence analysis
        self.lambda_history: List[Dict[str, float]] = []
        self.violation_history: List[Dict[str, float]] = []
        self.round_count = 0
    
    def _init_constraints(self, constraint_names: Optional[List[str]] = None):
        """Initialize constraint objects based on config."""
        all_constraints = {
            'capacity': lambda: CapacityConstraint(
                lambda_init=self.config.lambda_capacity
            ),
            'cold_start': lambda: ColdStartConstraint(
                min_alloc_per_new=self.config.min_alloc_per_new,
                lambda_init=self.config.lambda_cold_start
            ),
            'head_cap': lambda: HeadCapConstraint(
                max_share=self.config.max_share_top10,
                lambda_init=self.config.lambda_head_cap
            ),
            'whale_spread': lambda: WhaleSpreadConstraint(
                max_whale_per_streamer=self.config.max_whale_per_streamer,
                whale_threshold=self.config.whale_threshold,
                lambda_init=self.config.lambda_whale_spread
            ),
            'frequency': lambda: FrequencyConstraint(
                max_freq_per_pair=self.config.max_freq_per_pair,
                lambda_init=self.config.lambda_frequency
            ),
        }
        
        # Use provided constraint names or all enabled in config
        if constraint_names:
            active_constraints = constraint_names
        else:
            active_constraints = []
            if self.config.enable_capacity:
                active_constraints.append('capacity')
            if self.config.enable_cold_start:
                active_constraints.append('cold_start')
            if self.config.enable_head_cap:
                active_constraints.append('head_cap')
            if self.config.enable_whale_spread:
                active_constraints.append('whale_spread')
            if self.config.enable_frequency:
                active_constraints.append('frequency')
        
        for name in active_constraints:
            if name in all_constraints:
                self.constraints[name] = all_constraints[name]()
    
    def _reset_state(self):
        """Reset allocation state."""
        self.state = {
            'current_load': defaultdict(int),     # streamer_id -> concurrent users
            'capacities': {},                      # streamer_id -> capacity
            'new_alloc_counts': defaultdict(int), # streamer_id -> allocation count for new streamers
            'new_streamer_ids': set(),             # Set of new streamer IDs
            'streamer_revenue': defaultdict(float),  # streamer_id -> cumulative revenue
            'top10_streamer_ids': set(),           # Set of top 10% revenue streamers
            'top10_share': 0.0,                    # Current top 10% revenue share
            'whale_counts': defaultdict(int),     # streamer_id -> whale count
            'freq_counts': defaultdict(int),      # (user_id, streamer_id) -> frequency
            'total_revenue': 0.0,
        }
    
    def reset(self):
        """Full reset including lambda values."""
        self._reset_state()
        for constraint in self.constraints.values():
            constraint.reset()
        self.lambda_history = []
        self.violation_history = []
        self.round_count = 0
    
    def initialize_from_simulator(self, simulator: GiftLiveSimulator):
        """Initialize state from simulator (called once before simulation)."""
        # Extract streamer info
        for s in simulator.streamer_pool.streamers:
            self.state['capacities'][s.streamer_id] = s.capacity
            if s.is_new:
                self.state['new_streamer_ids'].add(s.streamer_id)
        
        # Initialize top 10% (empty at start)
        n_streamers = len(simulator.streamer_pool.streamers)
        n_top10 = max(1, int(n_streamers * 0.1))
        self.state['n_top10'] = n_top10
    
    def allocate(self, users: List[User], simulator: GiftLiveSimulator) -> List[int]:
        """Allocate users to streamers using shadow price scoring.
        
        For each user, compute:
            score(u, s) = EV(u, s) - Σ_c λ_c * penalty_c(u, s)
        
        Then select argmax_s score(u, s).
        """
        n_users = len(users)
        n_streamers = simulator.config.n_streamers
        
        # Get expected values matrix (n_users, n_streamers)
        ev_matrix = simulator.get_expected_values(users)
        
        # Prepare streamer list
        streamers = simulator.streamer_pool.streamers
        
        # Reset per-round load counts
        self.state['current_load'] = defaultdict(int)
        
        # Update top 10% streamer info
        self._update_top10_info(simulator)
        
        allocations = []
        
        for i, user in enumerate(users):
            # Compute penalty-adjusted scores for all streamers
            scores = ev_matrix[i].copy()
            
            for j, streamer in enumerate(streamers):
                total_penalty = 0.0
                for name, constraint in self.constraints.items():
                    penalty = constraint.compute_penalty(user, streamer, self.state)
                    total_penalty += constraint.lambda_value * penalty
                
                scores[j] -= total_penalty
            
            # Select best streamer
            best_idx = int(np.argmax(scores))
            allocations.append(best_idx)
            
            # Update state for next user in batch
            self._update_state_for_allocation(user, streamers[best_idx])
        
        return allocations
    
    def _update_state_for_allocation(self, user: User, streamer: Streamer):
        """Update state after allocating user to streamer."""
        sid = streamer.streamer_id
        
        # Update load
        self.state['current_load'][sid] += 1
        
        # Update new streamer allocations
        if streamer.is_new:
            self.state['new_alloc_counts'][sid] += 1
        
        # Update whale counts
        if 'whale_spread' in self.constraints:
            whale_constraint = self.constraints['whale_spread']
            if whale_constraint.is_whale(user):
                self.state['whale_counts'][sid] += 1
        
        # Update frequency
        pair_key = (user.user_id, sid)
        self.state['freq_counts'][pair_key] += 1
    
    def _update_top10_info(self, simulator: GiftLiveSimulator):
        """Update top 10% streamer info based on cumulative revenue."""
        revenues = [(s.streamer_id, s.cumulative_revenue) 
                   for s in simulator.streamer_pool.streamers]
        total_rev = sum(r for _, r in revenues)
        
        if total_rev > 0:
            # Sort by revenue descending
            revenues.sort(key=lambda x: x[1], reverse=True)
            n_top = self.state.get('n_top10', 50)
            
            self.state['top10_streamer_ids'] = set(
                sid for sid, _ in revenues[:n_top]
            )
            
            top10_rev = sum(r for _, r in revenues[:n_top])
            self.state['top10_share'] = top10_rev / total_rev
        else:
            self.state['top10_streamer_ids'] = set()
            self.state['top10_share'] = 0.0
        
        self.state['total_revenue'] = total_rev
    
    def update_dual_variables(self) -> Dict[str, float]:
        """Update Lagrangian multipliers after each round.
        
        λ_c^{t+1} = [λ_c^t + η * violation_c]_+
        
        Returns:
            Dict of constraint name -> violation
        """
        violations = {}
        lambdas = {}
        
        for name, constraint in self.constraints.items():
            violation = constraint.compute_violation(self.state)
            constraint.update_lambda(violation, self.config.eta)
            violations[name] = violation
            lambdas[name] = constraint.lambda_value
        
        # Record history
        self.lambda_history.append(lambdas.copy())
        self.violation_history.append(violations.copy())
        self.round_count += 1
        
        return violations
    
    def get_constraint_status(self) -> Dict[str, Dict]:
        """Get current status of all constraints."""
        status = {}
        for name, constraint in self.constraints.items():
            violation = constraint.compute_violation(self.state)
            status[name] = {
                'lambda': constraint.lambda_value,
                'violation': violation,
                'satisfied': violation <= 0
            }
        return status
    
    def get_final_lambdas(self) -> Dict[str, float]:
        """Get final lambda values."""
        return {name: c.lambda_value for name, c in self.constraints.items()}
    
    def get_lambda_stability(self, last_n: int = 50) -> Dict[str, float]:
        """Compute lambda stability (std/mean) over last N rounds."""
        if len(self.lambda_history) < last_n:
            last_n = len(self.lambda_history)
        
        if last_n == 0:
            return {}
        
        stability = {}
        for name in self.constraints:
            values = [h[name] for h in self.lambda_history[-last_n:]]
            mean_val = np.mean(values)
            std_val = np.std(values)
            stability[name] = std_val / (mean_val + 1e-8)
        
        return stability


# ============================================================
# Baseline Policies for Comparison
# ============================================================

class GreedyWithRulesPolicy(AllocationPolicy):
    """Greedy policy with simple rules (cold-start bonus + frequency penalty).
    
    This is a baseline that uses ad-hoc rules instead of unified shadow prices.
    """
    
    def __init__(
        self, 
        lambda_cold_start: float = 0.5,
        freq_penalty: float = 0.3,
        max_freq: int = 3,
        rng: Optional[np.random.Generator] = None
    ):
        self.lambda_cold_start = lambda_cold_start
        self.freq_penalty = freq_penalty
        self.max_freq = max_freq
        self.rng = rng or np.random.default_rng(42)
        self.freq_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    
    def reset(self):
        self.freq_counts = defaultdict(int)
    
    def allocate(self, users: List[User], simulator: GiftLiveSimulator) -> List[int]:
        ev_matrix = simulator.get_expected_values(users)
        streamers = simulator.streamer_pool.streamers
        
        # Build new streamer mask
        new_mask = np.array([s.is_new for s in streamers], dtype=float)
        
        allocations = []
        
        for i, user in enumerate(users):
            scores = ev_matrix[i].copy()
            
            # Add cold-start bonus
            scores += self.lambda_cold_start * new_mask
            
            # Apply frequency penalty
            for j, s in enumerate(streamers):
                pair_key = (user.user_id, s.streamer_id)
                freq = self.freq_counts[pair_key]
                if freq >= self.max_freq:
                    scores[j] -= self.freq_penalty
            
            # Select best
            best_idx = int(np.argmax(scores))
            allocations.append(best_idx)
            
            # Update freq counts
            pair_key = (user.user_id, best_idx)
            self.freq_counts[pair_key] += 1
        
        return allocations


# ============================================================
# Evaluation Utilities
# ============================================================

def evaluate_constraint_satisfaction(
    simulator: GiftLiveSimulator,
    allocator: ShadowPriceAllocator,
    records: List[Dict]
) -> Dict[str, float]:
    """Evaluate how well constraints are satisfied.
    
    Returns:
        Dict of constraint satisfaction metrics
    """
    n_streamers = simulator.config.n_streamers
    
    # Capacity satisfaction
    load_counts = defaultdict(int)
    for r in records:
        load_counts[r['streamer_id']] += 1
    
    capacity_satisfied = 0
    for sid, load in load_counts.items():
        cap = simulator.streamer_pool.get_streamer(sid).capacity
        if load <= cap:
            capacity_satisfied += 1
    capacity_rate = capacity_satisfied / len(load_counts) if load_counts else 1.0
    
    # Cold-start success rate
    new_streamers_with_gift = set()
    new_streamers = set(s.streamer_id for s in simulator.streamer_pool.streamers if s.is_new)
    for r in records:
        if r['streamer_is_new'] and r['did_gift']:
            new_streamers_with_gift.add(r['streamer_id'])
    cold_start_rate = len(new_streamers_with_gift) / len(new_streamers) if new_streamers else 0
    
    # Head share
    streamer_revenue = defaultdict(float)
    total_rev = 0
    for r in records:
        streamer_revenue[r['streamer_id']] += r['amount']
        total_rev += r['amount']
    
    if total_rev > 0:
        revenues = sorted(streamer_revenue.values(), reverse=True)
        n_top10 = max(1, len(revenues) // 10)
        top10_share = sum(revenues[:n_top10]) / total_rev
        head_share_ok = 1.0 if top10_share <= 0.5 else 0.0
    else:
        top10_share = 0
        head_share_ok = 1.0
    
    # Whale spread
    whale_counts = defaultdict(int)
    whale_threshold = 500.0
    for r in records:
        if r['user_wealth'] >= whale_threshold:
            whale_counts[r['streamer_id']] += 1
    
    n_whale_ok = sum(1 for count in whale_counts.values() if count <= 2)
    whale_spread_rate = n_whale_ok / len(whale_counts) if whale_counts else 1.0
    
    return {
        'capacity_satisfy_rate': capacity_rate,
        'cold_start_success_rate': cold_start_rate,
        'head_share_within_cap': head_share_ok,
        'whale_spread_rate': whale_spread_rate,
        'top_10_share': top10_share,
    }


def run_shadow_price_simulation(
    simulator: GiftLiveSimulator,
    allocator: ShadowPriceAllocator,
    n_rounds: int = 50,
    users_per_round: int = 200,
    track_lambda: bool = True,
    verbose: bool = False
) -> Dict:
    """Run simulation with shadow price allocator and dual variable updates.
    
    Returns:
        Simulation results with constraint metrics
    """
    simulator.reset()
    allocator.reset()
    allocator.initialize_from_simulator(simulator)
    
    all_records = []
    
    for round_idx in range(n_rounds):
        users = simulator.user_pool.sample_users(users_per_round)
        allocations = allocator.allocate(users, simulator)
        records = simulator.simulate_batch(users, allocations)
        all_records.extend(records)
        
        # Update dual variables after each round
        allocator.update_dual_variables()
        
        if verbose and (round_idx + 1) % 10 == 0:
            print(f"  Round {round_idx + 1}/{n_rounds}")
    
    # Compute metrics
    base_metrics = simulator.compute_metrics()
    constraint_metrics = evaluate_constraint_satisfaction(simulator, allocator, all_records)
    
    result = {**base_metrics, **constraint_metrics}
    
    if track_lambda:
        result['lambda_history'] = allocator.lambda_history
        result['lambda_final'] = allocator.get_final_lambdas()
        result['lambda_stability'] = allocator.get_lambda_stability()
    
    return result


if __name__ == "__main__":
    from .simulator import GiftLiveSimulator, SimConfig, GreedyPolicy
    
    print("Testing Shadow Price Allocator...")
    
    config = SimConfig(n_users=5000, n_streamers=250, seed=42, enable_capacity=True)
    sim = GiftLiveSimulator(config)
    
    # Test 1: Greedy baseline
    print("\n1. Greedy baseline:")
    sim.reset()
    greedy = GreedyPolicy()
    metrics = sim.run_simulation(greedy, n_rounds=20, users_per_round=100)
    print(f"  Revenue: {metrics['total_revenue']:.2f}")
    print(f"  Cold-start rate: {metrics['cold_start_success_rate']:.4f}")
    
    # Test 2: Shadow Price with core constraints
    print("\n2. Shadow Price (core constraints):")
    shadow_config = ShadowPriceConfig(
        enable_capacity=True,
        enable_cold_start=True,
        enable_head_cap=True,
        enable_whale_spread=False,
        enable_frequency=False,
        eta=0.05
    )
    shadow = ShadowPriceAllocator(shadow_config)
    metrics = run_shadow_price_simulation(sim, shadow, n_rounds=20, users_per_round=100)
    print(f"  Revenue: {metrics['total_revenue']:.2f}")
    print(f"  Cold-start rate: {metrics['cold_start_success_rate']:.4f}")
    print(f"  Capacity satisfy: {metrics['capacity_satisfy_rate']:.4f}")
    print(f"  Lambda final: {metrics['lambda_final']}")
