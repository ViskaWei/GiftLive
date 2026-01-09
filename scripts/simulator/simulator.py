#!/usr/bin/env python3
"""
GiftLive Simulator V1
Experiment ID: EXP-20260108-gift-allocation-07
MVP: MVP-0.3

A controllable simulator for live streaming gift behavior,
supporting allocation policy evaluation and OPE validation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
import json
from pathlib import Path


@dataclass
class SimConfig:
    """Simulator configuration."""
    # User pool
    n_users: int = 10000
    wealth_lognormal_mean: float = 3.0
    wealth_lognormal_std: float = 1.0
    wealth_pareto_alpha: float = 1.5
    wealth_pareto_min: float = 100.0
    wealth_pareto_weight: float = 0.05  # 5% are whales
    preference_dim: int = 16
    
    # Streamer pool
    n_streamers: int = 500
    popularity_pareto_alpha: float = 1.2
    content_dim: int = 16
    new_streamer_ratio: float = 0.2  # 20% are new streamers
    
    # Gift probability model
    gift_theta0: float = -5.0  # Base (very low)
    gift_theta1: float = 0.5   # Wealth coefficient
    gift_theta2: float = 1.0   # Match coefficient
    gift_theta3: float = 0.3   # Engagement coefficient
    gift_theta4: float = -0.1  # Crowding coefficient
    
    # Gift amount model (V1)
    amount_mu0: float = 2.0
    amount_mu1: float = 0.8    # Wealth coefficient
    amount_mu2: float = 0.5    # Match coefficient
    amount_sigma: float = 0.5
    
    # Gift amount model V2 (quantile-matched continuous)
    amount_version: int = 1  # 1 = V1 (original), 2 = V2 (lognormal+pareto), 3 = V2+ (discrete tiers)
    # V2 Lognormal parameters (for amounts < P90)
    v2_lognormal_mu: float = 0.693  # log(2) to match median=2
    v2_lognormal_sigma: float = 2.96  # calibrated for P90=88
    # V2 Pareto tail parameters (for amounts > P90)
    v2_pareto_threshold: float = 88.0  # P90 as threshold
    v2_pareto_alpha: float = 0.55  # shape parameter for heavy tail
    v2_tail_weight: float = 0.10  # probability of sampling from tail
    # V2 calibration targets (for reference)
    v2_target_median: float = 2.0
    v2_target_p90: float = 88.0
    v2_target_p99: float = 1488.0
    
    # Gift amount model V2+ (discrete tiers - new!)
    # Discrete gift tier values (real platform gift prices)
    v2plus_tiers: tuple = (1, 2, 10, 52, 100, 520, 1000, 1314, 5200, 13140)
    # Base tier probabilities (calibrated to match P50=2, P90=88, P99=1488)
    # v3 results: P50=2.0 ✓, P90=125.8 (need 88), P99=1498.7 ✓, Mean=94.2 ✓
    # Need to reduce 100/520/1000 tiers to bring P90 down
    # Recalibrated weights v4:
    v2plus_tier_probs: tuple = (0.38, 0.32, 0.15, 0.08, 0.035, 0.022, 0.008, 0.003, 0.0015, 0.0005)
    # Wealth effect on tier selection (higher wealth -> shift to higher tiers)
    v2plus_wealth_scale: float = 0.10
    # Match effect on tier selection
    v2plus_match_scale: float = 0.05
    
    # Externality / diminishing returns
    gamma: float = 0.05
    
    # MVP-4.2: Concurrency capacity model
    enable_capacity: bool = False  # Enable capacity constraints
    capacity_top10: int = 100      # Capacity for top 10% streamers
    capacity_middle: int = 50      # Capacity for middle tier
    capacity_tail: int = 20        # Capacity for tail streamers
    crowding_penalty_alpha: float = 0.5  # Penalty strength when over capacity
    
    # Random seed
    seed: int = 42


@dataclass
class User:
    """User entity."""
    user_id: int
    wealth: float
    preference: np.ndarray
    
    
@dataclass
class Streamer:
    """Streamer entity."""
    streamer_id: int
    content: np.ndarray
    popularity: float
    is_new: bool = False
    cumulative_revenue: float = 0.0
    n_big_donors: int = 0  # Number of big donors (for crowding effect)
    # MVP-4.2: Concurrency capacity
    capacity: int = 50  # Max concurrent users
    n_current_users: int = 0  # Current concurrent users this round


class UserPool:
    """User pool with wealth and preference distribution."""
    
    def __init__(self, config: SimConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.users: List[User] = []
        self._generate_users()
    
    def _generate_users(self):
        """Generate users with wealth and preference vectors."""
        n = self.config.n_users
        
        # Wealth: mixture of lognormal (95%) and pareto (5%)
        n_pareto = int(n * self.config.wealth_pareto_weight)
        n_lognormal = n - n_pareto
        
        wealth_lognormal = self.rng.lognormal(
            mean=self.config.wealth_lognormal_mean,
            sigma=self.config.wealth_lognormal_std,
            size=n_lognormal
        )
        wealth_pareto = (self.rng.pareto(self.config.wealth_pareto_alpha, size=n_pareto) + 1) * self.config.wealth_pareto_min
        
        wealth = np.concatenate([wealth_lognormal, wealth_pareto])
        self.rng.shuffle(wealth)
        
        # Preference vectors (unit normalized)
        preferences = self.rng.normal(0, 1, size=(n, self.config.preference_dim))
        preferences = preferences / np.linalg.norm(preferences, axis=1, keepdims=True)
        
        for i in range(n):
            self.users.append(User(
                user_id=i,
                wealth=wealth[i],
                preference=preferences[i]
            ))
    
    def get_user(self, user_id: int) -> User:
        return self.users[user_id]
    
    def sample_users(self, n: int) -> List[User]:
        """Sample n users without replacement."""
        indices = self.rng.choice(len(self.users), size=min(n, len(self.users)), replace=False)
        return [self.users[i] for i in indices]


class StreamerPool:
    """Streamer pool with content and popularity distribution."""
    
    def __init__(self, config: SimConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.streamers: List[Streamer] = []
        self._generate_streamers()
    
    def _generate_streamers(self):
        """Generate streamers with content vectors and popularity."""
        n = self.config.n_streamers
        
        # Popularity: pareto distribution (head effect)
        popularity = (self.rng.pareto(self.config.popularity_pareto_alpha, size=n) + 1)
        popularity = popularity / popularity.max()  # Normalize to [0, 1]
        
        # Content vectors (unit normalized)
        contents = self.rng.normal(0, 1, size=(n, self.config.content_dim))
        contents = contents / np.linalg.norm(contents, axis=1, keepdims=True)
        
        # New streamer flag
        n_new = int(n * self.config.new_streamer_ratio)
        is_new_flags = np.array([True] * n_new + [False] * (n - n_new))
        self.rng.shuffle(is_new_flags)
        
        # Determine capacity based on popularity tier
        popularity_ranks = np.argsort(popularity)[::-1]  # Descending
        n_top10 = int(n * 0.1)
        n_middle = int(n * 0.5)
        
        for i in range(n):
            # Assign capacity based on popularity tier
            rank = np.where(popularity_ranks == i)[0][0]
            if rank < n_top10:
                capacity = self.config.capacity_top10
            elif rank < n_top10 + n_middle:
                capacity = self.config.capacity_middle
            else:
                capacity = self.config.capacity_tail
            
            self.streamers.append(Streamer(
                streamer_id=i,
                content=contents[i],
                popularity=popularity[i],
                is_new=is_new_flags[i],
                capacity=capacity
            ))
    
    def get_streamer(self, streamer_id: int) -> Streamer:
        return self.streamers[streamer_id]
    
    def get_all_streamers(self) -> List[Streamer]:
        return self.streamers
    
    def reset_cumulative_stats(self):
        """Reset cumulative revenue and donor counts."""
        for s in self.streamers:
            s.cumulative_revenue = 0.0
            s.n_big_donors = 0
            s.n_current_users = 0
    
    def reset_round_stats(self):
        """Reset per-round stats (concurrent users)."""
        for s in self.streamers:
            s.n_current_users = 0


class GiftModel:
    """Gift probability and amount model."""
    
    def __init__(self, config: SimConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
    
    def compute_match(self, user: User, streamer: Streamer) -> float:
        """Compute user-streamer match score (dot product)."""
        return np.dot(user.preference, streamer.content)
    
    def compute_engagement(self, user: User, streamer: Streamer) -> float:
        """Compute engagement score (placeholder, could be more complex)."""
        # Simple: popularity-based engagement
        return streamer.popularity
    
    def gift_probability(self, user: User, streamer: Streamer) -> float:
        """Compute probability of gifting."""
        log_wealth = np.log1p(user.wealth)
        match = self.compute_match(user, streamer)
        engage = self.compute_engagement(user, streamer)
        crowd = streamer.n_big_donors
        
        logit = (
            self.config.gift_theta0 +
            self.config.gift_theta1 * log_wealth +
            self.config.gift_theta2 * match +
            self.config.gift_theta3 * engage +
            self.config.gift_theta4 * crowd
        )
        
        prob = 1.0 / (1.0 + np.exp(-logit))
        return prob
    
    def gift_amount(self, user: User, streamer: Streamer) -> float:
        """Sample gift amount given gifting occurred.
        
        V1: Original lognormal model with user/match features
        V2: Quantile-matched mixture (Lognormal + Pareto tail)
        V2+: Discrete tiers with wealth/match influence
        """
        if self.config.amount_version == 3:
            return self._gift_amount_v2plus(user, streamer)
        elif self.config.amount_version == 2:
            return self._gift_amount_v2(user, streamer)
        else:
            return self._gift_amount_v1(user, streamer)
    
    def _gift_amount_v1(self, user: User, streamer: Streamer) -> float:
        """V1: Original lognormal model."""
        log_wealth = np.log1p(user.wealth)
        match = self.compute_match(user, streamer)
        
        log_amount = (
            self.config.amount_mu0 +
            self.config.amount_mu1 * log_wealth +
            self.config.amount_mu2 * match +
            self.rng.normal(0, self.config.amount_sigma)
        )
        
        return max(1.0, np.exp(log_amount))  # Minimum amount is 1
    
    def _gift_amount_v2(self, user: User, streamer: Streamer) -> float:
        """V2: Quantile-matched mixture (Lognormal + Pareto tail).
        
        Calibrated to match real data:
        - P50 = 2.0
        - P90 = 88.0
        - P99 = 1488.0
        """
        # Decide: sample from body (Lognormal) or tail (Pareto)
        if self.rng.random() < self.config.v2_tail_weight:
            # Sample from Pareto tail (for large amounts)
            # Pareto: X = threshold * U^(-1/alpha), where U ~ Uniform(0,1)
            u = self.rng.random()
            # Ensure we don't get infinity
            u = max(u, 1e-10)
            amount = self.config.v2_pareto_threshold * (u ** (-1.0 / self.config.v2_pareto_alpha))
            # Cap at reasonable maximum
            amount = min(amount, 100000.0)
        else:
            # Sample from Lognormal body (for small/medium amounts)
            log_amount = self.rng.normal(
                self.config.v2_lognormal_mu,
                self.config.v2_lognormal_sigma
            )
            amount = np.exp(log_amount)
            # Truncate at threshold to avoid overlap with tail
            amount = min(amount, self.config.v2_pareto_threshold)
        
        # Add slight user/match effect (scaled down to preserve calibration)
        log_wealth = np.log1p(user.wealth)
        match = self.compute_match(user, streamer)
        # Small multiplier based on wealth and match (centered around 1.0)
        multiplier = 1.0 + 0.05 * (log_wealth / 5.0 - 1.0) + 0.02 * match
        multiplier = max(0.5, min(2.0, multiplier))  # Clamp to [0.5, 2.0]
        
        amount = amount * multiplier
        
        return max(1.0, amount)  # Minimum amount is 1
    
    def _gift_amount_v2plus(self, user: User, streamer: Streamer) -> float:
        """V2+: Discrete tier sampling with wealth/match influence.
        
        Samples from discrete gift tiers (1, 2, 10, 52, 100, 520, 1000, 1314, 5200, 13140)
        with probabilities adjusted based on user wealth and user-streamer match.
        
        Calibrated to match real data:
        - P50 ≈ 2.0
        - P90 ≈ 88-100
        - P99 ≈ 1000-1500
        """
        tiers = np.array(self.config.v2plus_tiers)
        base_probs = np.array(self.config.v2plus_tier_probs)
        
        # Compute user wealth effect (normalized)
        log_wealth = np.log1p(user.wealth)
        wealth_percentile = (log_wealth - 3.0) / 3.0  # Roughly normalize to [-1, 1]
        wealth_percentile = np.clip(wealth_percentile, -1.0, 2.0)
        
        # Compute match effect
        match = self.compute_match(user, streamer)
        match_effect = match  # Already roughly in [-1, 1]
        
        # Adjust tier log-probabilities (shift distribution for wealthier users)
        tier_indices = np.arange(len(tiers))
        # Higher wealth/match -> increase probability of higher tiers
        log_probs = np.log(base_probs + 1e-10)
        shift = (
            self.config.v2plus_wealth_scale * wealth_percentile * tier_indices +
            self.config.v2plus_match_scale * match_effect * tier_indices
        )
        adjusted_log_probs = log_probs + shift
        
        # Convert back to probabilities (softmax)
        adjusted_probs = np.exp(adjusted_log_probs - adjusted_log_probs.max())
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
        
        # Sample tier
        tier_idx = self.rng.choice(len(tiers), p=adjusted_probs)
        amount = float(tiers[tier_idx])
        
        return amount
    
    def apply_diminishing_returns(self, amount: float, streamer: Streamer) -> float:
        """Apply diminishing returns based on crowding and capacity (MVP-4.2)."""
        gamma = self.config.gamma
        base_penalty = 1.0 / (1.0 + gamma * streamer.n_big_donors)
        
        # MVP-4.2: Capacity-based crowding penalty
        if self.config.enable_capacity:
            capacity_penalty = self._compute_capacity_penalty(streamer)
            return amount * base_penalty * capacity_penalty
        else:
            return amount * base_penalty
    
    def _compute_capacity_penalty(self, streamer: Streamer) -> float:
        """Compute penalty when streamer is over capacity.
        
        Penalty = 1.0 if n_current <= capacity
        Penalty = 1 / (1 + alpha * (n_current - capacity) / capacity) otherwise
        """
        if streamer.n_current_users <= streamer.capacity:
            return 1.0
        else:
            overflow_ratio = (streamer.n_current_users - streamer.capacity) / streamer.capacity
            return 1.0 / (1.0 + self.config.crowding_penalty_alpha * overflow_ratio)
    
    def simulate_interaction(self, user: User, streamer: Streamer) -> Tuple[bool, float]:
        """Simulate a single user-streamer interaction.
        
        Returns:
            (did_gift, amount): Whether gift occurred and the amount (0 if no gift)
        """
        prob = self.gift_probability(user, streamer)
        did_gift = self.rng.random() < prob
        
        if did_gift:
            amount = self.gift_amount(user, streamer)
            amount = self.apply_diminishing_returns(amount, streamer)
            return True, amount
        else:
            return False, 0.0
    
    def expected_value(self, user: User, streamer: Streamer) -> float:
        """Compute expected gift value E[gift_prob * gift_amount]."""
        prob = self.gift_probability(user, streamer)
        
        if self.config.amount_version == 2:
            expected_amount = self._expected_amount_v2(user, streamer)
        else:
            expected_amount = self._expected_amount_v1(user, streamer)
        
        # Apply diminishing returns
        gamma = self.config.gamma
        expected_amount = expected_amount / (1.0 + gamma * streamer.n_big_donors)
        
        return prob * expected_amount
    
    def _expected_amount_v1(self, user: User, streamer: Streamer) -> float:
        """V1: Expected amount from lognormal model."""
        log_wealth = np.log1p(user.wealth)
        match = self.compute_match(user, streamer)
        expected_log = (
            self.config.amount_mu0 +
            self.config.amount_mu1 * log_wealth +
            self.config.amount_mu2 * match
        )
        # E[exp(log_amount)] when log_amount ~ N(mu, sigma^2) is exp(mu + sigma^2/2)
        expected_amount = np.exp(expected_log + 0.5 * self.config.amount_sigma**2)
        return max(1.0, expected_amount)
    
    def _expected_amount_v2(self, user: User, streamer: Streamer) -> float:
        """V2: Expected amount from mixture distribution.
        
        E[X] = (1-p) * E[Lognormal] + p * E[Pareto]
        where p = tail_weight
        """
        mu = self.config.v2_lognormal_mu
        sigma = self.config.v2_lognormal_sigma
        threshold = self.config.v2_pareto_threshold
        alpha = self.config.v2_pareto_alpha
        tail_weight = self.config.v2_tail_weight
        
        # E[Lognormal(mu, sigma)] = exp(mu + sigma^2/2)
        # But we truncate at threshold, so this is approximate
        e_lognormal = np.exp(mu + 0.5 * sigma**2)
        # Truncate expected value roughly
        e_lognormal = min(e_lognormal, threshold)
        
        # E[Pareto] with shape alpha < 1 is infinite, so use median instead
        # For alpha > 1: E[Pareto] = threshold * alpha / (alpha - 1)
        # For alpha <= 1: Use a capped approximation
        if alpha > 1:
            e_pareto = threshold * alpha / (alpha - 1)
        else:
            # Use P50 of Pareto as approximation: threshold * 2^(1/alpha)
            e_pareto = threshold * (2 ** (1.0 / alpha))
        e_pareto = min(e_pareto, 10000.0)  # Cap for stability
        
        # Mixture expected value
        expected_amount = (1 - tail_weight) * e_lognormal + tail_weight * e_pareto
        
        # Apply slight user effect
        log_wealth = np.log1p(user.wealth)
        match = self.compute_match(user, streamer)
        multiplier = 1.0 + 0.05 * (log_wealth / 5.0 - 1.0) + 0.02 * match
        multiplier = max(0.5, min(2.0, multiplier))
        
        return max(1.0, expected_amount * multiplier)


class GiftLiveSimulator:
    """Main simulator class."""
    
    def __init__(self, config: Optional[SimConfig] = None):
        self.config = config or SimConfig()
        self.rng = np.random.default_rng(self.config.seed)
        
        self.user_pool = UserPool(self.config, self.rng)
        self.streamer_pool = StreamerPool(self.config, self.rng)
        self.gift_model = GiftModel(self.config, self.rng)
        
        # Logging
        self.interaction_log: List[Dict] = []
    
    def reset(self):
        """Reset simulator state (cumulative stats, logs)."""
        self.streamer_pool.reset_cumulative_stats()
        self.interaction_log = []
    
    def simulate_batch(
        self,
        users: List[User],
        allocations: List[int],
        update_cumulative: bool = True
    ) -> List[Dict]:
        """Simulate a batch of user-streamer interactions.
        
        Args:
            users: List of users
            allocations: List of streamer IDs (one per user)
            update_cumulative: Whether to update cumulative stats
            
        Returns:
            List of interaction records
        """
        records = []
        
        # MVP-4.2: Reset per-round concurrent user counts
        if self.config.enable_capacity:
            self.streamer_pool.reset_round_stats()
            # Count concurrent users per streamer for this batch
            from collections import Counter
            allocation_counts = Counter(allocations)
            for streamer_id, count in allocation_counts.items():
                self.streamer_pool.get_streamer(streamer_id).n_current_users = count
        
        for user, streamer_id in zip(users, allocations):
            streamer = self.streamer_pool.get_streamer(streamer_id)
            did_gift, amount = self.gift_model.simulate_interaction(user, streamer)
            
            record = {
                'user_id': user.user_id,
                'streamer_id': streamer_id,
                'user_wealth': user.wealth,
                'streamer_is_new': streamer.is_new,
                'did_gift': did_gift,
                'amount': amount,
                'expected_value': self.gift_model.expected_value(user, streamer),
            }
            
            # MVP-4.2: Add capacity info to record
            if self.config.enable_capacity:
                record['streamer_capacity'] = streamer.capacity
                record['n_concurrent'] = streamer.n_current_users
                record['is_overcrowded'] = streamer.n_current_users > streamer.capacity
            
            records.append(record)
            
            if update_cumulative and did_gift:
                streamer.cumulative_revenue += amount
                if amount > 100:  # Big donor threshold
                    streamer.n_big_donors += 1
        
        self.interaction_log.extend(records)
        return records
    
    def run_simulation(
        self,
        policy: 'AllocationPolicy',
        n_rounds: int = 50,
        users_per_round: int = 200,
        verbose: bool = False
    ) -> Dict:
        """Run a full simulation with the given allocation policy.
        
        Args:
            policy: Allocation policy to use
            n_rounds: Number of rounds
            users_per_round: Users to allocate per round
            verbose: Print progress
            
        Returns:
            Simulation results
        """
        self.reset()
        
        for round_idx in range(n_rounds):
            users = self.user_pool.sample_users(users_per_round)
            allocations = policy.allocate(users, self)
            self.simulate_batch(users, allocations)
            
            if verbose and (round_idx + 1) % 10 == 0:
                print(f"  Round {round_idx + 1}/{n_rounds}")
        
        return self.compute_metrics()
    
    def compute_metrics(self) -> Dict:
        """Compute simulation metrics."""
        if not self.interaction_log:
            return {}
        
        # Convert log to arrays
        did_gift = np.array([r['did_gift'] for r in self.interaction_log])
        amounts = np.array([r['amount'] for r in self.interaction_log])
        streamer_ids = np.array([r['streamer_id'] for r in self.interaction_log])
        is_new = np.array([r['streamer_is_new'] for r in self.interaction_log])
        
        # Basic stats
        total_revenue = amounts.sum()
        n_gifts = did_gift.sum()
        gift_rate = n_gifts / len(did_gift) if len(did_gift) > 0 else 0
        
        # Amount distribution
        gift_amounts = amounts[amounts > 0]
        if len(gift_amounts) > 0:
            amount_mean = gift_amounts.mean()
            amount_median = np.median(gift_amounts)
            amount_p90 = np.percentile(gift_amounts, 90)
            amount_p99 = np.percentile(gift_amounts, 99)
        else:
            amount_mean = amount_median = amount_p90 = amount_p99 = 0
        
        # Streamer-level metrics
        streamer_revenue = {}
        for r in self.interaction_log:
            sid = r['streamer_id']
            if sid not in streamer_revenue:
                streamer_revenue[sid] = 0
            streamer_revenue[sid] += r['amount']
        
        revenues = np.array(list(streamer_revenue.values()))
        if len(revenues) > 0:
            streamer_gini = self._compute_gini(revenues)
            n_with_revenue = (revenues > 0).sum()
            coverage = n_with_revenue / self.config.n_streamers
            
            # Top share
            revenues_sorted = np.sort(revenues)[::-1]
            top_10_n = max(1, int(len(revenues) * 0.1))
            top_10_share = revenues_sorted[:top_10_n].sum() / revenues.sum() if revenues.sum() > 0 else 0
        else:
            streamer_gini = 0
            coverage = 0
            top_10_share = 0
        
        # User-level Gini
        user_spending = {}
        for r in self.interaction_log:
            uid = r['user_id']
            if uid not in user_spending:
                user_spending[uid] = 0
            user_spending[uid] += r['amount']
        
        user_amounts = np.array(list(user_spending.values()))
        user_gini = self._compute_gini(user_amounts) if len(user_amounts) > 0 else 0
        
        # Top user share
        if len(user_amounts) > 0 and user_amounts.sum() > 0:
            user_amounts_sorted = np.sort(user_amounts)[::-1]
            top_1_n = max(1, int(len(user_amounts) * 0.01))
            top_1_user_share = user_amounts_sorted[:top_1_n].sum() / user_amounts.sum()
        else:
            top_1_user_share = 0
        
        # Cold start metrics
        new_streamer_gifts = amounts[is_new & did_gift]
        new_streamer_revenue = new_streamer_gifts.sum()
        n_new_with_gift = len(set(
            r['streamer_id'] for r in self.interaction_log 
            if r['streamer_is_new'] and r['did_gift']
        ))
        n_new_total = sum(1 for s in self.streamer_pool.streamers if s.is_new)
        cold_start_success_rate = n_new_with_gift / n_new_total if n_new_total > 0 else 0
        
        result = {
            'total_revenue': float(total_revenue),
            'n_gifts': int(n_gifts),
            'gift_rate': float(gift_rate),
            'amount_mean': float(amount_mean),
            'amount_median': float(amount_median),
            'amount_p90': float(amount_p90),
            'amount_p99': float(amount_p99),
            'user_gini': float(user_gini),
            'streamer_gini': float(streamer_gini),
            'top_1_user_share': float(top_1_user_share),
            'top_10_streamer_share': float(top_10_share),
            'coverage': float(coverage),
            'cold_start_success_rate': float(cold_start_success_rate),
            'new_streamer_revenue': float(new_streamer_revenue),
        }
        
        # MVP-4.2: Capacity metrics
        if self.config.enable_capacity and 'is_overcrowded' in self.interaction_log[0]:
            overcrowded = np.array([r.get('is_overcrowded', False) for r in self.interaction_log])
            n_concurrent = np.array([r.get('n_concurrent', 0) for r in self.interaction_log])
            
            result['overcrowd_rate'] = float(overcrowded.mean())
            result['avg_concurrent'] = float(n_concurrent.mean())
            result['max_concurrent'] = int(n_concurrent.max())
            
            # Revenue from overcrowded vs normal interactions
            amounts_overcrowded = amounts[overcrowded & did_gift]
            amounts_normal = amounts[~overcrowded & did_gift]
            
            if len(amounts_overcrowded) > 0:
                result['avg_amount_overcrowded'] = float(amounts_overcrowded.mean())
            else:
                result['avg_amount_overcrowded'] = 0.0
            
            if len(amounts_normal) > 0:
                result['avg_amount_normal'] = float(amounts_normal.mean())
            else:
                result['avg_amount_normal'] = 0.0
        
        return result
    
    def _compute_gini(self, values: np.ndarray) -> float:
        """Compute Gini coefficient."""
        values = np.array(values, dtype=float)
        values = values[values > 0]
        if len(values) == 0:
            return 0.0
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)
        gini = (2 * np.sum((np.arange(1, n+1) * values)) - (n + 1) * np.sum(values)) / (n * np.sum(values))
        return gini
    
    def get_expected_values(self, users: List[User]) -> np.ndarray:
        """Compute expected value matrix for all user-streamer pairs (vectorized).
        
        Returns:
            (n_users, n_streamers) matrix of expected values
        """
        n_users = len(users)
        n_streamers = len(self.streamer_pool.streamers)
        
        # Vectorize user properties
        user_wealth = np.array([u.wealth for u in users])
        user_prefs = np.array([u.preference for u in users])  # (n_users, dim)
        
        # Vectorize streamer properties
        streamer_content = np.array([s.content for s in self.streamer_pool.streamers])  # (n_streamers, dim)
        streamer_popularity = np.array([s.popularity for s in self.streamer_pool.streamers])
        streamer_n_big = np.array([s.n_big_donors for s in self.streamer_pool.streamers])
        
        # Compute match scores: (n_users, n_streamers)
        match_scores = user_prefs @ streamer_content.T
        
        # Compute log wealth
        log_wealth = np.log1p(user_wealth)  # (n_users,)
        
        # Gift probability logit
        logit = (
            self.config.gift_theta0 +
            self.config.gift_theta1 * log_wealth[:, np.newaxis] +
            self.config.gift_theta2 * match_scores +
            self.config.gift_theta3 * streamer_popularity[np.newaxis, :] +
            self.config.gift_theta4 * streamer_n_big[np.newaxis, :]
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        
        # Expected amount based on version
        if self.config.amount_version == 2:
            expected_amount = self._get_expected_amount_v2_vectorized(
                log_wealth, match_scores, n_users, n_streamers
            )
        else:
            # V1: Original lognormal model
            expected_log = (
                self.config.amount_mu0 +
                self.config.amount_mu1 * log_wealth[:, np.newaxis] +
                self.config.amount_mu2 * match_scores
            )
            expected_amount = np.exp(expected_log + 0.5 * self.config.amount_sigma**2)
            expected_amount = np.maximum(1.0, expected_amount)
        
        # Apply diminishing returns
        gamma = self.config.gamma
        expected_amount = expected_amount / (1.0 + gamma * streamer_n_big[np.newaxis, :])
        
        return prob * expected_amount
    
    def _get_expected_amount_v2_vectorized(
        self, log_wealth: np.ndarray, match_scores: np.ndarray,
        n_users: int, n_streamers: int
    ) -> np.ndarray:
        """V2: Vectorized expected amount from mixture distribution."""
        mu = self.config.v2_lognormal_mu
        sigma = self.config.v2_lognormal_sigma
        threshold = self.config.v2_pareto_threshold
        alpha = self.config.v2_pareto_alpha
        tail_weight = self.config.v2_tail_weight
        
        # E[Lognormal]
        e_lognormal = np.exp(mu + 0.5 * sigma**2)
        e_lognormal = min(e_lognormal, threshold)
        
        # E[Pareto]
        if alpha > 1:
            e_pareto = threshold * alpha / (alpha - 1)
        else:
            e_pareto = threshold * (2 ** (1.0 / alpha))
        e_pareto = min(e_pareto, 10000.0)
        
        # Base expected amount
        base_amount = (1 - tail_weight) * e_lognormal + tail_weight * e_pareto
        
        # User multiplier (slight effect)
        multiplier = 1.0 + 0.05 * (log_wealth[:, np.newaxis] / 5.0 - 1.0) + 0.02 * match_scores
        multiplier = np.clip(multiplier, 0.5, 2.0)
        
        expected_amount = base_amount * multiplier
        return np.maximum(1.0, expected_amount)


class AllocationPolicy:
    """Base class for allocation policies."""
    
    def allocate(self, users: List[User], simulator: GiftLiveSimulator) -> List[int]:
        """Allocate users to streamers.
        
        Args:
            users: List of users to allocate
            simulator: Simulator instance (for accessing streamer info)
            
        Returns:
            List of streamer IDs (one per user)
        """
        raise NotImplementedError


class RandomPolicy(AllocationPolicy):
    """Random allocation policy."""
    
    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng(42)
    
    def allocate(self, users: List[User], simulator: GiftLiveSimulator) -> List[int]:
        n_streamers = simulator.config.n_streamers
        return self.rng.integers(0, n_streamers, size=len(users)).tolist()


class GreedyPolicy(AllocationPolicy):
    """Greedy allocation: argmax expected value."""
    
    def allocate(self, users: List[User], simulator: GiftLiveSimulator) -> List[int]:
        ev_matrix = simulator.get_expected_values(users)
        return np.argmax(ev_matrix, axis=1).tolist()


class RoundRobinPolicy(AllocationPolicy):
    """Round-robin allocation across streamers."""
    
    def __init__(self):
        self.counter = 0
    
    def allocate(self, users: List[User], simulator: GiftLiveSimulator) -> List[int]:
        n_streamers = simulator.config.n_streamers
        allocations = []
        for _ in users:
            allocations.append(self.counter % n_streamers)
            self.counter += 1
        return allocations


if __name__ == "__main__":
    # Quick test
    print("Testing GiftLiveSimulator...")
    
    config = SimConfig(n_users=1000, n_streamers=100, seed=42)
    sim = GiftLiveSimulator(config)
    
    policy = GreedyPolicy()
    metrics = sim.run_simulation(policy, n_rounds=10, users_per_round=100, verbose=True)
    
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
