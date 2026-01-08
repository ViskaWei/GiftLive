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
    
    # Gift amount model
    amount_mu0: float = 2.0
    amount_mu1: float = 0.8    # Wealth coefficient
    amount_mu2: float = 0.5    # Match coefficient
    amount_sigma: float = 0.5
    
    # Externality / diminishing returns
    gamma: float = 0.05
    
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
        
        for i in range(n):
            self.streamers.append(Streamer(
                streamer_id=i,
                content=contents[i],
                popularity=popularity[i],
                is_new=is_new_flags[i]
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
        """Sample gift amount given gifting occurred."""
        log_wealth = np.log1p(user.wealth)
        match = self.compute_match(user, streamer)
        
        log_amount = (
            self.config.amount_mu0 +
            self.config.amount_mu1 * log_wealth +
            self.config.amount_mu2 * match +
            self.rng.normal(0, self.config.amount_sigma)
        )
        
        return max(1.0, np.exp(log_amount))  # Minimum amount is 1
    
    def apply_diminishing_returns(self, amount: float, streamer: Streamer) -> float:
        """Apply diminishing returns based on crowding."""
        gamma = self.config.gamma
        return amount / (1.0 + gamma * streamer.n_big_donors)
    
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
        
        # Expected log amount
        log_wealth = np.log1p(user.wealth)
        match = self.compute_match(user, streamer)
        expected_log = (
            self.config.amount_mu0 +
            self.config.amount_mu1 * log_wealth +
            self.config.amount_mu2 * match
        )
        # E[exp(log_amount)] when log_amount ~ N(mu, sigma^2) is exp(mu + sigma^2/2)
        expected_amount = np.exp(expected_log + 0.5 * self.config.amount_sigma**2)
        expected_amount = max(1.0, expected_amount)
        
        # Apply diminishing returns
        gamma = self.config.gamma
        expected_amount = expected_amount / (1.0 + gamma * streamer.n_big_donors)
        
        return prob * expected_amount


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
        
        return {
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
        
        # Expected log amount
        expected_log = (
            self.config.amount_mu0 +
            self.config.amount_mu1 * log_wealth[:, np.newaxis] +
            self.config.amount_mu2 * match_scores
        )
        # E[exp(log_amount)] when log_amount ~ N(mu, sigma^2) is exp(mu + sigma^2/2)
        expected_amount = np.exp(expected_log + 0.5 * self.config.amount_sigma**2)
        expected_amount = np.maximum(1.0, expected_amount)
        
        # Apply diminishing returns
        gamma = self.config.gamma
        expected_amount = expected_amount / (1.0 + gamma * streamer_n_big[np.newaxis, :])
        
        return prob * expected_amount


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
