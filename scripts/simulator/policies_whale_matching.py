#!/usr/bin/env python3
"""
Whale Matching Policy for GiftLive Simulator
MVP-5.3: Hierarchical matching (whale layer + normal layer)

Implements two-layer allocation:
1. Whale layer: b-matching/min-cost-flow with capacity constraint (k whales per streamer)
2. Normal layer: Greedy allocation for remaining capacity
"""

import numpy as np
from typing import List, Dict, Set, Optional
from collections import defaultdict

from .simulator import AllocationPolicy, User, Streamer, GiftLiveSimulator
from .matching_algorithms import b_matching, greedy_with_swaps, min_cost_flow_matching


class WhaleMatchingPolicy(AllocationPolicy):
    """Hierarchical matching policy with whale layer and normal layer."""
    
    def __init__(
        self,
        algorithm: str = "b-matching",
        k: int = 2,
        whale_threshold: str = "Top 1%"
    ):
        """
        Args:
            algorithm: "b-matching" | "min-cost-flow" | "greedy-swaps"
            k: Maximum number of whales per streamer
            whale_threshold: "Top 0.1%" | "Top 1%" | "Top 5%"
        """
        self.algorithm = algorithm
        self.k = k
        self.whale_threshold = whale_threshold
        
        # Parse threshold percentage
        if "0.1%" in whale_threshold:
            self.whale_percentile = 0.001
        elif "1%" in whale_threshold:
            self.whale_percentile = 0.01
        elif "5%" in whale_threshold:
            self.whale_percentile = 0.05
        else:
            raise ValueError(f"Unknown whale_threshold: {whale_threshold}")
    
    def identify_whales(self, users: List[User], simulator: GiftLiveSimulator) -> Set[int]:
        """
        Identify whale users based on wealth (proxy for cumulative revenue).
        
        Note: We use wealth as a proxy since cumulative_revenue is not tracked per user.
        Wealth and cumulative revenue should be highly correlated.
        
        Args:
            users: List of users in current round
            simulator: Simulator instance (for accessing all users if needed)
        
        Returns:
            Set of whale user IDs
        """
        # Get wealth for all users in the pool
        all_users = simulator.user_pool.users
        wealths = np.array([u.wealth for u in all_users])
        
        # Sort by wealth descending
        sorted_indices = np.argsort(wealths)[::-1]
        
        # Get top X% as whales
        n_whales = max(1, int(len(all_users) * self.whale_percentile))
        whale_indices = sorted_indices[:n_whales]
        whale_ids = {all_users[i].user_id for i in whale_indices}
        
        # Filter to only users in current round
        current_user_ids = {u.user_id for u in users}
        return whale_ids & current_user_ids
    
    def whale_layer_matching(
        self,
        whale_users: List[User],
        simulator: GiftLiveSimulator
    ) -> Dict[int, List[int]]:
        """
        Match whale users to streamers using specified algorithm.
        
        Args:
            whale_users: List of whale users
            simulator: Simulator instance
        
        Returns:
            Dict[streamer_id, List[whale_user_id]]: Whale allocations
        """
        if not whale_users:
            return {}
        
        # Get EV matrix for whale users
        ev_matrix = simulator.get_expected_values(whale_users)
        n_whales, n_streamers = ev_matrix.shape
        
        whale_user_ids = [u.user_id for u in whale_users]
        streamer_ids = [s.streamer_id for s in simulator.streamer_pool.streamers]
        
        # Call appropriate matching algorithm
        if self.algorithm == "b-matching":
            allocations = b_matching(
                ev_matrix, self.k, whale_user_ids, streamer_ids
            )
        elif self.algorithm == "greedy-swaps":
            allocations = greedy_with_swaps(
                ev_matrix, self.k, whale_user_ids, streamer_ids
            )
        elif self.algorithm == "min-cost-flow":
            allocations = min_cost_flow_matching(
                ev_matrix, self.k, whale_user_ids, streamer_ids
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        return allocations
    
    def normal_layer_matching(
        self,
        normal_users: List[User],
        remaining_capacity: Dict[int, int],
        simulator: GiftLiveSimulator
    ) -> List[int]:
        """
        Greedy allocation for normal users with remaining capacity.
        
        Args:
            normal_users: List of normal (non-whale) users
            remaining_capacity: Dict[streamer_id, remaining_capacity]
            simulator: Simulator instance
        
        Returns:
            List[streamer_id]: Allocations for normal users
        """
        if not normal_users:
            return []
        
        ev_matrix = simulator.get_expected_values(normal_users)
        n_users, n_streamers = ev_matrix.shape
        
        allocations = []
        streamer_ids = [s.streamer_id for s in simulator.streamer_pool.streamers]
        
        for i in range(n_users):
            # Get EV for this user
            user_evs = ev_matrix[i]
            
            # Mask out streamers with no remaining capacity
            valid_mask = np.array([
                remaining_capacity.get(s_id, 0) > 0 for s_id in streamer_ids
            ])
            
            if valid_mask.any():
                masked_ev = user_evs.copy()
                masked_ev[~valid_mask] = -np.inf
                best_streamer_idx = int(np.argmax(masked_ev))
                best_streamer_id = streamer_ids[best_streamer_idx]
                
                allocations.append(best_streamer_id)
                
                # Update remaining capacity
                remaining_capacity[best_streamer_id] -= 1
            else:
                # No capacity left, assign to first available (shouldn't happen often)
                allocations.append(streamer_ids[0])
        
        return allocations
    
    def allocate(self, users: List[User], simulator: GiftLiveSimulator) -> List[int]:
        """
        Two-layer allocation: whale layer + normal layer.
        
        Args:
            users: List of users to allocate
            simulator: Simulator instance
        
        Returns:
            List[streamer_id]: Allocations for each user
        """
        # 1. Identify whales
        whale_ids = self.identify_whales(users, simulator)
        whale_users = [u for u in users if u.user_id in whale_ids]
        normal_users = [u for u in users if u.user_id not in whale_ids]
        
        # 2. Whale layer matching
        whale_allocations = self.whale_layer_matching(whale_users, simulator)
        
        # 3. Compute remaining capacity
        streamer_ids = [s.streamer_id for s in simulator.streamer_pool.streamers]
        remaining_capacity = {
            s.streamer_id: s.capacity for s in simulator.streamer_pool.streamers
        }
        
        # Subtract whale allocations from capacity
        for s_id, whale_list in whale_allocations.items():
            remaining_capacity[s_id] -= len(whale_list)
            remaining_capacity[s_id] = max(0, remaining_capacity[s_id])
        
        # 4. Normal layer matching
        normal_allocations = self.normal_layer_matching(
            normal_users, remaining_capacity, simulator
        )
        
        # 5. Merge allocations
        # Build mapping: user_id -> streamer_id
        user_to_streamer = {}
        
        # Add whale allocations
        for s_id, whale_list in whale_allocations.items():
            for w_id in whale_list:
                user_to_streamer[w_id] = s_id
        
        # Add normal allocations
        for i, s_id in enumerate(normal_allocations):
            user_id = normal_users[i].user_id
            user_to_streamer[user_id] = s_id
        
        # Convert to list in user order
        allocations = [user_to_streamer[u.user_id] for u in users]
        
        return allocations
