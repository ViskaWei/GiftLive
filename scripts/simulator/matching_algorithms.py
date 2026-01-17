#!/usr/bin/env python3
"""
Matching Algorithms for Whale Distribution
MVP-5.3: b-matching, min-cost-flow, greedy-swaps

Implements bipartite matching algorithms for whale user allocation
with capacity constraints (each streamer can have at most k whales).
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict


def b_matching(
    ev_matrix: np.ndarray,
    k_per_streamer: int,
    whale_user_ids: List[int],
    streamer_ids: List[int]
) -> Dict[int, List[int]]:
    """
    Bipartite matching with capacity constraints using greedy approach.
    
    Left side: whale users (each can match to 1 streamer)
    Right side: streamers (each can match to at most k whales)
    Edge weights: EV(u, s)
    Goal: Maximize total weight
    
    Args:
        ev_matrix: (n_whales, n_streamers) expected value matrix
        k_per_streamer: Maximum number of whales per streamer
        whale_user_ids: List of whale user IDs (for output)
        streamer_ids: List of streamer IDs (for output)
    
    Returns:
        Dict[streamer_id, List[whale_user_id]]: Allocations
    """
    n_whales, n_streamers = ev_matrix.shape
    allocations = defaultdict(list)
    
    # Track remaining capacity for each streamer
    remaining_capacity = {s_id: k_per_streamer for s_id in streamer_ids}
    
    # Create list of (whale_idx, streamer_idx, ev) tuples, sorted by EV descending
    edges = []
    for w_idx in range(n_whales):
        for s_idx in range(n_streamers):
            edges.append((w_idx, s_idx, ev_matrix[w_idx, s_idx]))
    
    # Sort by EV descending
    edges.sort(key=lambda x: x[2], reverse=True)
    
    # Greedy matching: assign highest EV edges first, respecting capacity
    assigned_whales = set()
    for w_idx, s_idx, ev in edges:
        if w_idx in assigned_whales:
            continue
        s_id = streamer_ids[s_idx]
        if remaining_capacity[s_id] > 0:
            allocations[s_id].append(whale_user_ids[w_idx])
            assigned_whales.add(w_idx)
            remaining_capacity[s_id] -= 1
    
    return dict(allocations)


def greedy_with_swaps(
    ev_matrix: np.ndarray,
    k_per_streamer: int,
    whale_user_ids: List[int],
    streamer_ids: List[int],
    n_swaps: int = 100
) -> Dict[int, List[int]]:
    """
    Greedy matching with post-processing swaps to improve total EV.
    
    Args:
        ev_matrix: (n_whales, n_streamers) expected value matrix
        k_per_streamer: Maximum number of whales per streamer
        whale_user_ids: List of whale user IDs
        streamer_ids: List of streamer IDs
        n_swaps: Number of swap attempts
    
    Returns:
        Dict[streamer_id, List[whale_user_id]]: Allocations
    """
    # Initial greedy matching
    allocations = b_matching(ev_matrix, k_per_streamer, whale_user_ids, streamer_ids)
    
    # Build reverse mapping: whale -> streamer
    whale_to_streamer = {}
    for s_id, whales in allocations.items():
        for w_id in whales:
            whale_to_streamer[w_id] = s_id
    
    # Create index mappings
    whale_idx_map = {w_id: i for i, w_id in enumerate(whale_user_ids)}
    streamer_idx_map = {s_id: i for i, s_id in enumerate(streamer_ids)}
    
    # Try swaps to improve total EV
    for _ in range(n_swaps):
        # Randomly select two whales
        if len(whale_to_streamer) < 2:
            break
        
        whale_ids = list(whale_to_streamer.keys())
        w1_id, w2_id = np.random.choice(whale_ids, size=2, replace=False)
        s1_id = whale_to_streamer[w1_id]
        s2_id = whale_to_streamer[w2_id]
        
        # Skip if same streamer
        if s1_id == s2_id:
            continue
        
        # Check if swap improves total EV
        w1_idx = whale_idx_map[w1_id]
        w2_idx = whale_idx_map[w2_id]
        s1_idx = streamer_idx_map[s1_id]
        s2_idx = streamer_idx_map[s2_id]
        
        current_ev = ev_matrix[w1_idx, s1_idx] + ev_matrix[w2_idx, s2_idx]
        swapped_ev = ev_matrix[w1_idx, s2_idx] + ev_matrix[w2_idx, s1_idx]
        
        # If swap improves, perform it
        if swapped_ev > current_ev:
            # Remove from old allocations
            allocations[s1_id].remove(w1_id)
            allocations[s2_id].remove(w2_id)
            
            # Add to new allocations
            allocations[s2_id].append(w1_id)
            allocations[s1_id].append(w2_id)
            
            # Update mapping
            whale_to_streamer[w1_id] = s2_id
            whale_to_streamer[w2_id] = s1_id
    
    return dict(allocations)


def min_cost_flow_matching(
    ev_matrix: np.ndarray,
    k_per_streamer: int,
    whale_user_ids: List[int],
    streamer_ids: List[int]
) -> Dict[int, List[int]]:
    """
    Min-cost flow matching (using negative EV as cost to maximize).
    
    For simplicity, we use a greedy approximation since full min-cost flow
    requires network flow libraries. This is a placeholder that calls b_matching.
    
    In production, could use:
    - networkx.max_weight_matching with capacity constraints
    - ortools min-cost flow solver
    
    Args:
        ev_matrix: (n_whales, n_streamers) expected value matrix
        k_per_streamer: Maximum number of whales per streamer
        whale_user_ids: List of whale user IDs
        streamer_ids: List of streamer IDs
    
    Returns:
        Dict[streamer_id, List[whale_user_id]]: Allocations
    """
    # For now, use b_matching as approximation
    # TODO: Implement full min-cost flow using networkx or ortools
    return b_matching(ev_matrix, k_per_streamer, whale_user_ids, streamer_ids)
