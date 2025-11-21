#!/usr/bin/env python3
"""
Question 5: Envy-free room allocation via the "heaviest path" pricing method.

1.   Find an assignment of players to rooms that maximizes the total value
     (maximum-weight perfect matching in a complete bipartite graph).
2.   Build the envy graph over players, compute the maximum mean cycle weight W,
     and create the adjusted graph where each edge weight is reduced by W.
3.   For every player, compute the weight of the heaviest path that starts at
     that player in the adjusted graph. These weights play the role of the
     subsidies q_i in the proof.
4.   Charge every player the common baseline c = (rent + sum(q_i)) / n and
     subtract their own subsidy. The resulting payment vector induces prices for
     the rooms that sum exactly to the rent and guarantee envy-freeness.
"""

from __future__ import annotations

from math import inf
from typing import List, Sequence, Tuple

import numpy as np


def envy_free_room_allocation(valuations: Sequence[Sequence[float]], rent: float, verbose: bool = True) -> Tuple[List[int], List[float]]:
    """
    Compute an envy-free room allocation using the heaviest-path pricing method.

    Parameters
    ----------
    valuations:
        Square matrix (n x n). valuations[i][j] is the monetary value player i assigns to room j.
        Utilities are quasilinear: u_i(j, p_j) = valuations[i][j] - p_j.
    rent:
        Total rent R that must be collected from all rooms combined.
    verbose:
        When True, print a human-readable description of the outcome.

    Returns
    -------
    assignment:
        List where assignment[i] = j means player i receives room j.
    prices:
        List where prices[j] is the rent charged for room j. Sum(prices) == rent (up to floating-point noise).
    """

    value_matrix = np.asarray(valuations, dtype=float)
    if value_matrix.ndim != 2 or value_matrix.shape[0] != value_matrix.shape[1]:
        raise ValueError("valuations must be a square matrix (same number of players and rooms)")
    n = value_matrix.shape[0]
    if n == 0:
        raise ValueError("valuations matrix must not be empty")

    assignment = _maximum_weight_assignment(value_matrix)
    envy = _build_envy_graph(value_matrix, assignment)
    w_star = _maximum_mean_cycle_weight(envy)
    adjusted_envy = envy - w_star
    subsidies = _heaviest_paths_from_each_vertex(adjusted_envy)
    payments = _payments_from_subsidies(subsidies, rent)
    prices = _prices_from_payments(payments, assignment, n)

    if verbose:
        for player, room in enumerate(assignment):
            value = value_matrix[player, room]
            payment = prices[room]
            print(f"Player {player} gets room {room} with value {value:.4f}, and pays {payment:.4f}")

    return assignment, prices


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #


def _maximum_weight_assignment(values: np.ndarray) -> List[int]:
    n = values.shape[0]
    max_value = np.max(values)
    cost_matrix = max_value - values
    assignment = _hungarian(cost_matrix)
    if sorted(assignment) != list(range(n)):
        raise AssertionError("Hungarian algorithm did not return a valid assignment")
    return assignment


def _hungarian(cost: np.ndarray) -> List[int]:
    """Hungarian algorithm (a.k.a. Kuhn-Munkres) for the square cost matrix."""
    n = cost.shape[0]
    # Convert to 1-based indexing for clarity compared to the classic pseudocode.
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        minv = np.full(n + 1, inf)
        used = np.zeros(n + 1, dtype=bool)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def _build_envy_graph(values: np.ndarray, assignment: Sequence[int]) -> np.ndarray:
    """Return the envy graph matrix E(i, j) = Vi(Xj) - Vi(Xi)."""
    n = len(assignment)
    envy = np.zeros((n, n), dtype=float)
    for i in range(n):
        own_room = assignment[i]
        own_value = values[i, own_room]
        for j in range(n):
            target_room = assignment[j]
            envy[i, j] = values[i, target_room] - own_value
    return envy


def _maximum_mean_cycle_weight(weights: np.ndarray) -> float:
    """Karp's algorithm for the maximum mean cycle weight in a dense graph."""
    n = weights.shape[0]
    dp = np.full((n + 1, n), -inf)
    dp[0, :] = 0.0

    for k in range(1, n + 1):
        for v in range(n):
            dp[k, v] = np.max(dp[k - 1, :] + weights[:, v])

    best = -inf
    for v in range(n):
        min_avg = inf
        for k in range(n):
            if dp[k, v] == -inf:
                continue
            avg = (dp[n, v] - dp[k, v]) / (n - k)
            if avg < min_avg:
                min_avg = avg
        if min_avg > best:
            best = min_avg
    return best


def _heaviest_paths_from_each_vertex(weights: np.ndarray) -> List[float]:
    """
    Given a graph with no positive cycles, compute the weight of the heaviest path
    that starts from every vertex (Bellman-Ford style relaxation).
    """
    n = weights.shape[0]
    subsidies: List[float] = []
    for source in range(n):
        dist = np.full(n, -inf)
        dist[source] = 0.0
        for _ in range(n - 1):
            updated = False
            for u in range(n):
                if dist[u] == -inf:
                    continue
                gained = dist[u] + weights[u]
                mask = gained > dist
                if np.any(mask):
                    dist[mask] = gained[mask]
                    updated = True
            if not updated:
                break
        subsidies.append(float(np.max(dist)))
    return subsidies


def _payments_from_subsidies(subsidies: Sequence[float], rent: float) -> List[float]:
    subsidies = np.asarray(subsidies, dtype=float)
    total_subsidy = float(np.sum(subsidies))
    common_charge = (rent + total_subsidy) / len(subsidies)
    return [common_charge - s for s in subsidies]


def _prices_from_payments(payments: Sequence[float], assignment: Sequence[int], n: int) -> List[float]:
    prices = [0.0] * n
    for player, room in enumerate(assignment):
        prices[room] = payments[player]
    return prices


if __name__ == "__main__":
    example_vals = [[150, 0], [140, 10]]
    example_rent = 100
    envy_free_room_allocation(example_vals, example_rent)

