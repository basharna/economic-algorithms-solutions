#!/usr/bin/env python3

"""
Question 3: Division using cvxpy

DATE: 2025-11-07
"""

import cvxpy
import numpy as np


def egalitarian_division(valuations, verbose=True):
    """
    Finds an egalitarian division of resources among agents.
    
    Parameters:
    -----------
    valuations : list of lists or numpy array
        A matrix where valuations[i][j] represents the value that agent i 
        assigns to resource j.
        Shape: (n_agents, n_resources)
    
    Returns:
    --------
    allocation : numpy array
        A matrix where allocation[i][j] represents the fraction of resource j
        given to agent i. Shape: (n_agents, n_resources)
    """
    # Convert to numpy array for easier manipulation
    valuations = np.array(valuations, dtype=float)
    n_agents, n_resources = valuations.shape
    
    # Create decision variables: allocation[i][j] = fraction of resource j given to agent i
    allocation_vars = cvxpy.Variable((n_agents, n_resources))
    
    # Calculate utilities for each agent
    utilities = []
    for i in range(n_agents):
        utility = sum(allocation_vars[i, j] * valuations[i, j] for j in range(n_resources))
        utilities.append(utility)
    
    # Variable to represent the minimum utility (what we want to maximize)
    min_utility = cvxpy.Variable()
    
    # Constraints
    constraints = []
    
    # 1. All allocations must be between 0 and 1
    constraints.append(allocation_vars >= 0)
    constraints.append(allocation_vars <= 1)
    
    # 2. Each resource must be fully allocated (fractions for each resource sum to 1)
    for j in range(n_resources):
        constraints.append(sum(allocation_vars[i, j] for i in range(n_agents)) == 1)
    
    # 3. min_utility is the minimum of all utilities
    for utility in utilities:
        constraints.append(min_utility <= utility)
    
    # Define and solve the optimization problem
    problem = cvxpy.Problem(
        cvxpy.Maximize(min_utility),
        constraints=constraints
    )
    
    problem.solve()
    
    # Check if solution was found
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {problem.status}")
    
    # Extract the allocation matrix
    allocation = allocation_vars.value
    
    # Round very small values to 0 and values very close to 1 to 1.0 for cleaner output
    tolerance = 1e-6
    allocation = np.where(np.abs(allocation) < tolerance, 0, allocation)
    allocation = np.where(np.abs(allocation - 1) < tolerance, 1, allocation)
    
    # Print the results (optional)
    if verbose:
        for i in range(n_agents):
            parts = []
            for j in range(n_resources):
                value = allocation[i, j]
                # Format: show 0.0 and 1.0 explicitly, otherwise use 2 decimal places
                if value == 0:
                    value_str = "0.0"
                elif value == 1:
                    value_str = "1.0"
                else:
                    value_str = f"{value:.2f}"
                parts.append(f"{value_str} of resource #{j+1}")
            
            allocation_str = ", and ".join([", ".join(parts[:-1]), parts[-1]] if len(parts) > 1 else parts)
            print(f"Agent #{i+1} gets {allocation_str}.")
    
    return allocation


def leximin_egalitarian_division(valuations, verbose=True):
    """
    Finds a leximin-egalitarian division of resources among agents.

    Parameters:
    -----------
    valuations : list of lists or numpy array
        A matrix where valuations[i][j] represents the value that agent i
        assigns to resource j. Shape: (n_agents, n_resources)

    Returns:
    --------
    allocation : numpy array
        Fractions allocation matrix of shape (n_agents, n_resources)
    """
    from itertools import combinations

    valuations = np.array(valuations, dtype=float)
    n_agents, n_resources = valuations.shape

    # Decision variables
    allocation_vars = cvxpy.Variable((n_agents, n_resources))

    # Utilities per agent
    utilities = []
    for i in range(n_agents):
        utility = sum(allocation_vars[i, j] * valuations[i, j] for j in range(n_resources))
        utilities.append(utility)

    # Base/fixed constraints (independent of objective stage)
    fixed_constraints = [
        allocation_vars >= 0,
        allocation_vars <= 1,
    ]
    for j in range(n_resources):
        fixed_constraints.append(sum(allocation_vars[i, j] for i in range(n_agents)) == 1)

    # Store the optimal values of k-sum minima from previous iterations
    ksum_targets = []  # list of floats: [t1, t2, ..., tk-1]

    # Iterate k from 1 to n_agents, maximizing the minimum sum over any k agents
    for k in range(1, n_agents + 1):
        t_k = cvxpy.Variable()

        constraints = list(fixed_constraints)

        # Enforce previously achieved minima for all smaller sizes
        # For k' = 1..k-1, require: t_{k'} <= sum(u_i for i in any k' agents)
        # Using the stored numeric values from previous solves.
        for kprime, t_kprime_value in enumerate(ksum_targets, start=1):
            if kprime == 1:
                # sums of 1 are just individual utilities
                constraints += [t_kprime_value <= u for u in utilities]
            else:
                for idxs in combinations(range(n_agents), kprime):
                    constraints.append(t_kprime_value <= sum(utilities[i] for i in idxs))

        # New constraints for current k: t_k <= sum of any k utilities
        if k == 1:
            constraints += [t_k <= u for u in utilities]
        else:
            for idxs in combinations(range(n_agents), k):
                constraints.append(t_k <= sum(utilities[i] for i in idxs))

        problem = cvxpy.Problem(cvxpy.Maximize(t_k), constraints=constraints)
        problem.solve()

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Leximin optimization failed at k={k} with status: {problem.status}")

        # Record achieved target for this k
        ksum_targets.append(t_k.value)

    # Extract allocation and print formatted result
    allocation = allocation_vars.value

    tolerance = 1e-6
    allocation = np.where(np.abs(allocation) < tolerance, 0, allocation)
    allocation = np.where(np.abs(allocation - 1) < tolerance, 1, allocation)

    if verbose:
        for i in range(n_agents):
            parts = []
            for j in range(n_resources):
                value = allocation[i, j]
                if value == 0:
                    value_str = "0.0"
                elif value == 1:
                    value_str = "1.0"
                else:
                    value_str = f"{value:.2f}"
                parts.append(f"{value_str} of resource #{j+1}")
            allocation_str = ", and ".join([", ".join(parts[:-1]), parts[-1]] if len(parts) > 1 else parts)
            print(f"Agent #{i+1} gets {allocation_str}.")

    return allocation


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    print("\n" + "="*70)
    print("test with the exact example from the question:")
    print("="*70)
    valuations = [[81, 19, 1], [70, 1, 29]]
    print(f"Valuations: {valuations}")
    print()
    allocation = egalitarian_division(valuations)
    print(f"\nAllocation matrix:\n{allocation}")
    
    # Calculate and print utilities
    valuations_np = np.array(valuations)
    utilities = [(allocation[i] * valuations_np[i]).sum() for i in range(len(valuations))]
    print(f"\nUtilities: {utilities}")
    print(f"Minimum utility: {min(utilities):.2f}")

    print("\n" + "="*70)
    print("Leximin-egalitarian division for the same instance:")
    print("="*70)
    allocation_lex = leximin_egalitarian_division(valuations)
    print(f"\nAllocation matrix (leximin):\n{allocation_lex}")

