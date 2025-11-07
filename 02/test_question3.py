#!/usr/bin/env python3

"""
Test file for Question 3 Part A: Egalitarian Division

This file contains additional test cases to verify the egalitarian_division function.
"""

from question3 import egalitarian_division, leximin_egalitarian_division
import numpy as np


def test_example_from_question():
    """Test with the exact example from the question."""
    print("="*70)
    print("Test 1: Example from question3")
    print("="*70)
    valuations = [[81, 19, 1], [70, 1, 29]]
    print(f"Valuations:\n  Agent #1: {valuations[0]}\n  Agent #2: {valuations[1]}\n")
    
    allocation = egalitarian_division(valuations, verbose=False)
    
    # Calculate utilities
    valuations_np = np.array(valuations)
    utilities = [(allocation[i] * valuations_np[i]).sum() for i in range(len(valuations))]
    print(f"\nUtilities: Agent #1: {utilities[0]:.2f}, Agent #2: {utilities[1]:.2f}")
    print(f"Minimum utility (maximized): {min(utilities):.2f}")
    print()


def test_symmetric_case():
    """Test with symmetric valuations - should give equal split."""
    print("="*70)
    print("Test 2: Symmetric case (both agents have identical valuations)")
    print("="*70)
    valuations = [[50, 30, 20], [50, 30, 20]]
    print(f"Valuations:\n  Agent #1: {valuations[0]}\n  Agent #2: {valuations[1]}\n")
    
    allocation = egalitarian_division(valuations, verbose=False)
    
    # Calculate utilities
    valuations_np = np.array(valuations)
    utilities = [(allocation[i] * valuations_np[i]).sum() for i in range(len(valuations))]
    print(f"\nUtilities: Agent #1: {utilities[0]:.2f}, Agent #2: {utilities[1]:.2f}")
    print(f"Expected: Both should get equal utility of 50.0")
    print()


def test_complementary_preferences():
    """Test with complementary preferences - should be easy to split."""
    print("="*70)
    print("Test 3: Complementary preferences (no conflicts)")
    print("="*70)
    valuations = [[100, 0, 0], [0, 100, 0], [0, 0, 100]]
    print(f"Valuations:\n  Agent #1: {valuations[0]}\n  Agent #2: {valuations[1]}\n  Agent #3: {valuations[2]}\n")
    
    allocation = egalitarian_division(valuations, verbose=False)
    
    # Calculate utilities
    valuations_np = np.array(valuations)
    utilities = [(allocation[i] * valuations_np[i]).sum() for i in range(len(valuations))]
    print(f"\nUtilities: Agent #1: {utilities[0]:.2f}, Agent #2: {utilities[1]:.2f}, Agent #3: {utilities[2]:.2f}")
    print(f"Expected: Each agent should get their preferred resource entirely (utility = 100)")
    print()


def test_many_agents():
    """Test with more agents than resources."""
    print("="*70)
    print("Test 4: Four agents, three resources")
    print("="*70)
    valuations = [
        [60, 30, 10],
        [10, 60, 30],
        [30, 10, 60],
        [40, 40, 20]
    ]
    for i, vals in enumerate(valuations, 1):
        print(f"  Agent #{i}: {vals}")
    print()
    
    allocation = egalitarian_division(valuations, verbose=False)
    
    # Calculate utilities
    valuations_np = np.array(valuations)
    utilities = [(allocation[i] * valuations_np[i]).sum() for i in range(len(valuations))]
    print(f"\nUtilities:")
    for i, util in enumerate(utilities, 1):
        print(f"  Agent #{i}: {util:.2f}")
    print(f"Minimum utility (maximized): {min(utilities):.2f}")
    print()


def test_leximin_matches_egalitarian_on_two_agents():
    """Leximin equals egalitarian on the 2-agent question example."""
    print("="*70)
    print("Test 5: Leximin equals Egalitarian on 2-agent example")
    print("="*70)
    valuations = [[81, 19, 1], [70, 1, 29]]
    alloc_egal = egalitarian_division(valuations, verbose=False)
    alloc_lex = leximin_egalitarian_division(valuations, verbose=False)

    v = np.array(valuations)
    utils_egal = [(alloc_egal[i] * v[i]).sum() for i in range(len(valuations))]
    utils_lex  = [(alloc_lex[i]  * v[i]).sum() for i in range(len(valuations))]

    print(f"Utilities (egalitarian): {[f'{float(u):.6f}' for u in utils_egal]}")
    print(f"Utilities (leximin):    {[f'{float(u):.6f}' for u in utils_lex]}")

    # same minimal utility
    assert np.isclose(min(utils_egal), min(utils_lex), atol=1e-6)


def test_leximin_lexicographically_dominates_example_from_class():
    """Leximin should lexicographically dominate egalitarian on a 4-agent tie case."""
    print("="*70)
    print("Test 6: Leximin lexicographically dominates on 4-agent scenario")
    print("="*70)
    # Derived from code/3-leximin-sums.py utilities setup
    # Resource values per agent: wood, oil, steel
    valuations = [
        [4, 0, 0],   # Agent A values only wood
        [0, 3, 0],   # Agent B values only oil
        [5, 5, 10],  # Agent C values all, steel most
        [5, 5, 10],  # Agent D same as C
    ]

    alloc_egal = egalitarian_division(valuations, verbose=False)
    alloc_lex = leximin_egalitarian_division(valuations, verbose=False)

    v = np.array(valuations)
    utils_egal = np.array([(alloc_egal[i] * v[i]).sum() for i in range(len(valuations))])
    utils_lex  = np.array([(alloc_lex[i]  * v[i]).sum() for i in range(len(valuations))])

    # Sort ascending for lexicographic comparison of utility profiles
    a = np.sort(utils_egal)
    b = np.sort(utils_lex)

    print(f"Utilities (egalitarian, sorted): {a}")
    print(f"Utilities (leximin, sorted):    {b}")

    # Same minimal utility, and leximin lexicographically >= egalitarian
    assert np.isclose(a[0], b[0], atol=1e-6)
    # lexicographic dominance: at first differing index, b should be larger
    greater_found = False
    for i in range(len(a)):
        if not np.isclose(a[i], b[i], atol=1e-6):
            assert b[i] > a[i]
            greater_found = True
            break
    # If all equal within tolerance, that's acceptable; otherwise we found a strictly greater
    if not np.allclose(a, b, atol=1e-6):
        assert greater_found


if __name__ == "__main__":
    test_example_from_question()
    test_symmetric_case()
    test_complementary_preferences()
    test_many_agents()
    test_leximin_matches_egalitarian_on_two_agents()
    test_leximin_lexicographically_dominates_example_from_class()
    
    print("="*70)
    print("All tests completed successfully!")
    print("="*70)

