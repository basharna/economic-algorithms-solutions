# Assignment 04: Envy-Free Room Allocation

## Overview

This assignment implements an algorithm for envy-free room allocation using the **heaviest-path pricing method**. The algorithm assigns rooms to players and determines prices such that no player envies another player's room-price combination.

## Implementation

The solution (`question5.py`) implements the algorithm in four main steps:

1. **Maximum-Weight Assignment**: Find an optimal assignment of players to rooms using the Hungarian algorithm
2. **Envy Graph Construction**: Build a graph representing how much each player envies others' allocations
3. **Heaviest Path Computation**: Calculate the weight of the heaviest path from each vertex in the adjusted envy graph (where edges are reduced by the maximum mean cycle weight)
4. **Price Calculation**: Compute room prices that sum to the total rent and guarantee envy-freeness

## Files

- `question5.py` - Main implementation of the envy-free room allocation algorithm
- `test_question5.py` - Unit tests validating the algorithm on various test cases

## Usage

```python
from question5 import envy_free_room_allocation

valuations = [[150, 0], [140, 10]]
rent = 100
assignment, prices = envy_free_room_allocation(valuations, rent)
```

## Testing

Run the tests with:

```bash
python test_question5.py
```

---

**Note 1:** Graph algorithms (Hungarian algorithm, Karp's algorithm for maximum mean cycle, and Bellman-Ford style path computation) were obtained from ChatGPT.

**Note 2:** Code was made cleaner and less cluttered using AI assistance.
