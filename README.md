# Economic Algorithms Solutions

Solutions for economic algorithm problems, including resource division using convex optimization and envy-free allocations.

## Structure

- `01/` - Problem set 1
- `02/` - Problem set 2: Egalitarian and leximin-egalitarian division using convex optimization
- `03/` - Problem set 3
- `04/` - Problem set 4: Envy-free room allocation using the heaviest-path pricing method

## Requirements

- Python 3.x
- `cvxpy`
- `numpy`

## Key Implementations

### Problem Set 02: Egalitarian Division

Implements egalitarian and leximin-egalitarian division algorithms that allocate resources to maximize the minimum utility among agents.

```bash
python 02/question3.py
```

### Problem Set 04: Envy-Free Room Allocation

Implements the heaviest-path pricing method for envy-free room allocation, ensuring no player envies another player's room-price combination.

```bash
python 04/question5.py
```

## Testing

Run tests for specific problem sets:

```bash
python 02/test_question3.py
python 04/test_question5.py
```
