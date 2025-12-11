from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass(frozen=True)
class PaymentResult:
    affordable: bool
    t: float  # final equal-payment level among non-broke supporters
    payments: Dict[int, float]  # citizen_index -> payment


def _equal_shares_payments(
    supporters: List[int],
    balances: List[float],
    cost: float,
    eps: float = 1e-12,
) -> PaymentResult:

    if cost <= eps:
        return PaymentResult(True, 0.0, {i: 0.0 for i in supporters})

    if not supporters:
        return PaymentResult(False, float("inf"), {})

    total_supporter_balance = sum(balances[i] for i in supporters)
    if total_supporter_balance + eps < cost:
        return PaymentResult(False, float("inf"), {})

    # Water-filling to find t.
    sup_sorted: List[Tuple[int, float]] = sorted(
        [(i, balances[i]) for i in supporters],
        key=lambda x: x[1],
    )

    remaining_cost = cost
    remaining_count = len(sup_sorted)
    prev = 0.0
    t = 0.0

    for _, b in sup_sorted:
        capacity = (b - prev) * remaining_count
        if remaining_cost <= capacity + eps:
            t = prev + remaining_cost / remaining_count
            remaining_cost = 0.0
            break

        remaining_cost -= capacity
        remaining_count -= 1
        prev = b

    if remaining_cost > eps:
        # Should still be solvable because total balance >= cost.
        t = prev + remaining_cost / remaining_count

    payments = {i: min(balances[i], t) for i in supporters}

    # Nudge away tiny FP drift so sums match cost.
    diff = cost - sum(payments.values())
    if abs(diff) > 1e-9:
        j = max(supporters, key=lambda i: payments[i])
        payments[j] = max(0.0, min(balances[j], payments[j] + diff))

    return PaymentResult(True, t, payments)


def elect_next_budget_item(
    votes: List[Set[str]],  # each citizen approves a set of items
    balances: List[float],  # virtual balance of each citizen
    costs: Dict[str, float],  # cost of each item
) -> None:
    """
    Performs ONE round of Equal Shares participatory budgeting:
    - For each item, check if its supporters can pay its cost under Equal Shares.
    - Elect the affordable item with minimal t (minimal burden level).
    - Print payments and updated balances.

    Tie-break: lexicographic by item name.
    """
    n = len(votes)
    if len(balances) != n:
        raise ValueError("votes and balances must have the same length")

    best_item: str | None = None
    best_res: PaymentResult | None = None

    for item, cost in costs.items():
        supporters = [i for i in range(n) if item in votes[i]]
        res = _equal_shares_payments(supporters, balances, cost)
        if not res.affordable:
            continue

        if best_item is None:
            best_item, best_res = item, res
            continue

        assert best_res is not None
        if (res.t, item) < (best_res.t, best_item):
            best_item, best_res = item, res

    if best_item is None or best_res is None:
        print("No item is affordable in this round.")
        return

    print(f'Round 1: "{best_item}" is elected.')

    payments_all = [0.0] * n
    for i, pay in best_res.payments.items():
        payments_all[i] = pay

    for i in range(n):
        new_balance = balances[i] - payments_all[i]
        if new_balance < 0.0 and abs(new_balance) < 1e-9:
            new_balance = 0.0
        print(
            f"Citizen {i + 1} pays {payments_all[i]:.10g} and has "
            f"{new_balance:.10g} remaining balance."
        )


def main() -> None:
    # Demo 1: the “broke supporters” example from part (a)
    votes = [
        {"Park in street X"},
        {"Park in street X"},
        {"Park in street X"},
        {"Park in street X"},
    ]
    balances = [1.0, 2.0, 10.0, 10.0]
    costs = {"Park in street X": 10.0}

    elect_next_budget_item(votes, balances, costs)

    print("\n---\n")

    # Demo 2: multiple items; algorithm picks the minimal-burden (smallest t) item
    votes2 = [
        {"A", "B"},
        {"A"},
        {"B"},
        {"B"},
    ]
    balances2 = [3.0, 3.0, 1.0, 3.0]
    costs2 = {
        "A": 4.0,  # supporters: 1,2 => likely t=2 each
        "B": 5.0,  # supporters: 1,3,4 with one weak supporter (balance=1)
    }

    elect_next_budget_item(votes2, balances2, costs2)


if __name__ == "__main__":
    main()