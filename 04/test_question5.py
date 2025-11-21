import sys
import unittest
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from question5 import envy_free_room_allocation


class TestQuestion5(unittest.TestCase):
    def _assert_envy_free(self, valuations, rent, assignment, prices, tol=1e-6):
        n = len(valuations)
        self.assertEqual(len(assignment), n)
        self.assertEqual(len(prices), n)
        self.assertTrue(np.allclose(sorted(assignment), np.arange(n)))
        self.assertAlmostEqual(sum(prices), rent, places=6)

        valuations = np.asarray(valuations, dtype=float)
        for i in range(n):
            my_room = assignment[i]
            my_utility = valuations[i, my_room] - prices[my_room]
            for j in range(n):
                alt_room = assignment[j]
                alt_utility = valuations[i, alt_room] - prices[alt_room]
                self.assertGreaterEqual(my_utility + tol, alt_utility, f"Player {i} envies player {j}")

    def test_homework_example(self):
        valuations = [[150, 0], [140, 10]]
        rent = 100
        assignment, prices = envy_free_room_allocation(valuations, rent, verbose=False)
        self._assert_envy_free(valuations, rent, assignment, prices)

    def test_three_players(self):
        valuations = [
            [70, 35, 45],
            [20, 45, 45],
            [10, 20, 10],
        ]
        rent = 100
        assignment, prices = envy_free_room_allocation(valuations, rent, verbose=False)
        self._assert_envy_free(valuations, rent, assignment, prices)

    def test_random_instances(self):
        rng = np.random.default_rng(42)
        for n in range(2, 5):
            for _ in range(5):
                valuations = rng.integers(0, 200, size=(n, n)).astype(float).tolist()
                rent = float(rng.integers(50, 400))
                assignment, prices = envy_free_room_allocation(valuations, rent, verbose=False)
                self._assert_envy_free(valuations, rent, assignment, prices)

    def test_single_player(self):
        valuations = [[125.0]]
        rent = 125.0
        assignment, prices = envy_free_room_allocation(valuations, rent, verbose=False)
        self._assert_envy_free(valuations, rent, assignment, prices)

    def test_non_square_matrix_raises(self):
        valuations = [[10, 5], [7, 3], [2, 1]]
        with self.assertRaises(ValueError):
            envy_free_room_allocation(valuations, rent=50.0, verbose=False)

    def test_empty_matrix_raises(self):
        with self.assertRaises(ValueError):
            envy_free_room_allocation([], rent=100.0, verbose=False)


if __name__ == "__main__":
    unittest.main()

