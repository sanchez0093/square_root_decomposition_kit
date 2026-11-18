import math
import unittest

from square_root_decomposition_kit import RangeSum, SqrtDecomposition


class TestSqrtDecompositionSum(unittest.TestCase):
    def test_empty(self):
        d = SqrtDecomposition([], lambda a, b: a + b, 0)
        self.assertEqual(len(d), 0)
        self.assertEqual(d.range_query(0, 0), 0)

    def test_single_element(self):
        d = SqrtDecomposition([7], lambda a, b: a + b, 0)
        self.assertEqual(d.get(0), 7)
        self.assertEqual(d.range_query(0, 1), 7)
        self.assertEqual(d.range_query(0, 0), 0)

    def test_full_range(self):
        values = list(range(10))
        d = SqrtDecomposition(values, lambda a, b: a + b, 0)
        self.assertEqual(d.range_query(0, 10), sum(values))

    def test_partial_ranges(self):
        values = [3, 1, 4, 1, 5, 9, 2, 6]
        d = SqrtDecomposition(values, lambda a, b: a + b, 0)
        for left in range(len(values)):
            for right in range(left, len(values) + 1):
                self.assertEqual(
                    d.range_query(left, right),
                    sum(values[left:right]),
                    msg=f"range [{left}, {right})",
                )

    def test_point_update(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        d = SqrtDecomposition(values, lambda a, b: a + b, 0)
        d.set(4, 100)
        self.assertEqual(d.get(4), 100)
        self.assertEqual(d.range_query(0, 9), sum(values) - 5 + 100)
        self.assertEqual(d.range_query(2, 6), 3 + 4 + 100 + 6)

    def test_update_then_query_consistency(self):
        values = [1] * 17
        d = SqrtDecomposition(values, lambda a, b: a + b, 0)
        d.set(0, 10)
        d.set(16, 20)
        self.assertEqual(d.range_query(0, 17), 10 + 15 + 20)

    def test_block_size_override(self):
        values = list(range(10))
        d = SqrtDecomposition(values, lambda a, b: a + b, 0, block_size=3)
        self.assertEqual(d.block_size, 3)
        self.assertEqual(d.range_query(1, 9), sum(values[1:9]))

    def test_bad_block_size(self):
        with self.assertRaises(ValueError):
            SqrtDecomposition([1, 2, 3], lambda a, b: a + b, 0, block_size=0)

    def test_index_errors(self):
        d = SqrtDecomposition([1, 2, 3], lambda a, b: a + b, 0)
        with self.assertRaises(IndexError):
            d.get(3)
        with self.assertRaises(IndexError):
            d.get(-1)
        with self.assertRaises(IndexError):
            d.set(3, 0)
        with self.assertRaises(IndexError):
            d.range_query(-1, 1)
        with self.assertRaises(IndexError):
            d.range_query(0, 4)
        with self.assertRaises(IndexError):
            d.range_query(2, 1)

    def test_non_commutative_combine(self):
        # String concatenation is associative but not commutative; the block
        # loop must preserve left-to-right order across block boundaries.
        values = ["a", "b", "c", "d", "e", "f", "g"]
        d = SqrtDecomposition(values, lambda a, b: a + b, "")
        self.assertEqual(d.range_query(0, 7), "abcdefg")
        self.assertEqual(d.range_query(1, 6), "bcdef")
        d.set(3, "X")
        self.assertEqual(d.range_query(0, 7), "abcXefg")
        self.assertEqual(d.range_query(2, 5), "cXe")

    def test_min_aggregate(self):
        values = [5, 3, 8, 1, 9, 2, 7, 4, 6]
        d = SqrtDecomposition(values, min, math.inf)
        for left in range(len(values)):
            for right in range(left + 1, len(values) + 1):
                self.assertEqual(d.range_query(left, right), min(values[left:right]))

    def test_empty_range_returns_identity(self):
        d = SqrtDecomposition([1, 2, 3], lambda a, b: a + b, 0)
        self.assertEqual(d.range_query(1, 1), 0)


class TestRangeSum(unittest.TestCase):
    def test_basic(self):
        r = RangeSum([1, 2, 3, 4, 5])
        self.assertEqual(r.range_sum(0, 5), 15)
        self.assertEqual(r.range_sum(1, 4), 9)
        self.assertEqual(r.range_sum(2, 2), 0)

    def test_update(self):
        r = RangeSum([1, 2, 3])
        r.set(1, 10)
        self.assertEqual(r.get(1), 10)
        self.assertEqual(r.range_sum(0, 3), 14)

    def test_matches_builtin_sum(self):
        values = [(-1) ** i * i for i in range(20)]
        r = RangeSum(values)
        for left in range(len(values)):
            for right in range(left, len(values) + 1):
                self.assertEqual(r.range_sum(left, right), sum(values[left:right]))

    def test_float_values(self):
        values = [0.5, 1.25, -2.0, 3.75]
        r = RangeSum(values)
        self.assertAlmostEqual(r.range_sum(0, 4), 3.5)
        r.set(2, 1.0)
        self.assertAlmostEqual(r.range_sum(0, 4), 6.5)


if __name__ == "__main__":
    unittest.main()
