# square_root_decomposition_kit

A block-based array decomposition supporting point updates and range aggregate queries in O(sqrt n) without preprocessing overhead constraints.

## Usage

```python
from square_root_decomposition_kit import SqrtDecomposition, RangeSum

# Generic aggregate: sum, min, max, string concatenation, ...
d = SqrtDecomposition([3, 1, 4, 1, 5, 9, 2, 6], lambda a, b: a + b, 0)
d.range_query(1, 6)   # 1 + 4 + 1 + 5 + 9 == 20
d.set(3, 10)          # values become [3, 1, 4, 10, 5, 9, 2, 6]
d.range_query(1, 6)   # 1 + 4 + 10 + 5 + 9 == 29

# Numeric convenience wrapper
r = RangeSum([1, 2, 3, 4, 5])
r.range_sum(0, 5)     # 15
r.set(2, 10)
r.range_sum(2, 4)     # 10 + 4 == 14
```

## Why this exists

A Fenwick tree answers prefix sums in O(log n), and a segment tree answers
arbitrary associative aggregates in O(log n), but both demand a fixed combine
operation and (for Fenwick) invertibility. Square root decomposition trades a
worse asymptotic bound for flexibility: any associative operation with an
identity works, and the code is small enough to audit. It is the right choice
when the aggregate is unusual, when the array is small, or when the constant
factor of a tree is not worth the extra code.

The block size defaults to `isqrt(n)` (rounded up to at least 1). Callers who
know their workload — for example, many point updates and few queries — can
pass an explicit `block_size` to shift the balance between update and query
cost.

## Awkward edge

An empty range (`left == right`) returns the identity value, not `None` or an
error. This is the only answer that composes correctly with the aggregate, but
it means `SqrtDecomposition([], combine, identity).range_query(0, 0)` returns
the identity the caller supplied. Choose an identity that makes sense for your
operation: `0` for sums, `math.inf` for `min`, `""` for string concatenation.
