"""Square root decomposition for point updates and range aggregate queries.

The structure splits an array into blocks of roughly sqrt(n) elements. Each block
stores a precomputed aggregate. A point update touches one element and one block
aggregate. A range query combines partial blocks element-by-element with whole
blocks in O(1), giving O(sqrt(n)) worst-case for both operations.

We deliberately avoid a Fenwick or segment tree here: those structures are faster
for the common operations but require the aggregate to be invertible (for prefix
sums) or to support a fixed associative combine. Square root decomposition works
for any associative aggregate, which is the trade-off this library makes.
"""

from __future__ import annotations

import math
from typing import Callable, Generic, List, Optional, Sequence, TypeVar

T = TypeVar("T")


class SqrtDecomposition(Generic[T]):
    """Block decomposition of a sequence supporting point updates and range queries.

    The aggregate function must be associative and have an identity element.
    It is applied to the elements of a range, in left-to-right order, but
    associativity means the grouping used by the implementation does not matter.

    Args:
        values: Initial sequence of elements. A copy is stored.
        combine: Associative binary operation ``combine(a, b)``.
        identity: Identity element for ``combine``: ``combine(identity, x) == x``
            and ``combine(x, identity) == x`` for every stored ``x``.
        block_size: Optional explicit block size. If omitted, ``ceil(sqrt(n))``
            is used. Supplying a size is useful when the caller knows the array
            shape (for example, many point updates and few queries).

    Raises:
        ValueError: If ``block_size`` is provided and not positive.
    """

    __slots__ = ("_values", "_combine", "_identity", "_block_size", "_blocks")

    def __init__(
        self,
        values: Sequence[T],
        combine: Callable[[T, T], T],
        identity: T,
        block_size: Optional[int] = None,
    ) -> None:
        self._values: List[T] = list(values)
        self._combine = combine
        self._identity = identity
        n = len(self._values)
        if block_size is None:
            block_size = max(1, math.isqrt(n)) if n else 1
        elif block_size <= 0:
            raise ValueError("block_size must be positive")
        self._block_size = block_size
        self._blocks: List[T] = []
        self._rebuild()

    def _rebuild(self) -> None:
        """Recompute every block aggregate from the current values."""
        self._blocks = []
        for start in range(0, len(self._values), self._block_size):
            self._blocks.append(self._aggregate(start, min(start + self._block_size, len(self._values))))

    def _aggregate(self, start: int, end: int) -> T:
        """Aggregate ``values[start:end]`` using ``combine`` and ``identity``."""
        acc = self._identity
        for i in range(start, end):
            acc = self._combine(acc, self._values[i])
        return acc

    def _block_index(self, index: int) -> int:
        return index // self._block_size

    def __len__(self) -> int:
        return len(self._values)

    @property
    def block_size(self) -> int:
        """Number of elements per block (the last block may be shorter)."""
        return self._block_size

    def get(self, index: int) -> T:
        """Return the element at ``index``.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if not 0 <= index < len(self._values):
            raise IndexError("index out of range")
        return self._values[index]

    def set(self, index: int, value: T) -> None:
        """Replace the element at ``index`` with ``value``.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if not 0 <= index < len(self._values):
            raise IndexError("index out of range")
        self._values[index] = value
        b = self._block_index(index)
        start = b * self._block_size
        end = min(start + self._block_size, len(self._values))
        self._blocks[b] = self._aggregate(start, end)

    def range_query(self, left: int, right: int) -> T:
        """Return the aggregate of ``values[left:right]`` (half-open).

        An empty range returns ``identity``. This is the only sensible answer
        given that the caller supplied the identity; returning ``None`` would
        force every caller to special-case emptiness.

        Raises:
            IndexError: If ``left`` or ``right`` is outside ``[0, len]`` or
                ``left > right``.
        """
        n = len(self._values)
        if left < 0 or right > n or left > right:
            raise IndexError("range out of bounds")
        if left == right:
            return self._identity

        left_block = self._block_index(left)
        right_block = self._block_index(right - 1)

        if left_block == right_block:
            return self._aggregate(left, right)

        acc = self._identity
        # Left partial block.
        left_end = (left_block + 1) * self._block_size
        acc = self._combine(acc, self._aggregate(left, left_end))
        # Whole blocks in between.
        for b in range(left_block + 1, right_block):
            acc = self._combine(acc, self._blocks[b])
        # Right partial block.
        right_start = right_block * self._block_size
        acc = self._combine(acc, self._aggregate(right_start, right))
        return acc


class RangeSum(Generic[T]):
    """Convenience wrapper around :class:`SqrtDecomposition` for numeric sums.

    Numbers default to ``0`` as the identity and ``+`` as the combine. This is
    the most common use of the structure and saves callers from passing the
    same two arguments every time.
    """

    __slots__ = ("_impl",)

    def __init__(self, values: Sequence[T], block_size: Optional[int] = None) -> None:
        self._impl = SqrtDecomposition(values, lambda a, b: a + b, 0, block_size)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self._impl)

    @property
    def block_size(self) -> int:
        return self._impl.block_size

    def get(self, index: int) -> T:
        return self._impl.get(index)

    def set(self, index: int, value: T) -> None:
        self._impl.set(index, value)

    def range_sum(self, left: int, right: int) -> T:
        return self._impl.range_query(left, right)
