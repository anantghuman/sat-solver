from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .cdcl import Solver


class Brancher(ABC):
    """Chooses the next decision literal. Returns a signed literal or 0 if
    every variable is already assigned."""

    @abstractmethod
    def pick(self, solver: "Solver") -> int: ...

    def on_assign(self, var: int, value: bool) -> None:
        pass

    def on_unassign(self, var: int) -> None:
        pass

    def on_conflict(self, learned: list[int], seen_vars: list[int]) -> None:
        pass


class FirstUnassignedBrancher(Brancher):
    """Placeholder brancher: picks the lowest-indexed unassigned variable,
    always positive."""

    def pick(self, solver: "Solver") -> int:
        for v in range(1, solver.num_vars + 1):
            if solver.trail.values[v] is None:
                return v
        return 0


class VSIDSBrancher(Brancher):
    """Classic VSIDS with exponential decay + phase saving.

    Activity is bumped for every variable touched during conflict analysis
    (`seen_vars`). Rather than rescaling all activities per conflict, we
    grow a per-bump increment by 1/decay, and rescale everything when the
    increment gets large."""

    def __init__(self, num_vars: int, decay: float = 0.95, rng_seed: int = 0) -> None:
        self.num_vars = num_vars
        self.decay = decay
        self.activity = [0.0] * (num_vars + 1)
        self.bump_increment = 1.0
        self.phase: list[Optional[bool]] = [None] * (num_vars + 1)
        # Max-heap keyed by (-activity, var). Entries may be stale (activity
        # updated after push); we lazily filter unassigned + valid.
        self._heap: list[tuple[float, int]] = []
        for v in range(1, num_vars + 1):
            heapq.heappush(self._heap, (0.0, v))

    def _bump(self, var: int) -> None:
        self.activity[var] += self.bump_increment
        heapq.heappush(self._heap, (-self.activity[var], var))
        if self.activity[var] > 1e100:
            for v in range(1, self.num_vars + 1):
                self.activity[v] *= 1e-100
            self.bump_increment *= 1e-100
            # Reheapify — cheaper than filtering stale entries.
            self._heap = [(-self.activity[v], v) for v in range(1, self.num_vars + 1)]
            heapq.heapify(self._heap)

    def on_conflict(self, learned: list[int], seen_vars: list[int]) -> None:
        for v in seen_vars:
            self._bump(v)
        self.bump_increment /= self.decay

    def on_assign(self, var: int, value: bool) -> None:
        self.phase[var] = value

    def pick(self, solver: "Solver") -> int:
        values = solver.trail.values
        heap = self._heap
        while heap:
            neg_act, var = heap[0]
            # Stale entry? (activity changed since push)
            if -neg_act != self.activity[var] or values[var] is not None:
                heapq.heappop(heap)
                continue
            heapq.heappop(heap)
            saved = self.phase[var]
            if saved is None:
                saved = False  # default negative phase per MiniSAT
            return var if saved else -var
        # Fallback (should not happen unless all vars assigned).
        for v in range(1, self.num_vars + 1):
            if values[v] is None:
                return v
        return 0


def luby(unit: int = 32):
    """Generator yielding the Luby sequence scaled by `unit`, used for
    conflict-count restart thresholds."""
    def _luby_term(k: int) -> int:
        # 1,1,2,1,1,2,4,1,1,2,1,1,2,4,8,...
        power = 1
        while power - 1 < k:
            power *= 2
        if power - 1 == k:
            return power // 2
        return _luby_term(k - (power // 2) + 1)

    k = 1
    while True:
        yield unit * _luby_term(k)
        k += 1
