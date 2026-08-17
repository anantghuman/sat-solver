from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

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
    always positive. Correct but slow — used until VSIDS lands."""

    def pick(self, solver: "Solver") -> int:
        for v in range(1, solver.num_vars + 1):
            if solver.trail.values[v] is None:
                return v
        return 0
