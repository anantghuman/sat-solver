from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .clause import Clause


@dataclass
class TrailEntry:
    var: int
    value: bool
    level: int
    antecedent: Optional[Clause]  # None for decisions


class Trail:
    """Assignment trail with O(1) lookup and O(popped) backjump."""

    def __init__(self, num_vars: int) -> None:
        self.num_vars = num_vars
        self.entries: list[TrailEntry] = []
        self.level_starts: list[int] = [0]  # level_starts[k] = trail index of first entry at level k
        self.values: list[Optional[bool]] = [None] * (num_vars + 1)
        self.levels: list[int] = [-1] * (num_vars + 1)
        self.antecedents: list[Optional[Clause]] = [None] * (num_vars + 1)

    @property
    def decision_level(self) -> int:
        return len(self.level_starts) - 1

    def value_of(self, lit: int) -> Optional[bool]:
        v = self.values[abs(lit)]
        if v is None:
            return None
        return v if lit > 0 else not v

    def is_true(self, lit: int) -> bool:
        return self.value_of(lit) is True

    def is_false(self, lit: int) -> bool:
        return self.value_of(lit) is False

    def is_unassigned(self, lit: int) -> bool:
        return self.values[abs(lit)] is None

    def new_decision_level(self) -> int:
        self.level_starts.append(len(self.entries))
        return self.decision_level

    def enqueue(self, lit: int, antecedent: Optional[Clause]) -> None:
        var = abs(lit)
        assert self.values[var] is None, f"var {var} already assigned"
        value = lit > 0
        level = self.decision_level
        self.values[var] = value
        self.levels[var] = level
        self.antecedents[var] = antecedent
        self.entries.append(TrailEntry(var=var, value=value, level=level, antecedent=antecedent))

    def backjump(self, level: int) -> list[int]:
        """Undo all assignments made above `level`. Returns the popped vars."""
        assert 0 <= level <= self.decision_level
        target_start = self.level_starts[level + 1] if level + 1 < len(self.level_starts) else len(self.entries)
        popped: list[int] = []
        while len(self.entries) > target_start:
            e = self.entries.pop()
            self.values[e.var] = None
            self.levels[e.var] = -1
            self.antecedents[e.var] = None
            popped.append(e.var)
        del self.level_starts[level + 1 :]
        return popped
