from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Clause:
    lits: list[int]
    w0: int = 0
    w1: int = 1
    is_learned: bool = False
    activity: float = 0.0

    def watched_lits(self) -> tuple[int, int]:
        return self.lits[self.w0], self.lits[self.w1]

    def other_watch(self, lit: int) -> int:
        a, b = self.lits[self.w0], self.lits[self.w1]
        if a == lit:
            return b
        assert b == lit, f"lit {lit} not watched in {self.lits}"
        return a


class ClauseStore:
    """Owns clauses and their watch lists (keyed by literal)."""

    def __init__(self, num_vars: int) -> None:
        self.num_vars = num_vars
        self.clauses: list[Clause] = []
        # watches[lit] = list of clauses watching `lit`. Keys are signed ints.
        self.watches: dict[int, list[Clause]] = {}

    def add_original(self, lits: list[int]) -> Clause | None:
        return self._add(lits, is_learned=False)

    def add_learned(self, lits: list[int]) -> Clause | None:
        return self._add(lits, is_learned=True)

    def _add(self, lits: list[int], *, is_learned: bool) -> Clause | None:
        if not lits:
            return None
        if len(lits) == 1:
            c = Clause(lits=list(lits), w0=0, w1=0, is_learned=is_learned)
            self.clauses.append(c)
            self.watches.setdefault(lits[0], []).append(c)
            return c
        c = Clause(lits=list(lits), w0=0, w1=1, is_learned=is_learned)
        self.clauses.append(c)
        self.watches.setdefault(lits[0], []).append(c)
        self.watches.setdefault(lits[1], []).append(c)
        return c

    def add_learned_watched(self, lits: list[int], w0: int, w1: int) -> Clause:
        """Add a learned clause with caller-chosen watches (the two most-
        recently-falsified literals, per 1-UIP convention)."""
        assert lits and w0 != w1 and 0 <= w0 < len(lits) and 0 <= w1 < len(lits)
        c = Clause(lits=list(lits), w0=w0, w1=w1, is_learned=True)
        self.clauses.append(c)
        self.watches.setdefault(lits[w0], []).append(c)
        self.watches.setdefault(lits[w1], []).append(c)
        return c

    def watches_for(self, lit: int) -> list[Clause]:
        return self.watches.setdefault(lit, [])
