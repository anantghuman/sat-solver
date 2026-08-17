from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .branching import Brancher, FirstUnassignedBrancher
from .clause import Clause, ClauseStore
from .parser import Cnf
from .trail import Trail


@dataclass
class SolveStats:
    decisions: int = 0
    conflicts: int = 0
    propagations: int = 0
    restarts: int = 0
    learned: int = 0


@dataclass
class SolveResult:
    sat: bool
    assignment: Optional[list[bool]]  # index-1; assignment[0] unused
    stats: SolveStats


class Solver:
    def __init__(self, cnf: Cnf, brancher: Optional[Brancher] = None) -> None:
        self.num_vars = cnf.num_vars
        self.store = ClauseStore(cnf.num_vars)
        self.trail = Trail(cnf.num_vars)
        self.brancher: Brancher = brancher or FirstUnassignedBrancher()
        self.stats = SolveStats()
        self._propagation_queue_head = 0
        self._unsat = False
        for lits in cnf.clauses:
            if self._add_input_clause(lits) is False:
                self._unsat = True
                return

    def _add_input_clause(self, lits: list[int]) -> bool:
        # Simplify against level-0 assignments while ingesting original clauses.
        simp: list[int] = []
        for lit in lits:
            v = self.trail.value_of(lit)
            if v is True:
                return True  # already satisfied
            if v is False:
                continue
            simp.append(lit)
        if not simp:
            return False  # empty clause → UNSAT
        if len(simp) == 1:
            lit = simp[0]
            existing = self.trail.value_of(lit)
            if existing is False:
                return False
            if existing is None:
                self.trail.enqueue(lit, None)
            return True
        self.store.add_original(simp)
        return True

    # ---- Propagation --------------------------------------------------------

    def propagate(self) -> Optional[Clause]:
        """BCP over watched literals. Returns the conflict clause, or None."""
        trail = self.trail
        watches = self.store.watches
        while self._propagation_queue_head < len(trail.entries):
            entry = trail.entries[self._propagation_queue_head]
            self._propagation_queue_head += 1
            self.stats.propagations += 1
            assigned_lit = entry.var if entry.value else -entry.var
            falsified = -assigned_lit
            watchers = watches.get(falsified)
            if not watchers:
                continue
            keep: list[Clause] = []
            i = 0
            conflict: Optional[Clause] = None
            while i < len(watchers):
                clause = watchers[i]
                i += 1
                if clause.w0 == clause.w1:
                    # Unit-length clause: the sole literal just became false.
                    keep.append(clause)
                    conflict = clause
                    break
                other = clause.other_watch(falsified)
                if trail.value_of(other) is True:
                    keep.append(clause)
                    continue
                replacement_idx = -1
                for idx, lit in enumerate(clause.lits):
                    if idx == clause.w0 or idx == clause.w1:
                        continue
                    if trail.value_of(lit) is not False:
                        replacement_idx = idx
                        break
                if replacement_idx >= 0:
                    if clause.lits[clause.w0] == falsified:
                        clause.w0 = replacement_idx
                    else:
                        clause.w1 = replacement_idx
                    watches.setdefault(clause.lits[replacement_idx], []).append(clause)
                    continue
                keep.append(clause)
                if trail.value_of(other) is False:
                    conflict = clause
                    break
                trail.enqueue(other, clause)
                self.brancher.on_assign(abs(other), other > 0)
            while i < len(watchers):
                keep.append(watchers[i])
                i += 1
            watches[falsified] = keep
            if conflict is not None:
                return conflict
        return None

    # ---- Main loop ----------------------------------------------------------

    def solve(self, trace_hook: Optional[Callable[["Solver", int], None]] = None) -> SolveResult:
        if self._unsat:
            return SolveResult(sat=False, assignment=None, stats=self.stats)

        # Chronological DPLL: track per-decision-level "has this decision
        # already tried its opposite phase?" 1-UIP conflict-driven learning
        # replaces this in the next commit.
        flipped: list[bool] = [False]  # index by decision level; [0] unused.

        while True:
            conflict = self.propagate()
            if conflict is not None:
                self.stats.conflicts += 1
                # Walk up the trail to find a decision whose second phase is
                # untried; if none, formula is UNSAT.
                found = False
                while self.trail.decision_level > 0:
                    level = self.trail.decision_level
                    was_flipped = flipped[level]
                    dec = self.trail.entries[self.trail.level_starts[level]]
                    popped = self.trail.backjump(level - 1)
                    for v in popped:
                        self.brancher.on_unassign(v)
                    self._propagation_queue_head = len(self.trail.entries)
                    flipped.pop()
                    if not was_flipped:
                        flipped_lit = -dec.var if dec.value else dec.var
                        self.trail.new_decision_level()
                        flipped.append(True)
                        self.trail.enqueue(flipped_lit, None)
                        self.brancher.on_assign(abs(flipped_lit), flipped_lit > 0)
                        found = True
                        break
                if not found:
                    return SolveResult(sat=False, assignment=None, stats=self.stats)
                continue

            if len(self.trail.entries) == self.num_vars:
                return self._sat_result()

            lit = self.brancher.pick(self)
            if lit == 0:
                return self._sat_result()
            self.stats.decisions += 1
            if trace_hook is not None:
                trace_hook(self, lit)
            self.trail.new_decision_level()
            flipped.append(False)
            self.trail.enqueue(lit, None)
            self.brancher.on_assign(abs(lit), lit > 0)

    def _sat_result(self) -> SolveResult:
        assignment = [False] * (self.num_vars + 1)
        for v in range(1, self.num_vars + 1):
            assignment[v] = self.trail.values[v] is True
        return SolveResult(sat=True, assignment=assignment, stats=self.stats)


def solve(cnf: Cnf, brancher: Optional[Brancher] = None) -> SolveResult:
    return Solver(cnf, brancher=brancher).solve()


def check_assignment(cnf: Cnf, assignment: list[bool]) -> bool:
    for clause in cnf.clauses:
        if not any((lit > 0 and assignment[abs(lit)]) or (lit < 0 and not assignment[abs(lit)]) for lit in clause):
            return False
    return True
