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

    # ---- Conflict analysis (1-UIP) ------------------------------------------

    def analyze(self, conflict: Clause) -> tuple[list[int], int, list[int]]:
        """1-UIP conflict analysis.

        Returns (learned_clause, backjump_level, seen_vars_for_bumping).
        The learned clause is arranged so that:
          - lits[0] is the asserting literal (unique current-level literal)
          - lits[1] (if present) is a literal at backjump_level
        which is what the watched-literal invariant needs after backjumping.
        """
        trail = self.trail
        current_level = trail.decision_level
        seen = [False] * (self.num_vars + 1)
        learned: list[int] = [0]  # slot 0 is the asserting lit, filled at end
        seen_vars: list[int] = []
        counter = 0  # unresolved literals still at current_level
        p_lit = 0
        reason: Optional[Clause] = conflict
        idx = len(trail.entries) - 1

        while True:
            assert reason is not None
            # Resolve on reason. Skip `p_lit`'s variable on the first non-conflict
            # iteration (that variable was resolved away).
            for lit in reason.lits:
                v = abs(lit)
                if seen[v] or v == abs(p_lit):
                    continue
                lvl = trail.levels[v]
                if lvl <= 0:
                    continue
                seen[v] = True
                seen_vars.append(v)
                if lvl >= current_level:
                    counter += 1
                else:
                    learned.append(lit)
            # Walk back to the next seen variable on the current level.
            while idx >= 0 and not seen[trail.entries[idx].var]:
                idx -= 1
            if idx < 0:
                # Should not happen for a well-formed conflict.
                raise RuntimeError("conflict analysis walked off the trail")
            e = trail.entries[idx]
            p_lit = e.var if e.value else -e.var
            reason = trail.antecedents[e.var]
            idx -= 1
            counter -= 1
            if counter == 0:
                # The asserting UIP is `-p_lit`.
                learned[0] = -p_lit
                break

        if len(learned) == 1:
            backjump_level = 0
        else:
            # Move the highest-level non-asserting literal to position 1.
            max_pos = 1
            max_level = trail.levels[abs(learned[1])]
            for k in range(2, len(learned)):
                lvl = trail.levels[abs(learned[k])]
                if lvl > max_level:
                    max_level = lvl
                    max_pos = k
            if max_pos != 1:
                learned[1], learned[max_pos] = learned[max_pos], learned[1]
            backjump_level = max_level

        return learned, backjump_level, seen_vars

    # ---- Main loop ----------------------------------------------------------

    def solve(self, trace_hook: Optional[Callable[["Solver", int], None]] = None) -> SolveResult:
        if self._unsat:
            return SolveResult(sat=False, assignment=None, stats=self.stats)

        while True:
            conflict = self.propagate()
            if conflict is not None:
                self.stats.conflicts += 1
                if self.trail.decision_level == 0:
                    return SolveResult(sat=False, assignment=None, stats=self.stats)
                learned, backjump_level, seen_vars = self.analyze(conflict)
                self.brancher.on_conflict(learned, seen_vars)
                popped = self.trail.backjump(backjump_level)
                for v in popped:
                    self.brancher.on_unassign(v)
                self._propagation_queue_head = len(self.trail.entries)
                # Install learned clause with watches on the asserting lit (0)
                # and (if any) the highest-level other lit (1).
                if len(learned) == 1:
                    self.trail.enqueue(learned[0], None)
                    self.brancher.on_assign(abs(learned[0]), learned[0] > 0)
                else:
                    c = self.store.add_learned_watched(learned, 0, 1)
                    self.stats.learned += 1
                    self.trail.enqueue(learned[0], c)
                    self.brancher.on_assign(abs(learned[0]), learned[0] > 0)
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
