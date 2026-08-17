"""Run the VSIDS solver on a set of instances, capturing (formula-state,
chosen-var, chosen-sign) at every decision point. Snapshots are small — just
the current assignment vector — so we store the initial clause set once per
instance and the trail state per decision.

If SATLIB isn't available, --synthetic generates random 3-SAT instances at
a given ratio; useful for smoke tests and for environments without the
benchmarks downloaded.
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from data.loader import iter_instances
from solver.branching import VSIDSBrancher
from solver.cdcl import Solver
from solver.parser import Cnf, parse_dimacs


@dataclass
class DecisionRecord:
    assignment: list[int]  # 0=unassigned, 1=true, -1=false (index 1..num_vars, index 0 unused)
    chosen_var: int
    chosen_sign: int  # +1 or -1


@dataclass
class InstanceTrace:
    num_vars: int
    clauses: list[list[int]]
    decisions: list[DecisionRecord]


def _snapshot(solver: Solver, chosen_lit: int) -> DecisionRecord:
    values = solver.trail.values
    encoded = [0] * (solver.num_vars + 1)
    for v in range(1, solver.num_vars + 1):
        if values[v] is True:
            encoded[v] = 1
        elif values[v] is False:
            encoded[v] = -1
    return DecisionRecord(
        assignment=encoded,
        chosen_var=abs(chosen_lit),
        chosen_sign=1 if chosen_lit > 0 else -1,
    )


def collect_from_cnf(cnf: Cnf, max_decisions: Optional[int] = None) -> InstanceTrace:
    trace = InstanceTrace(num_vars=cnf.num_vars, clauses=[list(c) for c in cnf.clauses], decisions=[])

    solver = Solver(cnf, brancher=VSIDSBrancher(cnf.num_vars))

    def hook(sv: Solver, chosen: int) -> None:
        if max_decisions is not None and len(trace.decisions) >= max_decisions:
            return
        trace.decisions.append(_snapshot(sv, chosen))

    solver.solve(trace_hook=hook)
    return trace


def _synthetic_instances(n: int, num_vars: int, ratio: float, seed: int) -> Iterable[Cnf]:
    rng = random.Random(seed)
    m = int(round(ratio * num_vars))
    for _ in range(n):
        clauses = []
        while len(clauses) < m:
            vs = rng.sample(range(1, num_vars + 1), 3)
            clauses.append([v if rng.random() < 0.5 else -v for v in vs])
        yield Cnf(num_vars=num_vars, clauses=clauses)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=None, help="SATLIB bucket name (e.g. uf20-91)")
    ap.add_argument("--synthetic", type=int, default=None, help="Generate N synthetic 3-SAT instances instead")
    ap.add_argument("--num-vars", type=int, default=20, help="Synthetic: vars per instance")
    ap.add_argument("--ratio", type=float, default=4.0, help="Synthetic: clauses/vars ratio")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-instances", type=int, default=200)
    ap.add_argument("--limit-decisions-per-instance", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True, help="Output .pkl path")
    args = ap.parse_args(argv)

    if bool(args.bucket) == bool(args.synthetic):
        ap.error("pick exactly one of --bucket or --synthetic")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    traces: list[InstanceTrace] = []
    total_decisions = 0

    if args.bucket:
        instances = iter_instances(args.bucket)
        for i, (path, _label) in enumerate(instances):
            if i >= args.limit_instances:
                break
            cnf = parse_dimacs(path)
            t = collect_from_cnf(cnf, max_decisions=args.limit_decisions_per_instance)
            traces.append(t)
            total_decisions += len(t.decisions)
    else:
        for i, cnf in enumerate(_synthetic_instances(args.synthetic, args.num_vars, args.ratio, args.seed)):
            if i >= args.limit_instances:
                break
            t = collect_from_cnf(cnf, max_decisions=args.limit_decisions_per_instance)
            traces.append(t)
            total_decisions += len(t.decisions)

    with open(args.out, "wb") as fh:
        pickle.dump(traces, fh)
    print(f"wrote {len(traces)} instances, {total_decisions} decisions -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
