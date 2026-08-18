"""Compare VSIDS vs GNN branchers on a bucket of instances.

For each instance, run both branchers under the same wall-clock timeout,
record decisions/conflicts/propagations/restarts/time, dump a CSV,
and print a summary. Any SAT-vs-UNSAT disagreement is flagged as a bug.
"""
from __future__ import annotations

import argparse
import csv
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from data.loader import iter_instances
from solver.branching import VSIDSBrancher
from solver.cdcl import Solver, check_assignment
from solver.parser import parse_dimacs


@dataclass
class RunResult:
    result: str  # "SAT" | "UNSAT" | "TIMEOUT" | "ERROR"
    decisions: int = 0
    conflicts: int = 0
    propagations: int = 0
    restarts: int = 0
    learned: int = 0
    wall_time: float = 0.0


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout()


def _run(cnf, brancher, timeout: float) -> RunResult:
    solver = Solver(cnf, brancher=brancher)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    t0 = time.perf_counter()
    try:
        r = solver.solve()
    except _Timeout:
        elapsed = time.perf_counter() - t0
        return RunResult(result="TIMEOUT", wall_time=elapsed, **_stats(solver))
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"[error] {e}", file=sys.stderr)
        return RunResult(result="ERROR", wall_time=elapsed, **_stats(solver))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    elapsed = time.perf_counter() - t0
    if r.sat:
        assert check_assignment(cnf, r.assignment), "invalid assignment returned"
    return RunResult(
        result="SAT" if r.sat else "UNSAT",
        decisions=r.stats.decisions,
        conflicts=r.stats.conflicts,
        propagations=r.stats.propagations,
        restarts=r.stats.restarts,
        learned=r.stats.learned,
        wall_time=elapsed,
    )


def _stats(solver: Solver) -> dict:
    s = solver.stats
    return dict(
        decisions=s.decisions,
        conflicts=s.conflicts,
        propagations=s.propagations,
        restarts=s.restarts,
        learned=s.learned,
    )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("eval/results.csv"))
    args = ap.parse_args(argv)

    from solver.gnn_brancher import GNNBrancher  # torch dep is lazy

    rows = []
    disagreements = 0
    for i, (path, label) in enumerate(iter_instances(args.bucket)):
        if i >= args.limit:
            break
        cnf = parse_dimacs(path)
        for name, factory in [
            ("vsids", lambda: VSIDSBrancher(cnf.num_vars)),
            ("gnn",   lambda: GNNBrancher(cnf.num_vars, args.model)),
        ]:
            r = _run(cnf, factory(), args.timeout)
            rows.append({
                "instance": path.name,
                "brancher": name,
                "result": r.result,
                "decisions": r.decisions,
                "conflicts": r.conflicts,
                "propagations": r.propagations,
                "restarts": r.restarts,
                "learned": r.learned,
                "wall_time": f"{r.wall_time:.3f}",
                "expected": {True: "SAT", False: "UNSAT"}.get(label, ""),
            })
        # disagreement check
        v = next(x for x in rows if x["instance"] == path.name and x["brancher"] == "vsids")
        g = next(x for x in rows if x["instance"] == path.name and x["brancher"] == "gnn")
        if v["result"] in ("SAT", "UNSAT") and g["result"] in ("SAT", "UNSAT") and v["result"] != g["result"]:
            print(f"[BUG] disagreement on {path.name}: vsids={v['result']} gnn={g['result']}", file=sys.stderr)
            disagreements += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summary on jointly-solved
    by_instance: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_instance.setdefault(r["instance"], {})[r["brancher"]] = r
    joint = [(v["vsids"], v["gnn"]) for v in by_instance.values()
             if v.get("vsids", {}).get("result") in ("SAT", "UNSAT")
             and v.get("gnn", {}).get("result") in ("SAT", "UNSAT")]
    if joint:
        def _gmean(xs): return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")
        ratios_dec = [max(1, int(g["decisions"])) / max(1, int(v["decisions"])) for v, g in joint]
        print(f"jointly solved: {len(joint)} of {len(by_instance)}")
        print(f"gmean(gnn_decisions / vsids_decisions) = {_gmean(ratios_dec):.3f}")
    vs_solved = sum(1 for v in by_instance.values() if v.get("vsids", {}).get("result") in ("SAT", "UNSAT"))
    gn_solved = sum(1 for v in by_instance.values() if v.get("gnn", {}).get("result") in ("SAT", "UNSAT"))
    print(f"solved: vsids={vs_solved} gnn={gn_solved} of {len(by_instance)}")
    print(f"disagreements: {disagreements}")
    print(f"csv -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
