from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from .branching import Brancher, VSIDSBrancher
from .cdcl import Solver, check_assignment
from .parser import parse_dimacs


def _build_brancher(name: str, num_vars: int, model_path: Optional[str]) -> Brancher:
    if name == "vsids":
        return VSIDSBrancher(num_vars)
    if name == "gnn":
        if model_path is None:
            raise SystemExit("--brancher gnn requires --model PATH")
        from .gnn_brancher import GNNBrancher
        return GNNBrancher(num_vars, model_path)
    raise SystemExit(f"unknown brancher {name!r}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="solver", description="CDCL SAT solver")
    ap.add_argument("cnf_path", help="Path to DIMACS .cnf file")
    ap.add_argument("--brancher", default="vsids", choices=["vsids", "gnn"])
    ap.add_argument("--model", default=None, help="Path to GNN model (--brancher gnn)")
    ap.add_argument("--restart-unit", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=None, help="Wall-clock timeout in seconds")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cnf = parse_dimacs(args.cnf_path)
    brancher = _build_brancher(args.brancher, cnf.num_vars, args.model)
    solver = Solver(cnf, brancher=brancher, restart_unit=args.restart_unit)

    t0 = time.perf_counter()
    result = solver.solve()
    elapsed = time.perf_counter() - t0

    if result.sat:
        if not args.quiet:
            print("s SATISFIABLE")
            # DIMACS v-line: signed literals, terminated with 0
            lits = [str(v if result.assignment[v] else -v) for v in range(1, cnf.num_vars + 1)]
            for i in range(0, len(lits), 20):
                print("v " + " ".join(lits[i : i + 20]))
            print("v 0")
        else:
            print("SAT")
        assert check_assignment(cnf, result.assignment), "solver returned invalid assignment"
    else:
        print("s UNSATISFIABLE" if not args.quiet else "UNSAT")

    if not args.quiet:
        s = result.stats
        print(
            f"c decisions={s.decisions} conflicts={s.conflicts} "
            f"propagations={s.propagations} restarts={s.restarts} "
            f"learned={s.learned} time={elapsed:.3f}s"
        )
    return 10 if result.sat else 20  # DIMACS-style exit codes


if __name__ == "__main__":
    sys.exit(main())
