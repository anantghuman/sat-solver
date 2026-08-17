"""End-to-end test: train a tiny model on synthetic traces, then use it as
the brancher and confirm the solver still returns correct results."""
import random
from pathlib import Path

import pytest

pytest.importorskip("torch")

from solver.cdcl import Solver, check_assignment
from solver.gnn_brancher import GNNBrancher
from solver.parser import Cnf


def _random_3sat(num_vars, num_clauses, seed):
    rng = random.Random(seed)
    clauses = []
    for _ in range(num_clauses):
        vs = rng.sample(range(1, num_vars + 1), 3)
        clauses.append([v if rng.random() < 0.5 else -v for v in vs])
    return Cnf(num_vars=num_vars, clauses=clauses)


def _train_tiny(tmp_path: Path):
    import pickle

    from training.collect_traces import collect_from_cnf
    from training.train import main as train_main

    traces = [collect_from_cnf(_random_3sat(10, 30, seed=s)) for s in range(20)]
    tp = tmp_path / "traces.pkl"
    with open(tp, "wb") as fh:
        pickle.dump(traces, fh)
    mp = tmp_path / "model.pt"
    rc = train_main([
        "--traces", str(tp), "--out", str(mp),
        "--epochs", "2", "--hidden", "8", "--rounds", "2",
        "--edge-budget", "2000", "--holdout-frac", "0.1",
    ])
    assert rc == 0
    return mp


def test_gnn_brancher_solves(tmp_path):
    model_path = _train_tiny(tmp_path)
    cnf = _random_3sat(10, 30, seed=999)
    solver = Solver(cnf, brancher=GNNBrancher(cnf.num_vars, model_path))
    result = solver.solve()
    if result.sat:
        assert check_assignment(cnf, result.assignment)
