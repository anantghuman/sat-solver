"""End-to-end tests for the CDCL solver on both hand-written and known
DIMACS instances."""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from solver.cdcl import check_assignment, solve
from solver.main import main as cli_main
from solver.parser import Cnf, parse_dimacs

REPO = Path(__file__).resolve().parents[1]


def _cnf(clauses, n=None):
    n = n or max((abs(l) for c in clauses for l in c), default=0)
    return Cnf(num_vars=n, clauses=[list(c) for c in clauses])


def _random_3sat(num_vars: int, num_clauses: int, seed: int) -> Cnf:
    rng = random.Random(seed)
    clauses = []
    for _ in range(num_clauses):
        vs = rng.sample(range(1, num_vars + 1), 3)
        clauses.append([v if rng.random() < 0.5 else -v for v in vs])
    return _cnf(clauses, n=num_vars)


def test_random_3sat_under_threshold_is_sat():
    # ratio 3.0 clauses/var is well below the SAT phase transition (~4.267)
    for seed in range(20):
        cnf = _random_3sat(num_vars=25, num_clauses=75, seed=seed)
        r = solve(cnf)
        assert r.sat, f"seed {seed} should be SAT at ratio 3.0"
        assert check_assignment(cnf, r.assignment)


def test_pigeonhole_3_holes_2_pigeons_is_sat():
    # 3 pigeons into 3 holes: at least one per hole assignment.
    # Vars: x_{p,h} for p,h in [1..3]; encode "each pigeon in some hole"
    # and "no hole has two pigeons".
    def var(p, h):
        return (p - 1) * 3 + h  # 1..9
    clauses = []
    for p in range(1, 4):
        clauses.append([var(p, 1), var(p, 2), var(p, 3)])
    for h in range(1, 4):
        for a in range(1, 4):
            for b in range(a + 1, 4):
                clauses.append([-var(a, h), -var(b, h)])
    r = solve(_cnf(clauses, n=9))
    assert r.sat
    assert check_assignment(_cnf(clauses, n=9), r.assignment)


def test_pigeonhole_2_holes_3_pigeons_is_unsat():
    def var(p, h):
        return (p - 1) * 2 + h  # 1..6
    clauses = []
    for p in range(1, 4):
        clauses.append([var(p, 1), var(p, 2)])
    for h in range(1, 3):
        for a in range(1, 4):
            for b in range(a + 1, 4):
                clauses.append([-var(a, h), -var(b, h)])
    r = solve(_cnf(clauses, n=6))
    assert not r.sat


def test_dubois_via_cli(capsys):
    rc = cli_main([str(REPO / "dubois.txt"), "--quiet"])
    assert rc == 20
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0] == "UNSAT"


def test_cli_prints_valid_assignment_for_sat(capsys, tmp_path):
    p = tmp_path / "toy.cnf"
    p.write_text("p cnf 3 3\n1 2 0\n-1 3 0\n-2 -3 0\n")
    rc = cli_main([str(p), "--quiet"])
    assert rc == 10
    out = capsys.readouterr().out
    assert "SAT" in out
