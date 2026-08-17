from pathlib import Path

from solver.cdcl import check_assignment, solve
from solver.parser import Cnf, parse_dimacs

REPO = Path(__file__).resolve().parents[1]


def _cnf(clauses, n=None):
    n = n or max((abs(l) for c in clauses for l in c), default=0)
    return Cnf(num_vars=n, clauses=[list(c) for c in clauses])


def test_empty_formula_is_sat():
    r = solve(_cnf([], n=0))
    assert r.sat
    assert r.assignment == [False]


def test_single_unit_clause():
    r = solve(_cnf([[1]]))
    assert r.sat
    assert r.assignment[1] is True


def test_direct_contradiction():
    r = solve(_cnf([[1], [-1]]))
    assert not r.sat


def test_simple_sat():
    cnf = _cnf([[1, 2], [-1, 2], [1, -2]])
    r = solve(cnf)
    assert r.sat
    assert check_assignment(cnf, r.assignment)


def test_simple_unsat():
    cnf = _cnf([[1, 2], [-1, 2], [1, -2], [-1, -2]])
    r = solve(cnf)
    assert not r.sat


def test_chain_of_implications():
    # x1 → x2 → x3 → x4 with x1 asserted, and -x4 asserted → UNSAT
    cnf = _cnf([[-1, 2], [-2, 3], [-3, 4], [1], [-4]])
    r = solve(cnf)
    assert not r.sat


def test_dubois_unsat():
    r = solve(parse_dimacs(REPO / "dubois.txt"))
    assert not r.sat
