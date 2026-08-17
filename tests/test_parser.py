from pathlib import Path

import pytest

from solver.parser import DimacsError, parse_dimacs, parse_dimacs_text

REPO = Path(__file__).resolve().parents[1]


def test_basic():
    cnf = parse_dimacs_text("p cnf 3 2\n1 -2 3 0\n-1 2 0\n")
    assert cnf.num_vars == 3
    assert cnf.clauses == [[1, -2, 3], [-1, 2]]


def test_comments_and_blank_lines():
    src = "c hello\n\nc world\np cnf 2 1\n1 2 0\n"
    cnf = parse_dimacs_text(src)
    assert cnf.clauses == [[1, 2]]


def test_multi_line_clause():
    src = "p cnf 4 1\n1 2\n3 4\n0\n"
    cnf = parse_dimacs_text(src)
    assert cnf.clauses == [[1, 2, 3, 4]]


def test_percent_terminator():
    src = "p cnf 2 1\n1 -2 0\n%\n0\n"
    cnf = parse_dimacs_text(src)
    assert cnf.clauses == [[1, -2]]


def test_tautology_dropped():
    cnf = parse_dimacs_text("p cnf 2 2\n1 -1 2 0\n1 2 0\n")
    assert cnf.clauses == [[1, 2]]


def test_duplicate_literal_deduped():
    cnf = parse_dimacs_text("p cnf 2 1\n1 1 -2 0\n")
    assert cnf.clauses == [[1, -2]]


def test_missing_header():
    with pytest.raises(DimacsError):
        parse_dimacs_text("1 2 0\n")


def test_out_of_range_literal():
    with pytest.raises(DimacsError):
        parse_dimacs_text("p cnf 2 1\n1 3 0\n")


def test_dubois_instance():
    cnf = parse_dimacs(REPO / "dubois.txt")
    assert cnf.num_vars == 60
    assert len(cnf.clauses) == 160
    for clause in cnf.clauses:
        assert clause and all(isinstance(l, int) and l != 0 for l in clause)
