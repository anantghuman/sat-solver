from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class DimacsError(ValueError):
    pass


@dataclass
class Cnf:
    num_vars: int
    clauses: list[list[int]]


def parse_dimacs(path: str | Path) -> Cnf:
    with open(path, "r") as fh:
        return parse_dimacs_text(fh.read())


def parse_dimacs_text(text: str) -> Cnf:
    tokens = _stream_tokens(text)
    header = next(tokens, None)
    if header is None:
        raise DimacsError("empty input: no `p cnf` header")
    if header != "p":
        raise DimacsError(f"expected `p cnf N M` header, got token {header!r}")
    if next(tokens, None) != "cnf":
        raise DimacsError("expected `cnf` after `p`")
    try:
        num_vars = int(next(tokens))
        _ = int(next(tokens))  # declared clause count — advisory only
    except (StopIteration, ValueError) as exc:
        raise DimacsError("malformed `p cnf N M` header") from exc

    clauses: list[list[int]] = []
    current: list[int] = []
    seen: set[int] = set()
    tautology = False
    for tok in tokens:
        try:
            lit = int(tok)
        except ValueError as exc:
            raise DimacsError(f"expected integer literal, got {tok!r}") from exc
        if lit == 0:
            if not tautology:
                clauses.append(current)
            current = []
            seen = set()
            tautology = False
            continue
        if abs(lit) > num_vars:
            raise DimacsError(f"literal {lit} out of range for {num_vars} variables")
        if -lit in seen:
            tautology = True
        if lit not in seen:
            seen.add(lit)
            current.append(lit)
    if current:
        raise DimacsError("trailing clause missing 0 terminator")

    return Cnf(num_vars=num_vars, clauses=clauses)


def _stream_tokens(text: str) -> Iterable[str]:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line[0] == "c":
            continue
        if line[0] == "%":
            # SATLIB files often terminate with `%\n0\n`; ignore anything past.
            break
        for tok in line.split():
            yield tok
