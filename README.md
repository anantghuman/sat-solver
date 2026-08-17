# sat-solver

A CDCL SAT solver in Python, being extended with a GNN-guided branching
heuristic. This PR is the solver core; the ML layer lands in a follow-up.

## Install

```
pip install -r requirements.txt
```

(`torch` is listed but only needed for the upcoming GNN brancher; the
VSIDS solver runs with just the stdlib.)

## Usage

```
python -m solver <file.cnf> [--brancher vsids|gnn] [--model PATH]
                            [--restart-unit N] [--timeout SEC] [--quiet]
```

Output follows the DIMACS competition convention:

```
$ python -m solver dubois.txt
s UNSATISFIABLE
c decisions=303 conflicts=150 propagations=1370 restarts=3 learned=145 time=0.009s
```

Exit codes: `10` for SAT, `20` for UNSAT.

## What's in this PR (Phase A)

- `solver/parser.py` — tolerant DIMACS parser (multi-line clauses,
  `%` end marker, tautology dropping, dedup).
- `solver/clause.py` — clause store with 2-watched literals.
- `solver/trail.py` — assignment trail with O(1) lookup and O(popped)
  backjump.
- `solver/cdcl.py` — CDCL loop: watched-literal BCP, 1-UIP conflict
  analysis, clause learning.
- `solver/branching.py` — `Brancher` ABC, `VSIDSBrancher` (exponential
  decay via growing bump increment + phase saving), Luby restart
  schedule.
- `solver/main.py` — CLI entry point.

## Tests

```
pytest -q
```

## What's next (Phase B–E, follow-up PR)

- SATLIB benchmark download + trace collection.
- NeuroSAT-lite GNN model + supervised training on VSIDS traces.
- `GNNBrancher` plugged into the same `Brancher` seam.
- Evaluation harness comparing VSIDS vs GNN on held-out instances.
