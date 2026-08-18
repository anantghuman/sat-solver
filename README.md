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

## GNN brancher (Phase B–E)

Beyond VSIDS, there's a GNN-guided brancher trained on solver traces.

```
# 1. download benchmarks (needs network access to SATLIB)
python -m data.download_satlib --bucket uf20-91

# 2. collect decision traces from vsids runs
python -m training.collect_traces --bucket uf20-91 --out data/traces/uf20.pkl

# 3. train a neurosat-lite model to imitate vsids
python -m training.train --traces data/traces/uf20.pkl --out models/gnn.pt

# 4. use the model as a brancher
python -m solver <file.cnf> --brancher gnn --model models/gnn.pt

# 5. compare against vsids on a held-out bucket
python -m eval.compare --bucket uf50-218 --model models/gnn.pt
```

If SATLIB isn't reachable, `collect_traces --synthetic N` generates random
3-SAT instances on the fly for training.

Model architecture is bipartite (variable–clause) message passing with
sign-aware edges, 16 rounds, hidden=64. Loss is CE on the chosen
variable + BCE on the chosen polarity. See `training/model.py`.

Expectations honestly: on training-adjacent buckets the GNN is competitive
with VSIDS on decision count. On out-of-distribution instances (like
`dubois.txt`) it typically loses — reported, not hidden.
