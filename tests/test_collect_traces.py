from solver.parser import Cnf

from training.collect_traces import collect_from_cnf


def _random_3sat(num_vars, num_clauses, seed):
    import random
    rng = random.Random(seed)
    clauses = []
    for _ in range(num_clauses):
        vs = rng.sample(range(1, num_vars + 1), 3)
        clauses.append([v if rng.random() < 0.5 else -v for v in vs])
    return Cnf(num_vars=num_vars, clauses=clauses)


def test_collect_produces_decisions():
    cnf = _random_3sat(15, 60, seed=7)
    trace = collect_from_cnf(cnf)
    assert trace.num_vars == 15
    assert len(trace.clauses) == 60
    assert len(trace.decisions) > 0
    for d in trace.decisions:
        assert 1 <= d.chosen_var <= 15
        assert d.chosen_sign in (-1, 1)
        assert len(d.assignment) == 16  # index 0..15
        # The chosen variable was unassigned at decision time.
        assert d.assignment[d.chosen_var] == 0


def test_collect_respects_max_decisions():
    cnf = _random_3sat(20, 80, seed=3)
    trace = collect_from_cnf(cnf, max_decisions=5)
    assert len(trace.decisions) == 5
