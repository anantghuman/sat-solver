from solver.clause import Clause, ClauseStore
from solver.trail import Trail


def test_clause_store_registers_watches():
    store = ClauseStore(num_vars=3)
    c = store.add_original([1, -2, 3])
    assert c is not None
    assert store.watches_for(1) == [c]
    assert store.watches_for(-2) == [c]
    assert store.watches_for(3) == []
    a, b = c.watched_lits()
    assert {a, b} == {1, -2}


def test_clause_store_unit_watches_single_lit():
    store = ClauseStore(num_vars=3)
    c = store.add_original([1])
    assert c is not None
    assert store.watches_for(1) == [c]
    assert c.w0 == c.w1 == 0


def test_add_learned_watched_uses_caller_indices():
    store = ClauseStore(num_vars=4)
    c = store.add_learned_watched([1, 2, 3, 4], w0=2, w1=3)
    assert c.watched_lits() == (3, 4)
    assert store.watches_for(3) == [c]
    assert store.watches_for(4) == [c]


def test_other_watch_returns_partner():
    c = Clause(lits=[1, -2, 3], w0=0, w1=1)
    assert c.other_watch(1) == -2
    assert c.other_watch(-2) == 1


def test_trail_assign_and_lookup():
    t = Trail(num_vars=3)
    t.enqueue(1, None)
    t.enqueue(-2, None)
    assert t.is_true(1) and t.is_false(-1)
    assert t.is_true(-2) and t.is_false(2)
    assert t.is_unassigned(3)


def test_trail_levels_and_backjump():
    t = Trail(num_vars=4)
    t.enqueue(1, None)              # level 0 (propagation-style, no decision made yet)
    assert t.decision_level == 0
    t.new_decision_level(); t.enqueue(2, None)  # level 1
    t.new_decision_level(); t.enqueue(3, None)  # level 2
    t.new_decision_level(); t.enqueue(4, None)  # level 3
    assert t.decision_level == 3
    popped = t.backjump(1)
    assert set(popped) == {3, 4}
    assert t.decision_level == 1
    assert t.is_true(1) and t.is_true(2)
    assert t.is_unassigned(3) and t.is_unassigned(4)


def test_trail_backjump_to_zero():
    t = Trail(num_vars=2)
    t.new_decision_level(); t.enqueue(1, None)
    t.new_decision_level(); t.enqueue(2, None)
    t.backjump(0)
    assert t.decision_level == 0
    assert t.is_unassigned(1) and t.is_unassigned(2)
