from itertools import islice

from solver.branching import VSIDSBrancher, luby


def test_luby_prefix():
    # Classic Luby: 1,1,2,1,1,2,4,1,1,2,1,1,2,4,8,...
    seq = list(islice(luby(unit=1), 15))
    assert seq == [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8]


def test_luby_scaled():
    seq = list(islice(luby(unit=32), 4))
    assert seq == [32, 32, 64, 32]


def test_vsids_bump_orders_by_activity():
    b = VSIDSBrancher(num_vars=3)

    class _StubSolver:
        def __init__(self, values):
            self.num_vars = 3
            self.trail = type("T", (), {"values": values})()

    # Initially all activities zero, tie broken by heap order.
    stub = _StubSolver([None, None, None, None])
    b.on_conflict(learned=[], seen_vars=[2])
    lit = b.pick(stub)
    assert abs(lit) == 2  # var 2 was bumped, so it wins


def test_vsids_phase_saving():
    b = VSIDSBrancher(num_vars=2)
    b.on_assign(1, True)
    # After unassign, phase memory persists.
    b.on_unassign(1)
    b.on_conflict(learned=[], seen_vars=[1])

    class _StubSolver:
        num_vars = 2
        trail = type("T", (), {"values": [None, None, None]})()

    lit = b.pick(_StubSolver())
    assert lit == 1  # var 1, positive because last saved phase was True
