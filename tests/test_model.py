import torch

from training.graph import build_graph, collate
from training.model import NeuroSATLite, masked_log_softmax_per_graph


def test_forward_pass_single_graph():
    g = build_graph(num_vars=4, clauses=[[1, -2, 3], [-1, 2, 4], [-3, -4]], assignment=[0, 0, 0, 0, 0])
    m = NeuroSATLite(hidden=16, rounds=2)
    scores, pol = m(g)
    assert scores.shape == (4,)
    assert pol.shape == (4,)
    assert torch.isfinite(scores).all()


def test_forward_pass_batched():
    g1 = build_graph(3, [[1, 2, 3], [-1, -2]], [0, 0, 0, 0])
    g2 = build_graph(5, [[1, -2, 3, 4], [-3, 5]], [0, 0, 0, 0, 0, 0])
    b = collate([g1, g2])
    m = NeuroSATLite(hidden=8, rounds=2)
    scores, pol = m(b)
    assert scores.shape == (3 + 5,)
    assert pol.shape == (3 + 5,)


def test_masked_log_softmax_respects_assigned_mask():
    g1 = build_graph(4, [[1, 2, 3]], [0, 1, 0, 0, 0])   # var 1 assigned
    g2 = build_graph(3, [[1, 2, 3]], [0, 0, 0, 0])
    b = collate([g1, g2])
    # mask: True for unassigned
    is_assigned = b.var_feats[:, 0].bool()
    unassigned = ~is_assigned
    scores = torch.randn(b.var_feats.shape[0])
    lsm = masked_log_softmax_per_graph(scores, b, unassigned)
    # first graph: index 0 (var 1) should be -inf
    assert lsm[0].item() == float("-inf")
    # other three in first graph should be finite and sum-exp to 1
    seg = lsm[1:4]
    assert torch.allclose(seg.exp().sum(), torch.tensor(1.0), atol=1e-5)
