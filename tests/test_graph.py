import torch

from training.graph import build_graph, collate, VAR_FEAT_DIM, CLAUSE_FEAT_DIM


def test_build_graph_shapes():
    clauses = [[1, -2, 3], [-1, 2], [1]]
    g = build_graph(num_vars=3, clauses=clauses, assignment=[0, 0, 1, 0])
    assert g.var_feats.shape == (3, VAR_FEAT_DIM)
    assert g.clause_feats.shape == (3, CLAUSE_FEAT_DIM)
    assert g.edge_var.shape == (6,)
    assert g.edge_clause.shape == (6,)
    assert g.edge_sign.shape == (6,)
    # Var 2 is assigned True → its is_assigned=1, val=1.
    assert g.var_feats[1, 0].item() == 1.0
    assert g.var_feats[1, 1].item() == 1.0
    # Clause 0 has length 3, 2 unassigned (var 1 and 3), var 2 is assigned.
    assert g.clause_feats[0, 0].item() == 3.0
    assert g.clause_feats[0, 1].item() == 2.0
    # Signs
    signs = g.edge_sign.tolist()
    assert +1.0 in signs and -1.0 in signs


def test_build_graph_occurrence_counts():
    g = build_graph(num_vars=2, clauses=[[1, -2], [1], [-1, 2]], assignment=[0, 0, 0])
    # var 1: 2 pos, 1 neg; var 2: 1 pos, 1 neg
    assert g.var_feats[0, 3].item() == 2.0
    assert g.var_feats[0, 4].item() == 1.0
    assert g.var_feats[1, 3].item() == 1.0
    assert g.var_feats[1, 4].item() == 1.0


def test_collate_offsets_indices():
    g1 = build_graph(2, [[1, -2]], [0, 0, 0])
    g2 = build_graph(3, [[1, 2, 3]], [0, 0, 0, 0])
    b = collate([g1, g2])
    assert b.var_feats.shape[0] == 5
    assert b.clause_feats.shape[0] == 2
    # g2's variable edges should live in indices [2, 5)
    assert (b.edge_var[-3:] >= 2).all()
    # per-graph pointer covers both
    assert b.graph_var_ptr.tolist() == [0, 2, 5]
    assert b.var_batch.tolist() == [0, 0, 1, 1, 1]
