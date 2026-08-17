"""Build the bipartite variable–clause graph for a formula state.

Node types:
  variable nodes  (V of them, one per SAT variable)
  clause nodes    (C of them)

Edges: one per literal occurrence, tagged with sign (+1 for positive, -1
for negative). Batching is done manually with offset-shifted indices so
we don't need torch_geometric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

VAR_FEAT_DIM = 5
CLAUSE_FEAT_DIM = 3


@dataclass
class Graph:
    var_feats: torch.Tensor    # [V, VAR_FEAT_DIM]
    clause_feats: torch.Tensor # [C, CLAUSE_FEAT_DIM]
    edge_var: torch.Tensor     # [E] var indices (long)
    edge_clause: torch.Tensor  # [E] clause indices (long)
    edge_sign: torch.Tensor    # [E] float32 in {-1, +1}


@dataclass
class BatchGraph(Graph):
    var_batch: torch.Tensor    # [V] which graph in the batch each var belongs to
    graph_var_ptr: torch.Tensor  # [B+1] CSR-style boundaries for slicing per-graph var scores


def build_graph(
    num_vars: int,
    clauses: Sequence[Sequence[int]],
    assignment: Sequence[int],
    vsids_activity: Sequence[float] | None = None,
    is_learned: Sequence[bool] | None = None,
) -> Graph:
    """Construct a Graph for a single formula state.

    `assignment`: length num_vars+1, values in {-1, 0, +1}. Index 0 unused.
    `vsids_activity`: length num_vars+1, optional. Zeros if not provided.
    `is_learned`: per-clause bool, optional. Default False.
    """
    if vsids_activity is None:
        vsids_activity = [0.0] * (num_vars + 1)
    if is_learned is None:
        is_learned = [False] * len(clauses)

    # Variable features
    pos_occ = [0] * (num_vars + 1)
    neg_occ = [0] * (num_vars + 1)
    edge_var: list[int] = []
    edge_clause: list[int] = []
    edge_sign: list[float] = []
    clause_lens: list[int] = []
    num_unassigned: list[int] = []

    for c_idx, clause in enumerate(clauses):
        clause_lens.append(len(clause))
        unassigned_here = 0
        for lit in clause:
            v = abs(lit)
            sign = 1.0 if lit > 0 else -1.0
            edge_var.append(v - 1)
            edge_clause.append(c_idx)
            edge_sign.append(sign)
            if lit > 0:
                pos_occ[v] += 1
            else:
                neg_occ[v] += 1
            if assignment[v] == 0:
                unassigned_here += 1
        num_unassigned.append(unassigned_here)

    var_feats = torch.zeros((num_vars, VAR_FEAT_DIM), dtype=torch.float32)
    for v in range(1, num_vars + 1):
        av = assignment[v]
        var_feats[v - 1, 0] = 1.0 if av != 0 else 0.0
        var_feats[v - 1, 1] = 1.0 if av > 0 else (-1.0 if av < 0 else 0.0)
        var_feats[v - 1, 2] = float(vsids_activity[v])
        var_feats[v - 1, 3] = float(pos_occ[v])
        var_feats[v - 1, 4] = float(neg_occ[v])

    clause_feats = torch.zeros((len(clauses), CLAUSE_FEAT_DIM), dtype=torch.float32)
    for c_idx, cl in enumerate(clauses):
        clause_feats[c_idx, 0] = float(clause_lens[c_idx])
        clause_feats[c_idx, 1] = float(num_unassigned[c_idx])
        clause_feats[c_idx, 2] = 1.0 if is_learned[c_idx] else 0.0

    return Graph(
        var_feats=var_feats,
        clause_feats=clause_feats,
        edge_var=torch.tensor(edge_var, dtype=torch.long),
        edge_clause=torch.tensor(edge_clause, dtype=torch.long),
        edge_sign=torch.tensor(edge_sign, dtype=torch.float32),
    )


def collate(graphs: list[Graph]) -> BatchGraph:
    """Stack single-formula graphs into one big graph with disjoint index
    ranges. Returns per-var batch indices and a CSR pointer for slicing."""
    var_feats = torch.cat([g.var_feats for g in graphs], dim=0)
    clause_feats = torch.cat([g.clause_feats for g in graphs], dim=0)
    edge_var_parts = []
    edge_clause_parts = []
    edge_sign_parts = []
    var_batch_parts = []
    graph_var_ptr = [0]
    v_off = 0
    c_off = 0
    for i, g in enumerate(graphs):
        edge_var_parts.append(g.edge_var + v_off)
        edge_clause_parts.append(g.edge_clause + c_off)
        edge_sign_parts.append(g.edge_sign)
        var_batch_parts.append(torch.full((g.var_feats.shape[0],), i, dtype=torch.long))
        v_off += g.var_feats.shape[0]
        c_off += g.clause_feats.shape[0]
        graph_var_ptr.append(v_off)
    return BatchGraph(
        var_feats=var_feats,
        clause_feats=clause_feats,
        edge_var=torch.cat(edge_var_parts),
        edge_clause=torch.cat(edge_clause_parts),
        edge_sign=torch.cat(edge_sign_parts),
        var_batch=torch.cat(var_batch_parts),
        graph_var_ptr=torch.tensor(graph_var_ptr, dtype=torch.long),
    )
