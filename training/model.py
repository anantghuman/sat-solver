"""NeuroSAT-lite: bipartite message passing with sign-aware edges.

Simplified from the original NeuroSAT: we operate on variable and clause
node embeddings directly (no literal-doubling), pushing sign through the
edge multiplication instead.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .graph import BatchGraph, Graph, VAR_FEAT_DIM, CLAUSE_FEAT_DIM


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class NeuroSATLite(nn.Module):
    def __init__(self, hidden: int = 64, rounds: int = 16) -> None:
        super().__init__()
        self.hidden = hidden
        self.rounds = rounds
        self.var_encoder = nn.Linear(VAR_FEAT_DIM, hidden)
        self.clause_encoder = nn.Linear(CLAUSE_FEAT_DIM, hidden)
        self.var_msg = _mlp(hidden, hidden, hidden)
        self.clause_msg = _mlp(hidden, hidden, hidden)
        self.var_update = _mlp(2 * hidden, hidden, hidden)
        self.clause_update = _mlp(2 * hidden, hidden, hidden)
        self.var_norm = nn.LayerNorm(hidden)
        self.clause_norm = nn.LayerNorm(hidden)
        self.head_score = _mlp(hidden, hidden, 1)
        self.head_polarity = _mlp(hidden, hidden, 1)

    def forward(self, g: Graph) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.var_encoder(g.var_feats)
        c = self.clause_encoder(g.clause_feats)
        for _ in range(self.rounds):
            # variable -> clause: sign-multiplied messages, summed per clause
            var_side = self.var_msg(v)                                 # [V, H]
            edge_v = var_side[g.edge_var] * g.edge_sign.unsqueeze(-1)  # [E, H]
            c_agg = torch.zeros_like(c)
            c_agg.index_add_(0, g.edge_clause, edge_v)
            c = self.clause_norm(self.clause_update(torch.cat([c, c_agg], dim=-1)))
            # clause -> variable: same story back
            clause_side = self.clause_msg(c)
            edge_c = clause_side[g.edge_clause] * g.edge_sign.unsqueeze(-1)
            v_agg = torch.zeros_like(v)
            v_agg.index_add_(0, g.edge_var, edge_c)
            v = self.var_norm(self.var_update(torch.cat([v, v_agg], dim=-1)))
        scores = self.head_score(v).squeeze(-1)          # [V]
        polarity_logits = self.head_polarity(v).squeeze(-1)  # [V]
        return scores, polarity_logits


def masked_log_softmax_per_graph(
    scores: torch.Tensor, batch: BatchGraph, mask: torch.Tensor
) -> torch.Tensor:
    """Per-graph log-softmax over variable scores, applying an unassigned
    mask. Uses the CSR graph_var_ptr for slicing. Returns shape [V]."""
    out = torch.full_like(scores, float("-inf"))
    scores_masked = scores.masked_fill(~mask, float("-inf"))
    for gi in range(batch.graph_var_ptr.shape[0] - 1):
        s = batch.graph_var_ptr[gi].item()
        e = batch.graph_var_ptr[gi + 1].item()
        seg = scores_masked[s:e]
        if torch.isfinite(seg).any():
            out[s:e] = torch.log_softmax(seg, dim=0)
    return out
