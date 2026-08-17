"""GNN-guided branching. Rebuilds the current formula graph on demand and
picks the highest-scoring unassigned variable, sign from the polarity head.

Falls back to VSIDS if the model somehow returns a picked variable that's
already assigned, or if torch isn't available (import error surfaces
lazily so vsids-only users don't need torch installed).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .branching import Brancher, VSIDSBrancher

if TYPE_CHECKING:
    from .cdcl import Solver


class GNNBrancher(Brancher):
    def __init__(self, num_vars: int, model_path: str | Path) -> None:
        import torch

        from training.graph import build_graph
        from training.model import NeuroSATLite

        self.num_vars = num_vars
        self._torch = torch
        self._build_graph = build_graph

        meta_path = Path(model_path).with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"hidden": 64, "rounds": 16}
        self.model = NeuroSATLite(hidden=meta["hidden"], rounds=meta["rounds"])
        state = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        self._fallback = VSIDSBrancher(num_vars)

    def on_assign(self, var: int, value: bool) -> None:
        self._fallback.on_assign(var, value)

    def on_unassign(self, var: int) -> None:
        self._fallback.on_unassign(var)

    def on_conflict(self, learned, seen_vars) -> None:
        self._fallback.on_conflict(learned, seen_vars)

    def pick(self, solver: "Solver") -> int:
        torch = self._torch
        # Encode current assignment as {-1, 0, +1}.
        values = solver.trail.values
        assignment = [0] * (self.num_vars + 1)
        for v in range(1, self.num_vars + 1):
            if values[v] is True:
                assignment[v] = 1
            elif values[v] is False:
                assignment[v] = -1

        clauses = [list(c.lits) for c in solver.store.clauses]
        is_learned = [c.is_learned for c in solver.store.clauses]
        activity = self._fallback.activity if hasattr(self._fallback, "activity") else None
        g = self._build_graph(
            num_vars=self.num_vars,
            clauses=clauses,
            assignment=assignment,
            vsids_activity=activity,
            is_learned=is_learned,
        )

        with torch.no_grad():
            scores, polarity = self.model(g)
        mask = torch.tensor([values[v] is None for v in range(1, self.num_vars + 1)], dtype=torch.bool)
        if not mask.any():
            return 0
        scores = scores.masked_fill(~mask, float("-inf"))
        var_idx = int(torch.argmax(scores).item())
        var = var_idx + 1
        if values[var] is not None:
            return self._fallback.pick(solver)
        pos = torch.sigmoid(polarity[var_idx]).item() >= 0.5
        return var if pos else -var
