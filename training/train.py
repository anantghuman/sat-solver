"""Supervised warm-start: train the GNN to reproduce VSIDS's decisions.

Loss = CE(softmax(unassigned-masked scores), one-hot(chosen_var))
     + BCE(sigmoid(polarity), chosen_sign > 0).

Batches are packed by edge count.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.optim import Adam

from training.collect_traces import DecisionRecord, InstanceTrace
from training.graph import Graph, build_graph, collate
from training.model import NeuroSATLite, masked_log_softmax_per_graph


def _flatten(traces: Sequence[InstanceTrace]) -> list[tuple[InstanceTrace, DecisionRecord]]:
    out = []
    for t in traces:
        for d in t.decisions:
            out.append((t, d))
    return out


def _graph_and_target(t: InstanceTrace, d: DecisionRecord) -> tuple[Graph, int, float]:
    g = build_graph(num_vars=t.num_vars, clauses=t.clauses, assignment=d.assignment)
    # target var index within graph is (chosen_var - 1)
    return g, d.chosen_var - 1, 1.0 if d.chosen_sign > 0 else 0.0


def _pack_batches(samples, edge_budget: int):
    random.shuffle(samples)
    batch: list = []
    var_targets: list = []
    pol_targets: list = []
    edges = 0
    for g, tv, tp in samples:
        e = g.edge_var.shape[0]
        if batch and edges + e > edge_budget:
            yield batch, var_targets, pol_targets
            batch, var_targets, pol_targets, edges = [], [], [], 0
        batch.append(g)
        var_targets.append(tv)
        pol_targets.append(tp)
        edges += e
    if batch:
        yield batch, var_targets, pol_targets


def _run_batch(model: NeuroSATLite, graphs, var_targets, pol_targets, *, train: bool, opt=None):
    b = collate(graphs)
    scores, pol_logits = model(b)
    is_assigned = b.var_feats[:, 0].bool()
    unassigned = ~is_assigned
    lsm = masked_log_softmax_per_graph(scores, b, unassigned)

    # Gather log-prob at the target variable per graph
    ce_losses = []
    top1_hits = 0
    for gi in range(b.graph_var_ptr.shape[0] - 1):
        s = b.graph_var_ptr[gi].item()
        e = b.graph_var_ptr[gi + 1].item()
        tv = var_targets[gi]
        logp = lsm[s + tv]
        if torch.isfinite(logp):
            ce_losses.append(-logp)
            # top-1 agreement in unassigned mask
            argmax = torch.argmax(lsm[s:e])
            if argmax.item() == tv:
                top1_hits += 1
    if not ce_losses:
        return None, 0
    ce = torch.stack(ce_losses).mean()

    # Polarity BCE only on target vars
    pol_target_tensor = torch.tensor(pol_targets, dtype=torch.float32)
    pol_gathered = torch.stack(
        [pol_logits[b.graph_var_ptr[gi].item() + var_targets[gi]] for gi in range(len(var_targets))]
    )
    bce = F.binary_cross_entropy_with_logits(pol_gathered, pol_target_tensor)

    loss = ce + bce
    if train:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), top1_hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--edge-budget", type=int, default=50000)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.traces, "rb") as fh:
        traces: list[InstanceTrace] = pickle.load(fh)
    flat = _flatten(traces)
    random.shuffle(flat)
    n_hold = max(1, int(len(flat) * args.holdout_frac))
    holdout = flat[:n_hold]
    train_set = flat[n_hold:]
    print(f"training on {len(train_set)} decisions, held out {len(holdout)}")

    train_samples = [_graph_and_target(t, d) for t, d in train_set]
    holdout_samples = [_graph_and_target(t, d) for t, d in holdout]

    model = NeuroSATLite(hidden=args.hidden, rounds=args.rounds)
    opt = Adam(model.parameters(), lr=args.lr)

    best_top1 = -1.0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        train_hits = 0
        train_n = 0
        for batch, tvs, tps in _pack_batches(list(train_samples), args.edge_budget):
            loss, hits = _run_batch(model, batch, tvs, tps, train=True, opt=opt)
            if loss is not None:
                losses.append(loss)
                train_hits += hits
                train_n += len(batch)
        model.eval()
        hold_hits = 0
        hold_n = 0
        with torch.no_grad():
            for batch, tvs, tps in _pack_batches(list(holdout_samples), args.edge_budget):
                _, hits = _run_batch(model, batch, tvs, tps, train=False)
                hold_hits += hits
                hold_n += len(batch)
        tl = sum(losses) / max(1, len(losses))
        train_top1 = train_hits / max(1, train_n)
        hold_top1 = hold_hits / max(1, hold_n)
        print(f"epoch {epoch:2d}  loss={tl:.4f}  train_top1={train_top1:.3f}  hold_top1={hold_top1:.3f}")

        if hold_top1 > best_top1:
            best_top1 = hold_top1
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)
            meta_path = args.out.with_suffix(".json")
            with open(meta_path, "w") as fh:
                json.dump({"hidden": args.hidden, "rounds": args.rounds}, fh)

    print(f"best held-out top-1: {best_top1:.3f}, saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
