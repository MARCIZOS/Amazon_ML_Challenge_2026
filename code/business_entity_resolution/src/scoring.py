"""Shared step used by training and prediction: candidates -> features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.blocking.candidates import decode
from src.features.pair_features import O_COLS, S1_COLS, add_group_features, pair_features


def build_pairs(gen, s1_rows):
    """Returns (meta, X): meta has s1_row, cand, s1_id, other_id; X the features."""
    cands = gen.generate(s1_rows)
    if cands.empty:
        return (pd.DataFrame(columns=["s1_row", "cand", "s1_id", "other_id"]), None)
    s1 = gen.records.take_s1(cands["s1_row"].to_numpy(), S1_COLS + ["entity_id"])
    o = gen.records.take_cand(cands["cand"].to_numpy(), O_COLS + ["entity_id"])
    src, _ = decode(cands["cand"].to_numpy())
    X = pair_features(s1, o, cands["n_keys"].to_numpy(), cands["pre_score"].to_numpy(), src)
    X["s1_row"] = cands["s1_row"].to_numpy()
    X = add_group_features(X).drop(columns=["s1_row"])
    meta = pd.DataFrame({"s1_row": cands["s1_row"].to_numpy(), "cand": cands["cand"].to_numpy(),
                         "s1_id": s1["entity_id"], "other_id": o["entity_id"]})
    return meta, X


def chunks(rows, size):
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    for i in range(0, len(rows), size):
        yield rows[i:i + size]
