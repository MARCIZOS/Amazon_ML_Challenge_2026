"""Entity-level decision: scored pairs -> match list per Source 1 entity.

decide_pairs() applies, in order:
  1. one-to-one: each S2/S3 record is kept only for its highest-scoring S1
     (0 of 7.64M train ground-truth records belong to two S1 entities - D009)
  2. absolute threshold: p >= threshold
  3. relative threshold: p >= relative * (best p of the same S1)
  4. singleton protection: if an S1 only has 1 match, require p >= single_min_p
S1 entities with nothing left get an empty list (predicted singletons).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def decide_pairs(scored: pd.DataFrame, threshold: float, relative: float = 0.0,
                 one_to_one: bool = True, single_min_p: float = 0.0) -> pd.DataFrame:
    """scored: s1_id, other_id, p  ->  kept pairs (same columns)."""
    df = scored[scored["p"] >= threshold]
    if one_to_one and len(df):
        df = df.sort_values("p", ascending=False, kind="stable").drop_duplicates("other_id", keep="first")
    if relative > 0 and len(df):
        best = df.groupby("s1_id")["p"].transform("max")
        df = df[df["p"] >= relative * best]
    if single_min_p > threshold and len(df):
        counts = df.groupby("s1_id")["p"].transform("size")
        df = df[(counts > 1) | (df["p"] >= single_min_p)]
    return df.reset_index(drop=True)


def tune_decision(scored: pd.DataFrame, true_pairs: pd.DataFrame, s1_ids, thresholds, relatives,
                  one_to_one=True, single_min_p_values=None):
    """Grid search of (threshold, relative, single_min_p) for macro F0.5. Returns (table, best_row)."""
    from src.evaluation.metrics import evaluate
    rows = []
    single_grid = single_min_p_values if single_min_p_values is not None else [0.0]
    for t in thresholds:
        for r in relatives:
            for smp in single_grid:
                # only evaluate meaningful smp >= t
                eff_smp = max(smp, t) if smp > 0.0 else 0.0
                kept = decide_pairs(scored, t, r, one_to_one, single_min_p=eff_smp)
                m = evaluate(kept, true_pairs, s1_ids)
                rows.append({"threshold": t, "relative": r, "single_min_p": eff_smp, **m})
    table = pd.DataFrame(rows).sort_values("macro_f05", ascending=False).reset_index(drop=True)
    return table, table.iloc[0].to_dict()


def pairs_to_lists(pairs: pd.DataFrame, all_s1_ids, col="other_id") -> dict:
    """{s1_id: [ids...]} for every S1 id (empty list when no pair)."""
    grouped = pairs.groupby("s1_id")[col].agg(list).to_dict() if len(pairs) else {}
    return {s: grouped.get(s, []) for s in all_s1_ids}
