"""Entity-level decision: scored pairs -> match list per Source 1 entity.

decide_pairs() applies, in order:
  1. one-to-one: each S2/S3 record is kept only for its highest-scoring S1
     (0 of 7.64M train ground-truth records belong to two S1 entities - D009)
  2. absolute threshold: p >= threshold
  3. relative threshold: p >= relative * (best p of the same S1)
S1 entities with nothing left get an empty list (predicted singletons).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def decide_pairs(scored: pd.DataFrame, threshold: float, relative: float = 0.0,
                 one_to_one: bool = True) -> pd.DataFrame:
    """scored: s1_id, other_id, p  ->  kept pairs (same columns)."""
    df = scored[scored["p"] >= threshold]
    if one_to_one and len(df):
        df = df.sort_values("p", ascending=False, kind="stable").drop_duplicates("other_id", keep="first")
    if relative > 0 and len(df):
        best = df.groupby("s1_id")["p"].transform("max")
        df = df[df["p"] >= relative * best]
    return df.reset_index(drop=True)


def tune_decision(scored: pd.DataFrame, true_pairs: pd.DataFrame, s1_ids, thresholds, relatives,
                  one_to_one=True):
    """Grid search of (threshold, relative) for macro F0.5. Returns (table, best_row)."""
    from src.evaluation.metrics import evaluate
    rows = []
    for t in thresholds:
        for r in relatives:
            kept = decide_pairs(scored, t, r, one_to_one)
            m = evaluate(kept, true_pairs, s1_ids)
            rows.append({"threshold": t, "relative": r, **m})
    table = pd.DataFrame(rows).sort_values("macro_f05", ascending=False).reset_index(drop=True)
    return table, table.iloc[0].to_dict()


def aggregate_matches(scored_pairs, all_s1_ids):
    """Legacy interface used by src/pipeline_baseline.py.

    scored_pairs: iterable of (s1_id, s2s3_id, is_match, score).
    """
    matches = defaultdict(list)
    for s1_id, s2s3_id, is_match, score in scored_pairs:
        if is_match:
            matches[s1_id].append((s2s3_id, score))
    result = {}
    for s1_id in all_s1_ids:
        seen, ids = set(), []
        for mid, _ in sorted(matches.get(s1_id, []), key=lambda x: -x[1]):
            if mid not in seen:
                seen.add(mid)
                ids.append(mid)
        result[s1_id] = ids
    return result


def pairs_to_lists(pairs: pd.DataFrame, all_s1_ids, col="other_id") -> dict:
    """{s1_id: [ids...]} for every S1 id (empty list when no pair)."""
    grouped = pairs.groupby("s1_id")[col].agg(list).to_dict() if len(pairs) else {}
    return {s: grouped.get(s, []) for s in all_s1_ids}
