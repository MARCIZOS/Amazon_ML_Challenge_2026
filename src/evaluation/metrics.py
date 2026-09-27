"""Official-style evaluation (Member 4).

Macro F0.5 over ALL Source 1 entities of the evaluation set:
  * true matches empty  -> 1.0 if the prediction is empty, else 0.0
  * prediction empty, true matches non-empty -> 0.0
  * otherwise F0.5 = 1.25 P R / (0.25 P + R)   (0 when no true positive)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def f05(tp, n_pred, n_true):
    tp, n_pred, n_true = (np.asarray(x, dtype=np.float64) for x in (tp, n_pred, n_true))
    out = np.zeros_like(tp)
    single = n_true == 0
    out[single] = (n_pred[single] == 0).astype(float)
    m = (~single) & (n_pred > 0) & (tp > 0)
    p = tp[m] / n_pred[m]
    r = tp[m] / n_true[m]
    out[m] = 1.25 * p * r / (0.25 * p + r)
    return out


def evaluate(pred_pairs: pd.DataFrame, true_pairs: pd.DataFrame, s1_ids) -> dict:
    """pred_pairs / true_pairs: columns s1_id, other_id. s1_ids: every evaluated S1."""
    s1_ids = pd.Index(pd.unique(np.asarray(s1_ids)))
    pred = pred_pairs[["s1_id", "other_id"]].drop_duplicates()
    true = true_pairs[["s1_id", "other_id"]].drop_duplicates()
    pred = pred[pred.s1_id.isin(s1_ids)]
    true = true[true.s1_id.isin(s1_ids)]
    tp = pred.merge(true, on=["s1_id", "other_id"])
    n_pred = pred.groupby("s1_id").size().reindex(s1_ids, fill_value=0).to_numpy()
    n_true = true.groupby("s1_id").size().reindex(s1_ids, fill_value=0).to_numpy()
    n_tp = tp.groupby("s1_id").size().reindex(s1_ids, fill_value=0).to_numpy()
    f = f05(n_tp, n_pred, n_true)
    single = n_true == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        prec_e = np.where(n_pred > 0, n_tp / np.maximum(n_pred, 1), np.nan)
        rec_e = np.where(n_true > 0, n_tp / np.maximum(n_true, 1), np.nan)
    return {
        "n_s1": int(len(s1_ids)),
        "macro_f05": float(f.mean()),
        "macro_f05_singletons": float(f[single].mean()) if single.any() else None,
        "macro_f05_non_singletons": float(f[~single].mean()) if (~single).any() else None,
        "singleton_share": float(single.mean()),
        "pairs_pred": int(n_pred.sum()), "pairs_true": int(n_true.sum()), "pairs_tp": int(n_tp.sum()),
        "micro_precision": float(n_tp.sum() / max(1, n_pred.sum())),
        "micro_recall": float(n_tp.sum() / max(1, n_true.sum())),
        "macro_precision": float(np.nanmean(prec_e)) if np.isfinite(prec_e).any() else None,
        "macro_recall": float(np.nanmean(rec_e)) if np.isfinite(rec_e).any() else None,
        "false_merges_on_singletons": int(((n_pred > 0) & single).sum()),
    }


def blocking_report(cand_pairs: pd.DataFrame, true_pairs: pd.DataFrame, s1_ids, n_other_total) -> dict:
    """Recall ceiling and reduction ratio of the candidate set."""
    s1_ids = pd.Index(pd.unique(np.asarray(s1_ids)))
    true = true_pairs[true_pairs.s1_id.isin(s1_ids)][["s1_id", "other_id"]].drop_duplicates()
    cand = cand_pairs[["s1_id", "other_id"]].drop_duplicates()
    hit = true.merge(cand, on=["s1_id", "other_id"])
    per = cand.groupby("s1_id").size().reindex(s1_ids, fill_value=0)
    all_pairs = len(s1_ids) * float(n_other_total)
    return {
        "blocking_recall": float(len(hit) / max(1, len(true))),
        "true_pairs": int(len(true)), "true_pairs_in_candidates": int(len(hit)),
        "candidate_pairs": int(len(cand)),
        "candidates_per_s1_mean": float(per.mean()), "candidates_per_s1_max": int(per.max()) if len(per) else 0,
        "s1_without_candidates": int((per == 0).sum()),
        "reduction_ratio": float(1 - len(cand) / all_pairs) if all_pairs else None,
    }
