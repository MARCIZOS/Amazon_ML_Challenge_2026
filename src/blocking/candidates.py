"""Candidate generation from the on-disk blocking indexes (Member 4).

Stage 1 - key join: for every S1 record, all S2/S3 records sharing at least one
          blocking key (union over keys; oversized keys were already dropped by
          src.preprocessing.build_indexes). Implemented with sorted uint64 key
          arrays + numpy searchsorted: no all-pairs comparison, memory ~13 bytes
          per index row.
Stage 2 - keep the `pre_k` candidates sharing the most keys.
Stage 3 - cheap re-rank with name/address token-set similarity (vectorised
          rapidfuzz) and keep `top_k`. These top_k are exactly the pairs the
          model scores, i.e. candidate_pairs.tsv.

Candidate ids are encoded as int64:  (src - 2) << 23 | row   (row < 8.3M).
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROW_BITS = 23
ROW_MASK = (1 << ROW_BITS) - 1
PAIR_SHIFT = 25                      # s1_row << 25 | cand   (cand < 2^24)

FEATURE_COLS = ["entity_id", "name_core", "name_key", "name_compact", "name_alt", "name_legal",
                "name_translit", "addr_core", "addr_key", "addr_nums", "addr_state", "addr_missing"]


def encode(src, row):
    return ((np.asarray(src, dtype=np.int64) - 2) << ROW_BITS) | np.asarray(row, dtype=np.int64)


def decode(cand):
    cand = np.asarray(cand, dtype=np.int64)
    return (cand >> ROW_BITS) + 2, cand & ROW_MASK


def _expand(s1_rows, s1_keys, okeys, ocands):
    """All (s1_row, cand) pairs whose key matches (both key arrays; okeys sorted)."""
    lo = np.searchsorted(okeys, s1_keys, side="left")
    hi = np.searchsorted(okeys, s1_keys, side="right")
    n = hi - lo
    keep = n > 0
    lo, n, s1_rows = lo[keep], n[keep], s1_rows[keep]
    total = int(n.sum())
    if total == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    starts = np.repeat(lo - np.concatenate(([0], np.cumsum(n)[:-1])), n)
    idx = starts + np.arange(total)
    return np.repeat(s1_rows, n).astype(np.int64), ocands[idx]


class Records:
    """Normalised records of one split held as pyarrow tables (compact strings)."""

    def __init__(self, processed, split, cols=FEATURE_COLS):
        self.tables = {}
        for k in (1, 2, 3):
            self.tables[k] = pq.read_table(os.path.join(processed, split, f"source{k}.parquet"), columns=cols)
        self.n = {k: t.num_rows for k, t in self.tables.items()}

    def s1_ids(self):
        return self.tables[1].column("entity_id").to_numpy(zero_copy_only=False)

    def take_s1(self, rows, cols):
        t = self.tables[1].take(np.asarray(rows, dtype=np.int64))
        return {c: np.asarray(t.column(c).to_pylist(), dtype=object) for c in cols}

    def take_cand(self, cand, cols):
        """Column values for encoded candidate ids, in the given order."""
        src, row = decode(cand)
        out = {c: np.empty(len(cand), dtype=object) for c in cols}
        for k in (2, 3):
            pos = np.nonzero(src == k)[0]
            if len(pos) == 0:
                continue
            t = self.tables[k].take(row[pos])
            for c in cols:
                out[c][pos] = np.asarray(t.column(c).to_pylist(), dtype=object)
        return out


class CandidateGenerator:
    def __init__(self, processed, split, keys, pre_k=120, top_k=40, records: Records | None = None):
        t0 = time.time()
        self.processed, self.split = processed, split
        self.keys = list(keys)
        self.pre_k, self.top_k = pre_k, top_k
        idx = os.path.join(processed, split, "index")
        self.others, self.s1 = {}, {}
        for name in self.keys:
            o = pd.read_parquet(os.path.join(idx, f"{name}__others.parquet"))
            self.others[name] = (o["key"].to_numpy(np.uint64), encode(o["src"].to_numpy(), o["row"].to_numpy()))
            s = pd.read_parquet(os.path.join(idx, f"{name}__s1.parquet")).sort_values("row", kind="stable")
            self.s1[name] = (s["row"].to_numpy(np.int64), s["key"].to_numpy(np.uint64))
        self.records = records or Records(processed, split)
        print(f"  blocking indexes loaded for {split} ({len(self.keys)} keys) in {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ stage 1 + 2
    def key_candidates(self, s1_rows) -> pd.DataFrame:
        """s1_row, cand, n_keys for the given S1 rows (top pre_k by n_keys)."""
        s1_rows = np.unique(np.asarray(s1_rows, dtype=np.int64))
        lo_row, hi_row = s1_rows.min(), s1_rows.max()
        parts = []
        wanted = pd.Index(s1_rows)
        for name in self.keys:
            rows, keys = self.s1[name]
            a, b = np.searchsorted(rows, [lo_row, hi_row + 1])
            r, k = rows[a:b], keys[a:b]
            m = wanted.get_indexer(r) >= 0
            r, k = r[m], k[m]
            order = np.argsort(k, kind="stable")
            sr, cand = _expand(r[order], k[order], *self.others[name])
            parts.append((sr << PAIR_SHIFT) | cand)
        if not parts or sum(len(p) for p in parts) == 0:
            return pd.DataFrame({"s1_row": np.empty(0, np.int64), "cand": np.empty(0, np.int64),
                                 "n_keys": np.empty(0, np.int16)})
        combined, counts = np.unique(np.concatenate(parts), return_counts=True)
        df = pd.DataFrame({"s1_row": combined >> PAIR_SHIFT, "cand": combined & ((1 << PAIR_SHIFT) - 1),
                           "n_keys": counts.astype(np.int16)})
        df = df.sort_values(["s1_row", "n_keys"], ascending=[True, False], kind="stable")
        df = df[df.groupby("s1_row", sort=False).cumcount() < self.pre_k]
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ stage 3
    def generate(self, s1_rows) -> pd.DataFrame:
        """Final candidates: s1_row, cand, n_keys, pre_score (top_k per S1)."""
        from src.features.pair_features import fast_similarity
        df = self.key_candidates(s1_rows)
        if df.empty:
            df["pre_score"] = np.empty(0, np.float32)
            return df
        s1 = self.records.take_s1(df["s1_row"].to_numpy(), ["name_core", "addr_core"])
        ot = self.records.take_cand(df["cand"].to_numpy(), ["name_core", "addr_core"])
        name_s, addr_s = fast_similarity(s1["name_core"], ot["name_core"], s1["addr_core"], ot["addr_core"])
        df["pre_score"] = (np.maximum(name_s, addr_s) + 0.5 * np.minimum(name_s, addr_s)
                           + 0.15 * np.minimum(df["n_keys"].to_numpy(), 4)).astype(np.float32)
        df = df.sort_values(["s1_row", "pre_score"], ascending=[True, False], kind="stable")
        df = df[df.groupby("s1_row", sort=False).cumcount() < self.top_k]
        return df.reset_index(drop=True)
