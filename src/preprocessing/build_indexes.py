"""Compact on-disk blocking indexes (Member 2 -> Member 4 hand-off).

For every split and every requested key (definitions in keys.py) this writes
two small integer-only Parquet files next to the processed data:

  data/processed/<split>/index/<key>__others.parquet   key (uint64), src (int8: 2|3), row (int32)
  data/processed/<split>/index/<key>__s1.parquet       key (uint64), row (int32)

`row` is the row position in data/processed/<split>/source{src}.parquet, so an
entity id is recovered with one array lookup. Key values whose block over
S2+S3 is larger than --max-block are OVERSIZED and dropped (counted in
index/stats.json); S1 rows keep only keys that exist in the S2+S3 index.
Keys are stable 64-bit hashes of 'country#key', so nothing crosses countries.

This is a data structure, not a blocking policy: Member 4 decides which keys
to union and how to cap candidates (see docs/BLOCKING_KEYS.md for recall/cost).

Usage (after src.preprocessing.preprocess):
    python -m src.preprocessing.build_indexes --processed data/processed
    python -m src.preprocessing.build_indexes --processed data/processed --splits test --keys num_street,name_skel_pair
    python -m src.preprocessing.build_indexes --config configs/pipeline.yaml      # per-key limits

Reading candidates back:
    from src.preprocessing.build_indexes import candidates
    cand = candidates("data/processed", "train", ["num_street", "name_skel_pair"])
    # -> DataFrame s1_entity_id, other_entity_id, n_keys  (one row per distinct pair)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src.preprocessing.keys import KEY_COLUMNS, KEYS, explode_keys

_POOL = None


def _explode_part(args):
    name, df, offset = args
    e = explode_keys(df, KEYS[name])
    e["rid"] += offset
    return e


def explode_parallel(df, name, workers):
    """explode_keys over row blocks on several processes (same result, row order kept)."""
    global _POOL
    if workers <= 1 or len(df) < 200_000:
        return explode_keys(df, KEYS[name])
    if _POOL is None:
        from multiprocessing import get_context
        _POOL = get_context("spawn").Pool(workers)
    step = -(-len(df) // (workers * 4))
    parts = [(name, df.iloc[i:i + step].reset_index(drop=True), i) for i in range(0, len(df), step)]
    return pd.concat(_POOL.map(_explode_part, parts), ignore_index=True)


def _read(processed, split, k, cols):
    return pd.read_parquet(os.path.join(processed, split, f"source{k}.parquet"),
                           columns=["entity_id", "country"] + cols)


def build(processed, splits=("train", "test"), keys=None, max_block=1000, workers=None):
    """keys: list of key names (all use max_block) or dict name -> max_block."""
    keys = keys or list(KEYS)
    limits = dict(keys) if isinstance(keys, dict) else {k: max_block for k in keys}
    keys = list(limits)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    stats = {}
    for split in splits:
        if not os.path.exists(os.path.join(processed, split, "source1.parquet")):
            print("missing split, skipped:", split)
            continue
        out_dir = os.path.join(processed, split, "index")
        os.makedirs(out_dir, exist_ok=True)
        stats[split] = {}
        for name in keys:
            t0 = time.time()
            fn, cols = KEYS[name], KEY_COLUMNS[name]
            max_block = limits[name]
            parts = []
            for k in (2, 3):
                df = _read(processed, split, k, cols)
                e = explode_parallel(df, name, workers)
                e["src"] = np.int8(k)
                parts.append(e)
                del df
            others = pd.concat(parts, ignore_index=True).rename(columns={"rid": "row"})
            del parts
            sizes = others["key"].value_counts()
            oversized = sizes[sizes > max_block]
            keep = others["key"].isin(sizes.index[sizes <= max_block])
            others = others[keep].sort_values("key", kind="stable")[["key", "src", "row"]]
            others.to_parquet(os.path.join(out_dir, f"{name}__others.parquet"), index=False)

            s1 = explode_parallel(_read(processed, split, 1, cols), name, workers).rename(columns={"rid": "row"})
            s1 = s1[s1["key"].isin(set(others["key"].unique()))]
            s1 = s1.sort_values("key", kind="stable")[["key", "row"]]
            s1.to_parquet(os.path.join(out_dir, f"{name}__s1.parquet"), index=False)

            st = {"others_rows": int(len(others)), "s1_rows": int(len(s1)),
                  "distinct_keys": int(len(sizes)), "oversized_keys": int(len(oversized)),
                  "records_in_oversized_blocks": int(oversized.sum()), "max_block": max_block,
                  "seconds": round(time.time() - t0, 1)}
            stats[split][name] = st
            print(f"  {split}/{name}: {st['others_rows']:,} S2/S3 key rows, {st['s1_rows']:,} S1 key rows, "
                  f"{st['oversized_keys']:,} oversized keys dropped ({st['seconds']}s)", flush=True)
            del others, s1
        with open(os.path.join(out_dir, "stats.json"), "w") as f:
            json.dump(stats[split], f, indent=1)
    return stats


def candidates(processed, split, keys, s1_rows=None) -> pd.DataFrame:
    """Union of candidate pairs over `keys` from the on-disk indexes.

    s1_rows: optional iterable of S1 row positions to restrict to (process S1 in
    chunks to bound memory). Returns s1_entity_id, other_entity_id, n_keys.
    """
    idx = os.path.join(processed, split, "index")
    pieces = []
    for name in keys:
        s1 = pd.read_parquet(os.path.join(idx, f"{name}__s1.parquet"))
        if s1_rows is not None:
            s1 = s1[s1["row"].isin(set(s1_rows))]
        oth = pd.read_parquet(os.path.join(idx, f"{name}__others.parquet"))
        oth = oth[oth["key"].isin(set(s1["key"].unique()))]
        m = s1.merge(oth, on="key", suffixes=("_s1", "_o"))[["row_s1", "src", "row_o"]]
        pieces.append(m.drop_duplicates())
    if not pieces:
        return pd.DataFrame(columns=["s1_entity_id", "other_entity_id", "n_keys"])
    allp = pd.concat(pieces, ignore_index=True)
    allp = allp.groupby(["row_s1", "src", "row_o"], sort=False).size().rename("n_keys").reset_index()
    ids = {k: pd.read_parquet(os.path.join(processed, split, f"source{k}.parquet"),
                              columns=["entity_id"])["entity_id"].to_numpy() for k in (1, 2, 3)}
    other = np.where(allp["src"].to_numpy() == 2,
                     ids[2][np.minimum(allp["row_o"].to_numpy(), len(ids[2]) - 1)],
                     ids[3][np.minimum(allp["row_o"].to_numpy(), len(ids[3]) - 1)])
    return pd.DataFrame({"s1_entity_id": ids[1][allp["row_s1"].to_numpy()],
                         "other_entity_id": other, "n_keys": allp["n_keys"].to_numpy()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", default=os.path.join("data", "processed"))
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--keys", default="all", help="comma list from keys.KEYS or 'all'")
    ap.add_argument("--max-block", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--config", default=None,
                    help="YAML with blocking.keys {name: max_block}; overrides --keys/--max-block")
    a = ap.parse_args()
    keys = list(KEYS) if a.keys == "all" else [k.strip() for k in a.keys.split(",")]
    if a.config:
        from src.config import load_config
        cfg = load_config(a.config)
        keys = dict(cfg["blocking"]["keys"])
        if a.processed == os.path.join("data", "processed"):
            a.processed = cfg["paths"]["processed_dir"]
    unknown = [k for k in keys if k not in KEYS]
    if unknown:
        ap.error(f"unknown keys {unknown}; choose from {list(KEYS)}")
    t0 = time.time()
    build(a.processed, tuple(a.splits.split(",")), keys, a.max_block, a.workers)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
