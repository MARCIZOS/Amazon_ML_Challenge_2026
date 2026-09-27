"""Reusable preprocessing pipeline: TSV -> normalised Parquet.

For every source file it streams the TSV in chunks (constant memory), normalises
the chunk on all CPU cores, and appends it to one Parquet file per source.
Original columns are kept unchanged; the normalised fields are added next to
them (see docs/DATA_DICTIONARY.md).

Usage (from project root):
    python -m src.preprocessing.preprocess --data-dir <DATASET_DIR> --out-dir data/processed
    python -m src.preprocessing.preprocess --data-dir <DATASET_DIR> --splits train --sources 1 --limit 100000

Outputs
    <out-dir>/<split>/source{1,2,3}.parquet
    <out-dir>/train/ground_truth.parquet   (source1_entity_id, matched_entity_ids)
    <out-dir>/manifest.json                 row counts, timings, settings
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context

import pandas as pd

from src.io.data_loader import ParquetAppender, gt_path, iter_tsv, read_tsv, source_path
from src.preprocessing.normalize import DEFAULT_VOCAB, OUTPUT_FIELDS, Normalizer

_NZ = None  # per-process normaliser (set by _init)


def _init(vocab_path):
    global _NZ
    _NZ = Normalizer(vocab_path=vocab_path)


def _norm_rows(rows):
    return [_NZ.record(n, a, c) for n, a, c in rows]


def _split(seq, n):
    k = max(1, len(seq) // n + (len(seq) % n > 0))
    return [seq[i:i + k] for i in range(0, len(seq), k)]


def normalize_df_parallel(df: pd.DataFrame, workers: int = 1, pool=None,
                          vocab_path: str | None = DEFAULT_VOCAB) -> pd.DataFrame:
    """Normalise a DataFrame, optionally with a process pool. Originals kept."""
    rows = list(zip(df["business_name"], df["business_address"], df["country"]))
    if workers <= 1 and pool is None:
        _init(vocab_path)
        recs = _norm_rows(rows)
    else:
        own = pool is None
        if own:
            pool = get_context("spawn").Pool(workers, initializer=_init, initargs=(vocab_path,))
        try:
            parts = pool.map(_norm_rows, _split(rows, (workers or 1) * 4))
        finally:
            if own:
                pool.close()
                pool.join()
        recs = [r for p in parts for r in p]
    extra = pd.DataFrame(recs, index=df.index, columns=OUTPUT_FIELDS)
    extra["name_translit"] = extra["name_translit"].astype("int8")
    extra["addr_missing"] = extra["addr_missing"].astype("int8")
    return pd.concat([df, extra], axis=1)


def run(data_dir, out_dir, splits=("train", "test"), sources=(1, 2, 3), workers=None,
        chunksize=250_000, vocab_path=DEFAULT_VOCAB, limit=None):
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if vocab_path and not os.path.exists(vocab_path):
        print(f"WARNING: vocab {vocab_path} not found - native-script words will only be "
              f"transliterated. Run src.preprocessing.translit_vocab first for best results.")
        vocab_path = ""
    manifest = {"settings": {"workers": workers, "chunksize": chunksize, "vocab": vocab_path,
                             "fields": OUTPUT_FIELDS}, "files": {}}
    ctx = get_context("spawn")
    t_all = time.time()
    with ctx.Pool(workers, initializer=_init, initargs=(vocab_path,)) as pool:
        for split in splits:
            for k in sources:
                src = source_path(data_dir, split, k)
                if not os.path.exists(src):
                    print("missing, skipped:", src)
                    continue
                dst = os.path.join(out_dir, split, f"source{k}.parquet")
                t0, n = time.time(), 0
                with ParquetAppender(dst) as w:
                    for chunk in iter_tsv(src, chunksize=chunksize):
                        if limit is not None:
                            chunk = chunk.iloc[: max(0, limit - n)]
                            if chunk.empty:
                                break
                        out = normalize_df_parallel(chunk, workers=workers, pool=pool)
                        w.write(out)
                        n += len(chunk)
                        rate = n / (time.time() - t0)
                        print(f"  {split}/source{k}: {n:,} rows ({rate:,.0f} rows/s)", flush=True)
                manifest["files"][f"{split}/source{k}"] = {
                    "rows": n, "seconds": round(time.time() - t0, 1), "path": dst,
                    "mb": round(os.path.getsize(dst) / 2**20, 1)}
            if split == "train" and os.path.exists(gt_path(data_dir)):
                gt = read_tsv(gt_path(data_dir))
                dst = os.path.join(out_dir, "train", "ground_truth.parquet")
                gt.to_parquet(dst, index=False)
                manifest["files"]["train/ground_truth"] = {"rows": len(gt), "path": dst}
    manifest["total_seconds"] = round(time.time() - t_all, 1)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default=os.path.join("data", "processed"))
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--sources", default="1,2,3")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--chunksize", type=int, default=250_000)
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    ap.add_argument("--limit", type=int, default=None, help="max rows per file (for quick tests)")
    a = ap.parse_args()
    run(a.data_dir, a.out_dir, tuple(a.splits.split(",")), tuple(int(x) for x in a.sources.split(",")),
        a.workers, a.chunksize, a.vocab, a.limit)


if __name__ == "__main__":
    main()
