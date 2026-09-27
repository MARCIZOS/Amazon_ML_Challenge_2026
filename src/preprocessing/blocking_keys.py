"""Blocking-key analysis on the normalised TRAIN parquet files.

For each candidate blocking key it reports
  * block sizes over Source 2+3 (how many records share a key value) and how
    many key values are OVERSIZED (> --max-block; those are dropped as keys)
  * pair recall: share of true (S1, S2/S3) pairs that share >= 1 usable key
  * candidates per S1 entity (deduplicated) - the cost of the key
and then evaluates the UNION of keys greedily (best recall gain per candidate).
It also counts duplicate records (same country + name_core + addr_core).

All keys are prefixed with the country label, so nothing is ever compared
across countries (the EDA showed 100% of true pairs share the country label).

Usage (after src.preprocessing.preprocess):
    python -m src.preprocessing.blocking_keys --processed data/processed
    python -m src.preprocessing.blocking_keys --processed data/processed --sample-s1 20000 --max-block 500
Writes docs/BLOCKING_KEYS.md and docs/blocking_keys.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src.preprocessing.keys import COLS, KEYS, explode_keys, key_strings  # noqa: E402


# ----------------------------------------------------------------------------- analysis
def analyse(processed, sample_s1=20_000, max_block=1000, seed=0, other_frac=1.0):
    t0 = time.time()
    tr = os.path.join(processed, "train")
    s1 = pd.read_parquet(os.path.join(tr, "source1.parquet"), columns=COLS)
    others = pd.concat([pd.read_parquet(os.path.join(tr, f"source{k}.parquet"), columns=COLS)
                        for k in (2, 3)], ignore_index=True)
    if other_frac < 1.0:
        others = others.sample(frac=other_frac, random_state=seed)
    gt = pd.read_parquet(os.path.join(tr, "ground_truth.parquet"))
    print(f"loaded S1 {len(s1):,}, S2+S3 {len(others):,} in {time.time() - t0:.0f}s", flush=True)

    report = {"n_s1": len(s1), "n_other": len(others), "max_block": max_block,
              "sample_s1": sample_s1, "keys": {}}

    # duplicates
    dup = others.duplicated(["country", "name_core", "addr_core"], keep=False)
    report["duplicate_records"] = {
        "records_in_duplicate_groups": int(dup.sum()),
        "share": float(dup.mean()),
        "groups": int(others[dup].groupby(["country", "name_core", "addr_core"]).ngroups),
    }

    # sample of S1 entities (with their true pairs) for recall / candidate counts
    rng = np.random.default_rng(seed)
    samp = s1.iloc[rng.choice(len(s1), size=min(sample_s1, len(s1)), replace=False)]
    samp = samp.reset_index(drop=True)
    others = others.reset_index(drop=True)
    samp_ids = set(samp.entity_id)
    gt_s = gt[gt.source1_entity_id.isin(samp_ids)]
    pairs = [(a, b) for a, ids in zip(gt_s.source1_entity_id, gt_s.matched_entity_ids)
             for b in ids.split(",") if b]
    pairs = pd.DataFrame(pairs, columns=["s1", "other"])
    other_ids_in_data = set(others.entity_id)
    pairs = pairs[pairs.other.isin(other_ids_in_data)]
    n_pairs = len(pairs)
    print(f"sample: {len(samp):,} S1 entities, {n_pairs:,} true pairs", flush=True)

    # integer ids: S2/S3 records 0..n-1, sampled S1 records 0..m-1
    o_map = {e: i for i, e in enumerate(others.entity_id)}
    s_map = {e: i for i, e in enumerate(samp.entity_id)}
    pairs = pd.DataFrame({"s1": pairs.s1.map(s_map).astype(np.int32),
                          "other": pairs.other.map(o_map).astype(np.int32)})

    cand_sets = {}      # key name -> DataFrame(s1, other) candidate pairs for the sample
    for name, fn in KEYS.items():
        tk = time.time()
        ok = explode_keys(others, fn, o_map)
        sizes = ok.key.value_counts()
        usable = sizes[sizes <= max_block]
        sk = explode_keys(samp, fn, s_map)
        sk = sk[sk.key.isin(usable.index)]
        ok_u = ok[ok.key.isin(set(sk.key))]
        cand = sk.merge(ok_u, on="key", suffixes=("_s1", "_o"))[["rid_s1", "rid_o"]]
        cand = cand.drop_duplicates().rename(columns={"rid_s1": "s1", "rid_o": "other"})
        cand_sets[name] = cand
        hit = pairs.merge(cand, on=["s1", "other"], how="inner")
        per_s1 = cand.groupby("s1").size().reindex(range(len(samp)), fill_value=0)
        top = sizes.head(8)
        names = key_strings(others, fn, {int(k) for k in top.index})
        report["keys"][name] = {
            "distinct_keys": int(len(sizes)),
            "block_size_median": float(sizes.median()) if len(sizes) else 0,
            "block_size_p99": float(sizes.quantile(.99)) if len(sizes) else 0,
            "block_size_max": int(sizes.max()) if len(sizes) else 0,
            "oversized_keys": int((sizes > max_block).sum()),
            "records_in_oversized": int(sizes[sizes > max_block].sum()),
            "top_oversized": [[names.get(int(k), str(k)), int(v)] for k, v in top.items()],
            "other_records_without_key": float(1 - ok.rid.nunique() / len(others)),
            "pair_recall": float(len(hit) / max(1, n_pairs)),
            "cands_per_s1_mean": float(per_s1.mean()),
            "cands_per_s1_p95": float(per_s1.quantile(.95)),
            "seconds": round(time.time() - tk, 1),
        }
        r = report["keys"][name]
        print(f"  {name:18s} recall {r['pair_recall']:.3f}  cands/S1 {r['cands_per_s1_mean']:8.1f}  "
              f"oversized {r['oversized_keys']:,}  ({r['seconds']}s)", flush=True)
        del ok, sk, ok_u

    # greedy union: repeatedly add the key with best recall gain per extra candidate
    chosen, cur = [], pd.DataFrame({"s1": np.array([], np.int32), "other": np.array([], np.int32)})
    steps = []
    remaining = set(cand_sets)
    base_hits = 0
    while remaining:
        best = None
        for name in remaining:
            u = pd.concat([cur, cand_sets[name]]).drop_duplicates()
            hits = len(pairs.merge(u, on=["s1", "other"]))
            gain = hits - base_hits
            cost = len(u) - len(cur)
            score = gain / max(1, cost) if gain > 0 else -1
            if best is None or score > best[0]:
                best = (score, name, u, hits)
        if best[0] <= 0:
            break
        _, name, cur, base_hits = best
        remaining.discard(name)
        chosen.append(name)
        steps.append({"added": name, "union_recall": base_hits / max(1, n_pairs),
                      "cands_per_s1_mean": len(cur) / max(1, len(samp))})
        print(f"  + {name:18s} union recall {steps[-1]['union_recall']:.4f}  "
              f"cands/S1 {steps[-1]['cands_per_s1_mean']:.1f}", flush=True)
    report["greedy_union"] = steps

    # which true pairs are still missed by the full union? (examples for error analysis)
    missed = pairs.merge(cur, on=["s1", "other"], how="left", indicator=True)
    missed = missed[missed["_merge"] == "left_only"].head(25)
    fmt = lambda r: f"{r.name_core} | {r.addr_core} | {r.addr_nums}"  # noqa: E731
    report["missed_examples"] = [{"s1": fmt(samp.iloc[a]), "other": fmt(others.iloc[b])}
                                 for a, b in zip(missed.s1, missed.other)]
    report["seconds"] = round(time.time() - t0, 1)
    return report


def to_markdown(rep) -> str:
    L = ["# Blocking-key analysis (train)", "",
         f"S1 records: {rep['n_s1']:,} · S2+S3 records: {rep['n_other']:,} · "
         f"S1 sample for recall: {rep['sample_s1']:,} · oversized = block > {rep['max_block']:,} records "
         "(oversized key values are dropped). All keys are prefixed by country.", "",
         "| key | pair recall | candidates / S1 (mean) | p95 | distinct keys | oversized keys | records in oversized | max block |",
         "|---|---|---|---|---|---|---|---|"]
    for k, r in sorted(rep["keys"].items(), key=lambda kv: -kv[1]["pair_recall"]):
        L.append(f"| {k} | {r['pair_recall']:.3f} | {r['cands_per_s1_mean']:.1f} | {r['cands_per_s1_p95']:.0f} | "
                 f"{r['distinct_keys']:,} | {r['oversized_keys']:,} | {r['records_in_oversized']:,} | {r['block_size_max']:,} |")
    L += ["", "## Greedy union (best recall gain per extra candidate)", "",
          "| step | key added | union recall | candidates / S1 |", "|---|---|---|---|"]
    for i, s in enumerate(rep["greedy_union"], 1):
        L.append(f"| {i} | {s['added']} | {s['union_recall']:.4f} | {s['cands_per_s1_mean']:.1f} |")
    L += ["", "## Largest key values (oversized candidates)", ""]
    for k, r in rep["keys"].items():
        L.append(f"- **{k}**: " + ", ".join(f"`{a}` ({b:,})" for a, b in r["top_oversized"][:6]))
    d = rep["duplicate_records"]
    L += ["", "## Duplicate records in S2+S3", "",
          f"{d['records_in_duplicate_groups']:,} records ({d['share']:.2%}) share identical "
          f"country + name_core + addr_core with another record ({d['groups']:,} groups). "
          "They are not errors - duplicates of one business should all be matched - but they "
          "can be scored once and the score copied.", "",
          "## True pairs missed by the full union (sample)", ""]
    for m in rep["missed_examples"]:
        L.append(f"- S1: {m['s1']}  \n  other: {m['other']}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", required=True, help="output folder of src.preprocessing.preprocess")
    ap.add_argument("--sample-s1", type=int, default=20_000)
    ap.add_argument("--max-block", type=int, default=1000)
    ap.add_argument("--other-frac", type=float, default=1.0, help="subsample S2+S3 if RAM is short")
    ap.add_argument("--out-dir", default="docs")
    a = ap.parse_args()
    rep = analyse(a.processed, a.sample_s1, a.max_block, other_frac=a.other_frac)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "blocking_keys.json"), "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(a.out_dir, "BLOCKING_KEYS.md"), "w", encoding="utf-8") as f:
        f.write(to_markdown(rep))
    print(f"done in {rep['seconds']}s -> {a.out_dir}/BLOCKING_KEYS.md")


if __name__ == "__main__":
    main()
