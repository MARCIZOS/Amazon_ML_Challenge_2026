"""Score every TEST Source 1 entity once and write output/candidate_pairs.tsv.

  python -m src.predict                     # uses configs/pipeline.yaml + models/

Writes
  output/candidate_pairs.tsv               exactly the pairs the model scored (top_k per S1)
  data/interim/scored_test.parquet         s1_id, other_id, p   (p >= min_keep_prob)
Then run  python -m src.make_submission --variant 1  (seconds) to create
output/matching_results.tsv from the stored scores.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src.blocking.candidates import CandidateGenerator
from src.config import load_config
from src.io.output_writer import CAND_HEADER, StreamingIdListWriter
from src.matching.model import LGBMMatcher
from src.scoring import build_pairs, chunks

SCORED_PATH = os.path.join("data", "interim", "scored_test.parquet")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join("configs", "pipeline.yaml"))
    ap.add_argument("--split", default="test", help="test (default) or train for a dry run")
    ap.add_argument("--limit", type=int, default=None, help="only the first N S1 rows (smoke test)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    P, B = cfg["paths"], cfg["blocking"]
    with open(os.path.join(P["models_dir"], "decision.json")) as f:
        decision = json.load(f)
    if decision["blocking"] != B:
        print("WARNING: blocking config differs from the one used in training:", decision["blocking"])
    model = LGBMMatcher.load(P["models_dir"])
    t0 = time.time()
    gen = CandidateGenerator(P["processed_dir"], a.split, B["keys"], B["pre_k"], B["top_k"])
    s1_ids = gen.records.s1_ids()
    rows = np.arange(len(s1_ids)) if a.limit is None else np.arange(min(a.limit, len(s1_ids)))

    cand_path = os.path.join(P["output_dir"], "candidate_pairs.tsv")
    writer = StreamingIdListWriter(cand_path, CAND_HEADER)
    scored_parts, n_pairs = [], 0
    for part in chunks(rows, B["s1_chunk"]):
        meta, X = build_pairs(gen, part)
        lists = {}
        if X is not None:
            p = model.predict(X)
            meta["p"] = p.astype(np.float32)
            lists = meta.groupby("s1_id", sort=False)["other_id"].agg(list).to_dict()
            keep = meta[meta["p"] >= decision["min_keep_prob"]][["s1_id", "other_id", "p"]]
            scored_parts.append(keep)
            n_pairs += len(meta)
        writer.write(s1_ids[part], lists)
        print(f"  {part[-1] + 1:,}/{len(rows):,} S1 scored, {n_pairs:,} candidate pairs "
              f"({time.time() - t0:.0f}s)", flush=True)
    writer.close()
    scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else \
        pd.DataFrame(columns=["s1_id", "other_id", "p"])
    os.makedirs(os.path.dirname(SCORED_PATH), exist_ok=True)
    scored.to_parquet(SCORED_PATH if a.split == "test" else SCORED_PATH.replace("test", a.split), index=False)
    meta_out = {"split": a.split, "n_s1": int(len(rows)), "candidate_pairs": int(n_pairs),
                "scored_kept": int(len(scored)), "seconds": round(time.time() - t0),
                "model_exp_id": decision["exp_id"]}
    with open(os.path.join("data", "interim", f"predict_{a.split}.json"), "w") as f:
        json.dump(meta_out, f, indent=1)
    print(json.dumps(meta_out, indent=1))
    print(f"wrote {cand_path}; next: python -m src.make_submission --variant 1")


if __name__ == "__main__":
    main()
