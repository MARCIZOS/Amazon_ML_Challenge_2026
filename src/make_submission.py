"""Build output/matching_results.tsv from the stored test scores (seconds).

  python -m src.make_submission --variant 1        # v1_best (see models/decision.json)
  python -m src.make_submission --variant 3
  python -m src.make_submission --threshold 0.6 --relative 0.5

Every run: writes output/matching_results.tsv (+ a copy in
data/interim/submissions/), checks it against output/candidate_pairs.tsv and
the test S1 ids, runs the official validator, and logs the variant in
data/interim/submissions/log.csv.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil

import pandas as pd

from src.config import load_config
from src.io.output_writer import (MATCH_HEADER, check_submission, run_official_validator,
                                  write_id_lists)
from src.matching.aggregation import decide_pairs, pairs_to_lists
from src.predict import SCORED_PATH


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join("configs", "pipeline.yaml"))
    ap.add_argument("--variant", type=int, default=1, help="1..5 from models/decision.json")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--relative", type=float, default=None)
    ap.add_argument("--no-validator", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    P = cfg["paths"]
    with open(os.path.join(P["models_dir"], "decision.json")) as f:
        decision = json.load(f)
    if a.threshold is not None:
        v = {"name": "custom", "threshold": a.threshold,
             "relative": a.relative if a.relative is not None else decision["best"]["relative"],
             "valid_macro_f05": None}
    else:
        v = decision["variants"][a.variant - 1]
    print(f"variant {v['name']}: threshold={v['threshold']} relative={v['relative']} "
          f"(validation macro F0.5 = {v['valid_macro_f05']})")

    scored = pd.read_parquet(SCORED_PATH)
    kept = decide_pairs(scored, v["threshold"], v["relative"], decision["one_to_one"])
    s1_ids = pd.read_parquet(os.path.join(P["processed_dir"], "test", "source1.parquet"),
                             columns=["entity_id"])["entity_id"].tolist()
    lists = pairs_to_lists(kept, s1_ids)
    out = os.path.join(P["output_dir"], "matching_results.tsv")
    write_id_lists(out, MATCH_HEADER, s1_ids, lists)
    n_empty = sum(1 for s in s1_ids if not lists[s])
    print(f"wrote {out}: {len(s1_ids):,} S1 rows, {len(kept):,} matches, "
          f"{n_empty:,} predicted singletons ({n_empty / len(s1_ids):.1%})")

    cand = os.path.join(P["output_dir"], "candidate_pairs.tsv")
    problems = check_submission(out, cand if os.path.exists(cand) else None, s1_ids)
    print("own checks:", "PASS" if not problems else problems)
    code = -1
    if not a.no_validator:
        code = run_official_validator(P.get("validator"), out, cand if os.path.exists(cand) else None,
                                      os.path.join(P["dataset_dir"], "test"))
    sub_dir = os.path.join("data", "interim", "submissions")
    os.makedirs(sub_dir, exist_ok=True)
    tag = f"{v['name']}_t{v['threshold']}_r{v['relative']}"
    shutil.copy(out, os.path.join(sub_dir, f"matching_results_{tag}.tsv"))
    log = os.path.join(sub_dir, "log.csv")
    new = not os.path.exists(log)
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "variant", "threshold", "relative", "valid_macro_f05", "matches",
                        "pred_singletons", "own_checks", "validator_exit", "model_exp_id"])
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), v["name"], v["threshold"], v["relative"],
                    v["valid_macro_f05"], len(kept), n_empty, "PASS" if not problems else "FAIL", code,
                    decision["exp_id"]])
    if problems or code not in (0, -1):
        raise SystemExit("submission NOT safe to upload - see messages above")
    print("ready to upload: output/matching_results.tsv")


if __name__ == "__main__":
    main()
