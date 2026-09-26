"""Train + validate the matcher on the TRAIN split (no test labels are used).

  python -m src.train                      # uses configs/pipeline.yaml
  python -m src.train --n-train 60000 --n-valid 20000     # quicker run

Steps
  1. sample disjoint S1 sets from train: fit (+10% for early stopping) and valid
  2. blocking -> candidates (against ALL train S2/S3 records, i.e. real distractors)
  3. pair features; label = pair is in train_ground_truth
  4. LightGBM fit, early stopping on the early-stop S1 set
  5. score valid candidates; tune (threshold, relative) for macro F0.5 with
     one-to-one resolution; choose 5 submission variants
  6. write models/ (model + decision.json) and experiments/results/<exp_id>/
     (report.md, report.json, tuning.csv, feature_importance.csv, errors.tsv)
     and append a row to experiments/experiment_registry.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd

from src.blocking.candidates import CandidateGenerator
from src.config import load_config
from src.evaluation.metrics import blocking_report, evaluate
from src.matching.aggregation import decide_pairs, tune_decision
from src.matching.model import LGBMMatcher
from src.scoring import build_pairs, chunks


def true_pairs_from_gt(processed) -> pd.DataFrame:
    gt = pd.read_parquet(os.path.join(processed, "train", "ground_truth.parquet"))
    ids = gt["matched_entity_ids"].str.split(",")
    long = pd.DataFrame({"s1_id": np.repeat(gt["source1_entity_id"].to_numpy(), ids.str.len()),
                         "other_id": np.concatenate(ids.to_numpy())})
    return long[long["other_id"] != ""].reset_index(drop=True)


def collect(gen, rows, s1_chunk, true_pairs, label=True, tag=""):
    metas, Xs = [], []
    t0 = time.time()
    done = 0
    for part in chunks(rows, s1_chunk):
        meta, X = build_pairs(gen, part)
        done += len(part)
        if X is not None:
            metas.append(meta)
            Xs.append(X)
        print(f"  [{tag}] {done:,}/{len(rows):,} S1 -> {sum(len(m) for m in metas):,} pairs "
              f"({time.time() - t0:.0f}s)", flush=True)
    meta = pd.concat(metas, ignore_index=True)
    X = pd.concat(Xs, ignore_index=True)
    if label:
        key = true_pairs.assign(_y=1)
        meta = meta.merge(key, on=["s1_id", "other_id"], how="left")
        meta["_y"] = meta["_y"].fillna(0).astype(np.int8)
    return meta, X


def pick_variants(table: pd.DataFrame, best: dict) -> list[dict]:
    """Five DISTINCT submission variants around the validation optimum:
    best, stricter, looser, other relative rule, most precise; any collision is
    replaced by the next-best distinct grid point."""
    t0, r0 = float(best["threshold"]), float(best["relative"])
    grid_t = sorted(float(x) for x in table.threshold.unique())
    i = grid_t.index(t0)
    wish = [("v1_best", t0, r0),
            ("v2_stricter", grid_t[min(i + 1, len(grid_t) - 1)], r0),
            ("v3_looser", grid_t[max(i - 1, 0)], r0),
            ("v4_rel", t0, 0.5 if r0 != 0.5 else 0.0),
            ("v5_precision", grid_t[min(i + 2, len(grid_t) - 1)], max(r0, 0.5))]
    ranked = [(float(r.threshold), float(r.relative)) for r in table.itertuples()]
    seen, out = set(), []
    for name, t, r in wish:
        if (t, r) in seen:
            t, r = next(x for x in ranked if x not in seen)
            name += "_alt"
        seen.add((t, r))
        row = table[np.isclose(table.threshold, t) & np.isclose(table.relative, r)].iloc[0]
        out.append({"name": name, "threshold": t, "relative": r,
                    "valid_macro_f05": float(row["macro_f05"]),
                    "valid_precision": float(row["micro_precision"]), "valid_recall": float(row["micro_recall"])})
    return out


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join("configs", "pipeline.yaml"))
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-valid", type=int, default=None)
    ap.add_argument("--exp-id", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    P, B, T, D = cfg["paths"], cfg["blocking"], cfg["train"], cfg["decision"]
    n_train = a.n_train or T["n_train_s1"]
    n_valid = a.n_valid or T["n_valid_s1"]
    exp_id = a.exp_id or "M" + dt.datetime.now().strftime("%Y%m%d_%H%M")
    res_dir = os.path.join(P["results_dir"], exp_id)
    os.makedirs(res_dir, exist_ok=True)
    t_all = time.time()

    print("loading train records + blocking indexes ...", flush=True)
    gen = CandidateGenerator(P["processed_dir"], "train", B["keys"], B["pre_k"], B["top_k"])
    s1_ids = gen.records.s1_ids()
    n_other = gen.records.n[2] + gen.records.n[3]
    true_pairs = true_pairs_from_gt(P["processed_dir"])

    rng = np.random.default_rng(T["seed"])
    perm = rng.permutation(len(s1_ids))
    n_es = max(1000, n_train // 10)
    rows_fit = perm[:n_train]
    rows_es = perm[n_train:n_train + n_es]
    rows_val = perm[n_train + n_es:n_train + n_es + n_valid]

    meta_fit, X_fit = collect(gen, rows_fit, B["s1_chunk"], true_pairs, tag="fit")
    meta_es, X_es = collect(gen, rows_es, B["s1_chunk"], true_pairs, tag="early-stop")
    meta_val, X_val = collect(gen, rows_val, B["s1_chunk"], true_pairs, tag="valid")

    val_ids = s1_ids[np.sort(rows_val)]
    blk = blocking_report(meta_val, true_pairs, val_ids, n_other)
    print("blocking (valid):", json.dumps(blk), flush=True)

    M = cfg["model"]
    model = LGBMMatcher(M["params"], M["num_boost_round"], M["early_stopping_rounds"])
    print(f"training LightGBM on {len(X_fit):,} pairs ({int(meta_fit._y.sum()):,} positive) ...", flush=True)
    model.fit(X_fit, meta_fit["_y"], X_es, meta_es["_y"])

    meta_val["p"] = model.predict(X_val)
    scored = meta_val[meta_val["p"] >= D["min_keep_prob"]][["s1_id", "other_id", "p", "s1_row", "cand"]]
    table, best = tune_decision(scored, true_pairs, val_ids, D["threshold_grid"], D["relative_grid"],
                                D.get("one_to_one", True))
    variants = pick_variants(table, best)
    print("best decision on valid:", {k: best[k] for k in ("threshold", "relative", "macro_f05",
                                                         "micro_precision", "micro_recall")}, flush=True)

    # --------------------------------------------------------------- save model + decision
    model.save(P["models_dir"])
    decision = {"exp_id": exp_id, "one_to_one": D.get("one_to_one", True), "min_keep_prob": D["min_keep_prob"],
                "best": {"threshold": best["threshold"], "relative": best["relative"]},
                "variants": variants, "blocking": B}
    with open(os.path.join(P["models_dir"], "decision.json"), "w") as f:
        json.dump(decision, f, indent=1)

    # --------------------------------------------------------------- report
    table.to_csv(os.path.join(res_dir, "tuning.csv"), index=False)
    model.feature_importance().to_csv(os.path.join(res_dir, "feature_importance.csv"), index=False)
    kept = decide_pairs(scored, best["threshold"], best["relative"], D.get("one_to_one", True))
    ev = evaluate(kept, true_pairs, val_ids)
    # model-only ceiling: what if every true pair in the candidates were found
    ceil = evaluate(meta_val[meta_val._y == 1][["s1_id", "other_id"]], true_pairs, val_ids)
    errors = []
    kset = kept.assign(_k=1)
    m2 = meta_val.merge(kset[["s1_id", "other_id", "_k"]], on=["s1_id", "other_id"], how="left")
    fp = m2[(m2._k == 1) & (m2._y == 0)].head(40)
    fn = m2[(m2._k != 1) & (m2._y == 1)].sort_values("p", ascending=False).head(40)
    rec = gen.records
    for kind, part in (("false_merge", fp), ("missed_match", fn)):
        if part.empty:
            continue
        s1 = rec.take_s1(part.s1_row.to_numpy(), ["name_core", "addr_core"])
        ot = rec.take_cand(part.cand.to_numpy(), ["name_core", "addr_core"])
        for i in range(len(part)):
            errors.append({"type": kind, "p": round(float(part.p.iloc[i]), 3),
                           "s1_id": part.s1_id.iloc[i], "s1_name": s1["name_core"][i], "s1_addr": s1["addr_core"][i],
                           "other_id": part.other_id.iloc[i], "other_name": ot["name_core"][i],
                           "other_addr": ot["addr_core"][i]})
    pd.DataFrame(errors).to_csv(os.path.join(res_dir, "errors.tsv"), sep="\t", index=False)

    runtime = time.time() - t_all
    try:
        import psutil
        peak_mb = psutil.Process().memory_info().peak_wset / 2**20 if hasattr(
            psutil.Process().memory_info(), "peak_wset") else psutil.Process().memory_info().rss / 2**20
    except Exception:
        peak_mb = None
    report = {"exp_id": exp_id, "date": dt.datetime.now().isoformat(timespec="seconds"),
              "git_commit": git_commit(), "config": cfg, "n_fit_s1": int(len(rows_fit)),
              "n_valid_s1": int(len(rows_val)), "fit_pairs": int(len(X_fit)),
              "fit_positive": int(meta_fit._y.sum()), "blocking_valid": blk, "valid_at_best": ev,
              "valid_ceiling_if_perfect_matcher": ceil, "best_decision": {"threshold": best["threshold"],
              "relative": best["relative"]}, "variants": variants, "runtime_seconds": round(runtime),
              "peak_memory_mb": peak_mb, "best_iteration": model.booster.best_iteration}
    with open(os.path.join(res_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)

    md = [f"# Experiment {exp_id}", "",
          f"- Date: {report['date']}  |  commit: `{report['git_commit']}`  |  runtime: {runtime / 60:.1f} min",
          f"- Train S1 used to fit: {len(rows_fit):,} ({len(X_fit):,} candidate pairs, "
          f"{int(meta_fit._y.sum()):,} positive); early-stop S1: {len(rows_es):,}; validation S1: {len(rows_val):,}",
          f"- Blocking keys: {B['keys']}; pre_k={B['pre_k']}, top_k={B['top_k']}", "",
          "## Blocking (validation S1)", "", "| metric | value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in blk.items()]
    md += ["", "## Matching (validation S1, best decision)", "",
           f"threshold = {best['threshold']}, relative = {best['relative']}, one-to-one = {D.get('one_to_one', True)}",
           "", "| metric | value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in ev.items()]
    md += ["", f"Ceiling with a perfect matcher on these candidates: macro F0.5 = {ceil['macro_f05']:.4f}", "",
           "## Submission variants", "", "| variant | threshold | relative | valid macro F0.5 |", "|---|---|---|---|"]
    md += [f"| {v['name']} | {v['threshold']} | {v['relative']} | {v['valid_macro_f05']} |" for v in variants]
    top = table.head(10)[["threshold", "relative", "macro_f05", "micro_precision", "micro_recall",
                          "false_merges_on_singletons"]]
    md += ["", "## Top of the tuning grid", "", "| " + " | ".join(top.columns) + " |",
           "|" + "---|" * len(top.columns)]
    md += ["| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |"
           for r in top.itertuples(index=False)]
    with open(os.path.join(res_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    reg = os.path.join("experiments", "experiment_registry.csv")
    if os.path.exists(reg):
        with open(reg, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([exp_id, report["date"], "Member 1/4 (pipeline)", report["git_commit"],
                                    "v2", "+".join(B["keys"]) + f" top{B['top_k']}", "pair_features v1",
                                    "lightgbm", f"t={best['threshold']} rel={best['relative']}",
                                    blk["candidate_pairs"], round(blk["candidates_per_s1_mean"], 1),
                                    blk["candidates_per_s1_max"], round(blk["blocking_recall"], 4),
                                    round(blk["reduction_ratio"] or 0, 6), round(runtime),
                                    round(peak_mb) if peak_mb else "",
                                    f"valid macro F0.5={ev['macro_f05']:.4f} P={ev['micro_precision']:.4f} "
                                    f"R={ev['micro_recall']:.4f}", "candidate",
                                    f"{len(rows_val):,} valid S1; see {res_dir}/report.md"])
    print("\n".join(md))
    print(f"\nsaved model to {P['models_dir']}/, report to {res_dir}/")


if __name__ == "__main__":
    main()
