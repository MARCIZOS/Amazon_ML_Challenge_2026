# End-to-end pipeline — run guide

All commands run from the repository root. Settings live in `configs/pipeline.yaml`
(paths, blocking keys, K, training sizes, model parameters, decision grid).

```
raw TSV ──preprocess──▶ processed Parquet ──build_indexes──▶ blocking indexes
        (Member 2)        (original + 14 fields)   (Member 2)
   ──train──▶ models/lgbm_matcher.* + models/decision.json + experiments/results/<id>/report.md
   ──predict──▶ output/candidate_pairs.tsv + data/interim/scored_test.parquet
   ──make_submission --variant N──▶ output/matching_results.tsv  (+ official validator)
```

## Commands

| step | command | laptop estimate* |
|---|---|---|
| 0 | `py -3.12 -m pip install -r requirements.txt` | 1 min |
| 1 | `py -3.12 -m src.preprocessing.preprocess --data-dir <DATASET_DIR>` (**already done** on this laptop) | 19 min |
| 2 | `py -3.12 -m src.preprocessing.build_indexes --config configs/pipeline.yaml` | 5–15 min |
| 3 | `py -3.12 -m src.train` | 15–30 min |
| 4 | `py -3.12 -m src.predict` | 20–40 min |
| 5 | `py -3.12 -m src.make_submission --variant 1` … `--variant 5` | seconds each |

Or steps 2–5 in one go: `py -3.12 -m src.pipeline` (then variants 2–5 with step 5).

*Estimates are extrapolated from sandbox benchmarks (features ≈ 75k pairs/s on 2 CPUs,
index build ≈ 64k row-keys/s per core); the real numbers are printed by each step.

## What each step prints / writes

* **train** — blocking recall and reduction ratio on held-out validation S1, the tuned decision
  (threshold, relative), validation macro F0.5 / precision / recall, the ceiling macro F0.5 of a
  perfect matcher on the same candidates, and 5 submission variants. Writes
  `experiments/results/<exp_id>/{report.md, report.json, tuning.csv, feature_importance.csv, errors.tsv}`
  and appends a row to `experiments/experiment_registry.csv`.
* **predict** — scores every test S1 once; `output/candidate_pairs.tsv` is exactly the set the model
  scored (top_k per S1), so every final match is contained in it.
* **make_submission** — applies one-to-one resolution + thresholds, writes
  `output/matching_results.tsv`, a copy in `data/interim/submissions/`, runs our checks and the
  official `validate_submission.py`, and logs the variant in `data/interim/submissions/log.csv`.
  It refuses (non-zero exit) if any check fails.

## The 5 leaderboard submissions

`models/decision.json` lists five distinct variants chosen on validation:
`v1_best`, `v2_stricter` (next higher threshold), `v3_looser`, `v4_rel` (other relative rule),
`v5_precision` (two steps stricter + relative 0.5). Upload `output/matching_results.tsv` after
each `make_submission --variant N` (or the copies in `data/interim/submissions/`), and record the
public score next to the variant in `data/interim/submissions/log.csv` / the experiment registry.

## Running training on another (stronger) machine

Copy to the other machine:
1. this repository (git clone `feat/data-engineering` or the folder), and
2. either the processed folder
   `student_resource\processed\` (≈3.2 GB: `train/`, `test/`, `manifest.json`) **or** the raw
   `dataset\` folder (then run step 1 there).

On that machine: edit `paths` in `configs/pipeline.yaml` (dataset_dir, processed_dir, validator),
run steps 0, 2, 3, 4, 5. Bring back `models/`, `output/` and `experiments/results/`.
More RAM/CPU → raise `train.n_train_s1`, `blocking.top_k`, `blocking.s1_chunk`.

## If memory runs out
Lower `blocking.s1_chunk` (e.g. 20000), `blocking.pre_k` (e.g. 60) and `train.n_train_s1`
(e.g. 80000) in `configs/pipeline.yaml`; nothing else changes.
