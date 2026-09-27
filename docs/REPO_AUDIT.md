# Repository Audit Report (Phase 0)

**Date:** 2026-09-27  
**Pipeline Run Audited:** M20260927_1454 (Public Leaderboard Score: 0.915344 macro F0.5)

---

## 1. File Inventory and Tracked / Generated Status

A complete inventory of the repository identified both active source files, pipeline configurations, documentation, and large generated / raw data artifacts:

| Category | Files / Directories | Size | Status / Recommendation |
|---|---|---|---|
| **Raw Datasets** | `dataset/train/*.tsv` (4 files: S1, S2, S3, GT)<br>`dataset/test/*.tsv` (3 files: S1, S2, S3) | ~2.48 GB total | Raw competition data. Excluded via updated `.gitignore` (`dataset/**/*.tsv`). |
| **Active Code** | `src/`<br>├── `blocking/candidates.py`<br>├── `features/pair_features.py`<br>├── `features/feature_pipeline.py`<br>├── `matching/model.py`<br>├── `matching/aggregation.py`<br>├── `preprocessing/normalize.py`<br>├── `preprocessing/keys.py`<br>├── `preprocessing/build_indexes.py`<br>├── `preprocessing/preprocess.py`<br>├── `preprocessing/translit_vocab.py`<br>├── `evaluation/metrics.py`<br>├── `evaluation/validation.py`<br>├── `io/`<br>├── `pipeline.py`, `train.py`, `predict.py`, `make_submission.py`, `scoring.py`, `config.py` | ~85 KB | **Retained & Active.** Canonical pipeline path. Fully offline, open-source compliant (MIT/Apache 2.0). |
| **Active Configs** | `configs/pipeline.yaml`<br>`configs/lexicons.json`<br>`configs/translit_vocab.json` | ~55 KB | **Retained & Active.** Core pipeline and offline multilingual lookup tables. |
| **Active Model Artifacts** | `models/lgbm_matcher.txt`<br>`models/lgbm_matcher.json`<br>`models/decision.json` | 4.2 MB total | Active baseline model checkpoint for experiment `M20260927_1454`. Kept for inference reproducibility. |
| **Active Outputs** | `output/candidate_pairs.tsv` (896.6 MB)<br>`output/matching_results.tsv` (89.4 MB) | ~986 MB | Generated inference outputs from test split. Added to `.gitignore` (`output/*.tsv`). |
| **Submission Variants** | `submissions/log.csv`<br>`submissions/matching_results_v1_best_t0.6_r0.7.tsv`<br>`submissions/matching_results_v2_stricter_t0.65_r0.7.tsv`<br>`submissions/matching_results_v3_looser_t0.55_r0.7.tsv`<br>`submissions/matching_results_v4_rel_t0.6_r0.5.tsv`<br>`submissions/matching_results_v5_precision_t0.7_r0.7.tsv` | ~450 MB total | Duplicate variants from test tuning grid. `log.csv` is preserved; heavy duplicate `.tsv` variants covered by `.gitignore`. |
| **Experiment Results** | `experiments/results/M20260927_1454/`<br>`experiments/experiment_registry.csv` | ~40 KB | Preserved historical audit trail and baseline metrics. |
| **Tests** | `tests/test_preprocessing.py`<br>`tests/test_matching_pipeline.py`<br>`tests/test_schema_validator.py`<br>`tests/test_smoke.py` | ~25 KB | Retained & passing unit tests. |

---

## 2. Dead / Legacy Files Confirmed and Removed

The following files were confirmed to be completely unreferenced on the active pipeline execution path (`configs/pipeline.yaml` -> `src/pipeline.py` -> `src/train.py` / `src/predict.py` / `src/make_submission.py`):

1. **`configs/blocking.yaml`**: Legacy Phase-1 config superseded by `configs/pipeline.yaml`. **Removed.**
2. **`configs/matching.yaml`**: Legacy Phase-1 config superseded by `configs/pipeline.yaml`. **Removed.**
3. **`src/blocking/multi_pass_blocking.py`**: Legacy unindexed multi-pass blocking stub superseded by high-performance index-join `src/blocking/candidates.py` (`CandidateGenerator`). **Removed.**
4. **`src/pipeline_baseline.py`**: Legacy baseline driver superseded by `src/pipeline.py`. **Removed.**
5. **`src/evaluation/blocking_metrics.py`**: 8-line stub superseded by production `blocking_report()` in `src/evaluation/metrics.py`. **Removed.**

---

## 3. Real 0.915344 Submission Identification

Cross-referencing:
1. `submissions/log.csv`:
   - `2026-09-27T16:25:54,v1_best,0.6,0.7,0.9305113265526137,5193761,133601,PASS,0,M20260927_1454`
2. `models/decision.json`:
   - `exp_id`: `M20260927_1454`
   - `best`: `{"threshold": 0.6, "relative": 0.7}`
   - `variant`: `v1_best` (validation macro F0.5 = 0.930511, micro precision = 0.986139, micro recall = 0.856352)
3. `output/matching_results.tsv`:
   - Exact file size: `89,407,049 bytes`
   - Identical byte-for-byte to `submissions/matching_results_v1_best_t0.6_r0.7.tsv`.

**Conclusion:** The exact submitted run that achieved 0.915344 is **`v1_best`** (threshold 0.60, relative 0.70, one-to-one resolution, experiment ID `M20260927_1454`).

---

## 4. Updates Made
- Cleaned up unreferenced legacy files.
- Hardened `.gitignore` to prevent tracking multi-gigabyte raw datasets, output TSVs, and duplicate submission files.
