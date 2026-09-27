# AI-Agent Context Preservation

This file is the persistent handoff document for AI coding agents.

## Current project state

- Current phase: Phase 3 — full pipeline built; awaiting first full-data run
- Current objective: Confirm schemas and shared interfaces
- Last completed task: Initial repository structure created
- Next task: Inspect official datasets and update data contracts
- Current branch:
- Latest commit:

## Contracts

- Reference dataset schema: confirmed - see `docs/DATA_DICTIONARY.md` (S1 = deduplicated reference)
- Source dataset schema: confirmed - `entity_id, business_name, business_address, country`; read with quoting OFF
- Normalised data contract: `data/processed/<split>/source{1,2,3}.parquet` = 4 original columns + 14 fields listed in `docs/DATA_DICTIONARY.md`
- Candidate-pair schema: Pending confirmation
- Final output schema: Pending confirmation
- Main execution command: `python -m src.pipeline` (see docs/PIPELINE.md)

## Component status

| Component | Status | Owner | Notes |
|---|---|---|---|
| Data loader | Done | Member 2 | `src/io/data_loader.py`: original loaders + chunked `iter_tsv`, `ParquetAppender`, `ground_truth_pairs` |
| Normalization | Done (v2) | Member 2 | `src/preprocessing/normalize.py` + `preprocess.py`; full data processed in 19 min; `canonicalize.py` kept for the Phase-1 baseline |
| Blocking | Done (v2) | Member 4 | Key analysis available: `python -m src.preprocessing.blocking_keys` -> `docs/BLOCKING_KEYS.md`. Pass 4 'postcode' actually uses house numbers (D005) |
| Features | Done | Member 1 | `src/features/pair_features.py` (~50 vectorised features) |
| Matching | Done | Member 1 | LightGBM `src/matching/model.py`; decision rules `src/matching/aggregation.py` |
| Evaluation | Done | Member 4 | `src/evaluation/metrics.py` macro F0.5 + blocking report |
| Output writer | Done | Member 4 | `src/io/output_writer.py` + official validator |

## Instructions for every agent

1. Read this file and the roadmap before coding.
2. Inspect existing files before creating new ones.
3. Preserve public interfaces unless explicitly approved.
4. Add or update tests.
5. Run the smallest relevant test suite.
6. Report changed files, commands, results, and risks.
7. Update this file before ending the session.

## Agent session record

### Date

- Agent/tool:
- Task:
- Files changed:
- Tests run:
- Results:
- Decisions:
- Remaining work:
- Recommended next task:

### 2026-09-26 — Member 2 preprocessing

- Agent/tool: Claude (Cowork)
- Task: EDA, data dictionary, Unicode-safe normalisation, memory-safe preprocessing pipeline, benchmark and blocking-key analysis (Member 2 scope)
- Files changed:
  - new: `src/preprocessing/{normalize,translit_vocab,preprocess,keys,blocking_keys,build_indexes}.py`, `src/io/benchmark.py`,
    `configs/{lexicons,translit_vocab}.json`, `scripts/{eda,eval_normalization,build_lexicons}.py`,
    `tests/test_preprocessing.py`, `docs/PREPROCESSING.md`, `docs/assets/eda/*.png`
  - updated: `src/io/data_loader.py` (additions only), `docs/EDA_REPORT.md`, `docs/DATA_DICTIONARY.md`,
    `docs/06_DECISION_LOG.md` (D003 accepted, D005-D009), `experiments/experiment_registry.csv` (P001),
    `requirements.txt`, `README.md` (pointer section)
  - not touched: `src/pipeline.py`, `src/blocking/`, `src/features/`, `src/matching/`, `src/preprocessing/canonicalize.py`
- Tests run: `python -m pytest -q` - 40 passed (3 existing + 37 new)
- Results: 24,229,173 rows normalised in 1,134 s; true-pair name similarity 0.849 -> 0.925,
  Indian-script names 0.146 -> 0.826, address 0.853 -> 0.933 (held-out EDA sample)
- Decisions: D003 accepted; D005 no postcodes; D006 anyascii + learned vocab; D007 Parquet store; D008/D009 proposed
- Remaining work: run `src.io.benchmark` and `src.preprocessing.blocking_keys` on the full data and commit
  `docs/BENCHMARK.md` + `docs/BLOCKING_KEYS.md`; Member 4 to switch blocking to the processed fields
- Recommended next task: Member 4 - replace baseline blocking passes with the greedy-union keys from
  `docs/BLOCKING_KEYS.md`, using the prebuilt indexes (`python -m src.preprocessing.build_indexes`,
  then `build_indexes.candidates(...)`); Member 1 - build features on
  `name_core`, `name_key`, `name_compact`, `addr_core`, `addr_key`, `addr_nums`, `addr_state`, `name_legal`
- Known issues for others: `src/create_mini_dataset.py` reads TSVs without `quoting=csv.QUOTE_NONE`
  (can mis-parse names containing quotes); use `src.io.data_loader.read_tsv`

### 2026-09-26 — Full pipeline (all roles)

- Agent/tool: Claude (Cowork)
- Task: blocking, features, model, decision, evaluation, outputs, entry points
- Files: new `src/blocking/candidates.py`, `src/features/pair_features.py`, `src/evaluation/metrics.py`,
  `src/scoring.py`, `src/train.py`, `src/predict.py`, `src/make_submission.py`, `configs/pipeline.yaml`,
  `docs/PIPELINE.md`, `tests/test_matching_pipeline.py`; rewritten `src/pipeline.py` (driver),
  `src/matching/{model,aggregation}.py` (legacy classes kept), `src/io/output_writer.py`, `src/config.py`;
  old driver kept as `src/pipeline_baseline.py`; `build_indexes.py` parallel + per-key limits
- Tests: `python -m pytest -q` - 46 passed; end-to-end run on a 3,000-group real-row dev set:
  both output files produced, official validator PASS for all 5 variants
- Results on full data: none yet (awaiting the first run - do not quote dev-set numbers)
- Next: run `python -m src.pipeline`, upload variants, record public scores; tune `top_k` from the
  blocking recall in the report; error analysis from `errors.tsv`
