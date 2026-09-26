# Amazon Entity Resolution Challenge

A reproducible, modular entity-resolution pipeline for candidate generation, pairwise matching, evaluation, and submission artifact creation.

## Project priorities

1. High-recall and scalable candidate generation.
2. Efficient blocking rather than full all-pairs comparison.
3. Transparent and testable pairwise features.
4. Reproducible matching and aggregation.
5. Complete experiment and decision tracking.
6. Clear documentation for human contributors and AI coding agents.

## Repository workflow

- Do not commit directly to `main`.
- Create a feature branch for each task.
- Open a pull request with tests and documentation.
- Record meaningful experiments in `experiments/experiment_registry.csv`.
- Update `docs/07_AI_AGENT_CONTEXT.md` after every substantial agent session.
- Never commit credentials, private data, or unrestricted generated artifacts.

## Suggested setup

```bash
python -m venv .venv
# Activate the environment
pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest -q
```

Run the pipeline: see *Full pipeline* below.

## Full pipeline (all members)

Running it on another machine: see [`RUN_ME_FIRST.md`](RUN_ME_FIRST.md).

See [`docs/PIPELINE.md`](docs/PIPELINE.md). Short version:

```bash
pip install -r requirements.txt
python -m src.preprocessing.preprocess --data-dir <DATASET_DIR>   # once
python -m src.pipeline                                          # indexes -> train -> predict -> submission v1
python -m src.make_submission --variant 2                       # v2..v5 in seconds
```

Outputs: `output/matching_results.tsv`, `output/candidate_pairs.tsv` (validated), reports in
`experiments/results/<exp_id>/`.

## Preprocessing (Member 2)

Data loading, EDA, normalisation and blocking-key analysis: see
[`docs/PREPROCESSING.md`](docs/PREPROCESSING.md), [`docs/EDA_REPORT.md`](docs/EDA_REPORT.md)
and [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

```bash
python -m src.preprocessing.preprocess --data-dir <DATASET_DIR>   # -> data/processed/*.parquet
```

## Team ownership

| Member | Ownership |
|---|---|
| Member 1 | Features and matching |
| Member 2 | EDA and preprocessing |
| Member 3 | Documentation and AI-agent context |
| Member 4 | Blocking, evaluation, and integration |

## Current status

The repository contains the initial structure, documentation, contracts, and implementation roadmap. Dataset-specific schemas and algorithms must be finalized after inspecting the official challenge files.
