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

Run the pipeline after the dataset schema and implementation are finalized:

```bash
python -m src.pipeline
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
