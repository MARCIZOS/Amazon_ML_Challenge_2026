# Amazon Entity Resolution Challenge

## Objective

Build a scalable entity-resolution pipeline that maps records from the source datasets to the corresponding entities in the reference dataset.

The solution must prioritize:

1. High candidate recall during blocking.
2. Small and computationally manageable candidate sets.
3. Reliable pairwise matching.
4. Correct entity-level aggregation.
5. Reproducible experiments and documented decisions.
6. Valid final submission artifacts, including `candidate_pairs.tsv`.

## Working Principles

- Every meaningful change is linked to an issue or task.
- Every experiment has a configuration and a result record.
- No model is promoted without comparison against a baseline.
- Blocking and matching are evaluated separately.
- AI-agent-generated code must be reviewed, tested, and documented by a human.
- Never commit raw confidential data, credentials, or large generated artifacts unless explicitly permitted.

## Quick Start

```bash
python -m venv .venv
# Activate the environment
pip install -r requirements.txt
python -m src.pipeline
```

The exact commands will be updated after the dataset schema and execution environment are confirmed.
