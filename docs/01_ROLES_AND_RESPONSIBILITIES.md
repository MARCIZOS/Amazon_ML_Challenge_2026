# Team Roles and Responsibilities

## Member 1 — Modeling and Feature Engineering

Owns:

- Pairwise features.
- Baseline scoring.
- Model experiments.
- Threshold selection.
- Matching and aggregation logic.

Primary paths:

```text
src/features/
src/matching/
tests/test_features.py
tests/test_matching.py
```

## Member 2 — EDA and Preprocessing

Owns:

- Dataset profiling.
- Data dictionary.
- Missingness and duplicate analysis.
- Normalization and canonicalization.

Primary paths:

```text
src/io/data_loader.py
src/preprocessing/
notebooks/01_data_profile.ipynb
docs/DATA_DICTIONARY.md
docs/EDA_REPORT.md
```

## Member 3 — Documentation and AI-Agent Operations

Owns:

- Implementation plan.
- Decision log.
- Experiment registry.
- Agent context and handoffs.
- Final methodology document.

Primary paths:

```text
docs/
experiments/experiment_registry.csv
README.md
```

## Member 4 — Blocking, Evaluation, and Integration

Owns:

- Multi-pass blocking.
- Candidate statistics.
- Evaluation utilities.
- Output validation.
- End-to-end pipeline integration.

Primary paths:

```text
src/blocking/
src/evaluation/
src/io/output_writer.py
src/pipeline.py
tests/test_blocking.py
tests/test_outputs.py
```

## Shared expectations

- Use separate branches.
- Open pull requests.
- Write tests.
- Review at least one teammate's pull request.
- Document assumptions and limitations.
- Coordinate changes to shared interfaces.
