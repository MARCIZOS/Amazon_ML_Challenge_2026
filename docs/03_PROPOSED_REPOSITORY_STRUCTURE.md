# 3. Proposed Repository Structure

## 3.1 Purpose

This repository structure separates data processing, blocking, feature engineering, matching, evaluation, experimentation, and documentation.

The design supports:

- Parallel development by four team members.
- Independent testing of each module.
- Reproducible experiments.
- Easy handoff between humans and AI coding agents.
- Clear documentation of technical decisions.
- Scalable candidate generation without unnecessary all-pairs comparisons.

## 3.2 Directory Structure

```text
amazon-entity-resolution/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── data/
│   ├── raw/                  # Local-only input data
│   ├── interim/              # Intermediate transformed data
│   └── processed/            # Canonical/processed data
│
├── configs/
│   ├── config.example.yaml
│   ├── blocking.yaml
│   └── matching.yaml
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── config.py
│   │
│   ├── io/
│   │   ├── data_loader.py
│   │   ├── schema_validator.py
│   │   └── output_writer.py
│   │
│   ├── preprocessing/
│   │   ├── normalize_text.py
│   │   ├── normalize_name.py
│   │   ├── normalize_address.py
│   │   └── canonicalize.py
│   │
│   ├── blocking/
│   │   ├── exact_blocking.py
│   │   ├── token_blocking.py
│   │   ├── fuzzy_blocking.py
│   │   ├── multi_pass_blocking.py
│   │   └── candidate_reduction.py
│   │
│   ├── features/
│   │   ├── name_features.py
│   │   ├── address_features.py
│   │   ├── field_agreement.py
│   │   └── feature_pipeline.py
│   │
│   ├── matching/
│   │   ├── baseline_rules.py
│   │   ├── model.py
│   │   ├── thresholding.py
│   │   └── aggregation.py
│   │
│   └── evaluation/
│       ├── blocking_metrics.py
│       ├── matching_metrics.py
│       ├── error_analysis.py
│       └── validation.py
│
├── notebooks/
│   ├── 01_data_profile.ipynb
│   ├── 02_normalization_analysis.ipynb
│   ├── 03_blocking_experiments.ipynb
│   └── 04_matching_analysis.ipynb
│
├── experiments/
│   ├── experiment_registry.csv
│   ├── configs/
│   └── results/
│
├── tests/
│   ├── test_normalization.py
│   ├── test_blocking.py
│   ├── test_features.py
│   ├── test_matching.py
│   └── test_outputs.py
│
├── docs/
│   ├── 00_PROJECT_CHARTER.md
│   ├── 01_ROLES_AND_RESPONSIBILITIES.md
│   ├── 02_ARCHITECTURE.md
│   ├── 03_PROPOSED_REPOSITORY_STRUCTURE.md
│   ├── 04_SYSTEM_ARCHITECTURE.md
│   ├── 05_IMPLEMENTATION_ROADMAP.md
│   ├── 06_DECISION_LOG.md
│   ├── 07_AI_AGENT_CONTEXT.md
│   ├── DATA_DICTIONARY.md
│   ├── EDA_REPORT.md
│   └── decisions/
│
├── output/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
│
└── models/
```

## 3.3 Directory Responsibilities

| Directory | Responsibility |
|---|---|
| `data/` | Raw, intermediate, and processed datasets |
| `configs/` | Runtime settings, blocking strategies, and model thresholds |
| `src/io/` | Data loading, schema validation, and output writing |
| `src/preprocessing/` | Field normalization and canonicalization |
| `src/blocking/` | Candidate-pair generation and candidate reduction |
| `src/features/` | Pairwise similarity and agreement features |
| `src/matching/` | Scoring, thresholding, and entity-level aggregation |
| `src/evaluation/` | Blocking metrics, matching metrics, and error analysis |
| `notebooks/` | Exploratory analysis and visual investigation |
| `experiments/` | Experiment configurations and result records |
| `tests/` | Unit, integration, and output validation tests |
| `docs/` | Project knowledge, decisions, plans, and agent handoffs |
| `output/` | Generated submission and analysis artifacts |
| `models/` | Locally generated model artifacts, if needed |

## 3.4 Ownership by Team Member

### Member 1 — Modeling and Feature Engineering

Primary ownership:

```text
src/features/
src/matching/
tests/test_features.py
tests/test_matching.py
```

### Member 2 — EDA and Preprocessing

Primary ownership:

```text
notebooks/01_data_profile.ipynb
src/io/data_loader.py
src/preprocessing/
docs/DATA_DICTIONARY.md
docs/EDA_REPORT.md
tests/test_normalization.py
```

### Member 3 — Documentation and AI-Agent Operations

Primary ownership:

```text
docs/
experiments/experiment_registry.csv
README.md
```

Member 3 coordinates documentation but does not need to author every technical detail. Each module owner must document their own implementation decisions.

### Member 4 — Blocking, Evaluation, and Integration

Primary ownership:

```text
src/blocking/
src/evaluation/
src/io/output_writer.py
tests/test_blocking.py
tests/test_outputs.py
src/pipeline.py
```

## 3.5 Repository Rules

1. Do not commit raw or restricted datasets unless explicitly permitted.
2. Do not commit credentials, API keys, or local environment files.
3. Keep raw and normalized fields separate.
4. Avoid hard-coded thresholds in source code.
5. Keep blocking separate from final matching.
6. Every important experiment must have a recorded configuration and result.
7. Every AI-agent implementation must be reviewed and tested by a team member.
8. Do not merge changes that modify shared interfaces without notifying the relevant owners.
