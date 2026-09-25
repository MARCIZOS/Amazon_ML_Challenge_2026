# 3. Proposed Repository Structure

## Purpose

This structure separates data loading, preprocessing, blocking, feature engineering, matching, evaluation, experiments, tests, and documentation.

```text
amazon-entity-resolution/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── configs/
│   ├── config.example.yaml
│   ├── blocking.yaml
│   └── matching.yaml
├── src/
│   ├── pipeline.py
│   ├── config.py
│   ├── io/
│   ├── preprocessing/
│   ├── blocking/
│   ├── features/
│   ├── matching/
│   └── evaluation/
├── notebooks/
├── experiments/
│   ├── configs/
│   ├── results/
│   └── experiment_registry.csv
├── tests/
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
│   └── EDA_REPORT.md
├── output/
└── models/
```

## Ownership

- Member 1: `src/features/`, `src/matching/`
- Member 2: `src/io/data_loader.py`, `src/preprocessing/`, EDA notebooks
- Member 3: `docs/`, experiment registry, agent handoffs
- Member 4: `src/blocking/`, `src/evaluation/`, integration, output validation

## Rules

- Preserve raw and normalized fields separately.
- Keep blocking independent from matching.
- Do not hard-code experiment thresholds.
- Record meaningful experiments and decisions.
- Review and test AI-generated code before merging.
