# Team Roles and Responsibilities

The four roles are separated by ownership, but all members review interfaces and contribute to integration.

## Member 1 — Modeling and Feature Engineering

### Primary ownership
- Pairwise feature design.
- Baseline scoring model.
- Supervised or weakly supervised model experiments, if applicable.
- Threshold selection and calibration.
- Model artifact and inference interface.

### Deliverables
- `src/features/`
- `src/matching/model.py`
- `docs/decisions/`
- Model experiment records in `experiments/`

### Must coordinate with
- EDA member for field reliability.
- Blocking member for candidate-pair schema.
- Evaluation member for threshold and metric analysis.

## Member 2 — EDA and Data Preprocessing

### Primary ownership
- Dataset profiling.
- Schema and data-quality report.
- Missing-value analysis.
- Normalization and standardization.
- Duplicate and collision analysis.
- Reusable preprocessing functions.

### Deliverables
- `notebooks/01_data_profile.ipynb`
- `src/io/data_loader.py`
- `src/preprocessing/`
- `docs/DATA_DICTIONARY.md`
- `docs/EDA_REPORT.md`

### Must coordinate with
- Blocking member on normalized keys.
- Modeling member on feature reliability.
- Documentation member on dataset observations.

## Member 3 — Documentation and Research Operations

### Primary ownership
- Repository governance.
- Implementation plan and phase tracking.
- Decision log.
- Experiment registry.
- AI-agent context preservation.
- Final methodology document.

### Deliverables
- `docs/IMPLEMENTATION_PLAN.md`
- `docs/DECISION_LOG.md`
- `docs/EXPERIMENT_LOG.md`
- `docs/AI_AGENT_CONTEXT.md`
- Final submission document.

### Must coordinate with
- All members. Documentation is updated from actual merged work, not assumptions.

## Member 4 — Blocking, Matching Integration, and Evaluation

### Primary ownership
- Candidate generation and blocking.
- Candidate-pair statistics.
- Evaluation framework.
- Output validation.
- End-to-end integration.

### Deliverables
- `src/blocking/`
- `src/evaluation/`
- `src/io/output_writer.py`
- `tests/test_blocking.py`
- `tests/test_outputs.py`

### Must coordinate with
- EDA member on keys.
- Modeling member on candidate-pair features.
- Documentation member on experiment results.

## Shared Responsibilities

Every member must:

- Work on a separate branch.
- Write tests for non-trivial logic.
- Open a pull request.
- Explain changes in the PR description.
- Review at least one teammate's PR.
- Record assumptions and unresolved risks.
- Avoid directly editing another member's module without coordination.
