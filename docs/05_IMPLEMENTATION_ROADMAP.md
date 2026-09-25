# 5. Implementation Roadmap

## 5.1 Roadmap Objective

Build a complete, reproducible, and measurable entity-resolution pipeline through incremental phases.

Each phase has:

- An owner.
- Defined tasks.
- Expected outputs.
- Exit criteria.
- Documentation requirements.

The team should not begin complex modeling before the data contract and blocking baseline are understood.

## 5.2 Phase 0 — Repository and Project Contract

### Owners
All members; coordination by Member 3.

### Tasks

- Create the GitHub repository.
- Add the initial directory structure.
- Add environment and installation instructions.
- Confirm dataset filenames.
- Confirm input columns and identifier fields.
- Confirm candidate-pair output schema.
- Confirm final submission schema.
- Add issue labels and branch conventions.
- Create the decision log and experiment registry.
- Create the AI-agent context file.

### Deliverables

```text
README.md
docs/06_DECISION_LOG.md
docs/07_AI_AGENT_CONTEXT.md
experiments/experiment_registry.csv
configs/config.example.yaml
```

### Exit Criteria

- Every member can clone and set up the repository.
- Shared interfaces are written down.
- A basic smoke test passes.
- Open questions about the data schema are explicitly listed.

---

## 5.3 Phase 1 — Exploratory Data Analysis

### Owner
Member 2.

### Supporting members
Member 4 for blocking implications; Member 1 for feature implications; Member 3 for documentation.

### Tasks

- Inspect row counts for every dataset.
- Identify all columns and data types.
- Measure missingness.
- Check identifier uniqueness.
- Analyze duplicate records.
- Inspect name and address lengths.
- Analyze punctuation, casing, Unicode, and formatting variations.
- Identify possible country, language, and transliteration issues.
- Identify fields that are reliable enough for blocking.
- Prepare the data dictionary.

### Deliverables

```text
notebooks/01_data_profile.ipynb
docs/DATA_DICTIONARY.md
docs/EDA_REPORT.md
```

### Exit Criteria

- Data quality findings are documented.
- Candidate blocking keys have an evidence-based justification.
- Normalization requirements are approved by the team.

---

## 5.4 Phase 2 — Preprocessing and Normalization

### Owner
Member 2.

### Supporting members
Member 4 for blocking keys; Member 1 for feature consistency.

### Tasks

- Implement reusable normalization functions.
- Preserve raw and normalized values.
- Normalize casing and whitespace.
- Handle punctuation and Unicode consistently.
- Implement field-specific name and address normalization.
- Standardize missing values.
- Test normalization edge cases.
- Measure collisions introduced by normalization.

### Deliverables

```text
src/preprocessing/
tests/test_normalization.py
```

### Exit Criteria

- Normalization is deterministic.
- Tests cover nulls, whitespace, punctuation, Unicode, and repeated tokens.
- Normalization version is recorded in the configuration.
- The team agrees on the canonical field representation.

---

## 5.5 Phase 3 — Blocking Baselines

### Owner
Member 4.

### Supporting members
Member 2 for normalized keys; Member 3 for experiment documentation.

### Experiment Sequence

| Experiment | Strategy | Purpose |
|---|---|---|
| B0 | Small-sample all-pairs comparison | Debugging and reference baseline only |
| B1 | Exact normalized-name blocking | Establish simple baseline |
| B2 | Token-based name blocking | Handle token order and partial overlap |
| B3 | Name plus country/region | Reduce incompatible candidates |
| B4 | Name plus address/location | Add complementary context |
| B5 | Multi-pass union | Improve candidate recall |
| B6 | Candidate reduction | Control extreme candidate groups |

### Tasks

- Implement each blocking strategy independently.
- Deduplicate candidate pairs.
- Record statistics for every pass.
- Compare recall and candidate-set size.
- Identify overly large blocks.
- Implement configurable blocking passes.
- Generate the required candidate-pair artifact.

### Metrics

- Total candidate pairs.
- Mean candidates per reference entity.
- Median candidates per reference entity.
- Maximum candidates per reference entity.
- Candidate recall, if labels are available.
- Reduction ratio.
- Runtime.
- Peak memory.

### Deliverables

```text
src/blocking/
tests/test_blocking.py
experiments/results/
output/candidate_pairs.tsv
```

### Exit Criteria

- A blocking approach is selected using measured results.
- Candidate generation is reproducible.
- The team can explain the trade-off between candidate recall and candidate-set size.
- Candidate-pair output passes validation.

---

## 5.6 Phase 4 — Pairwise Feature Engineering

### Owner
Member 1.

### Supporting members
Member 2 for data-quality interpretation; Member 4 for candidate-pair interface.

### Tasks

- Define the feature schema.
- Implement name similarity features.
- Implement address similarity features.
- Implement exact agreement indicators.
- Add missingness indicators.
- Add conflict indicators.
- Check feature distributions.
- Detect leakage or unavailable inference-time features.
- Version the feature configuration.

### Deliverables

```text
src/features/
tests/test_features.py
configs/matching.yaml
```

### Exit Criteria

- Features can be generated from candidate pairs.
- Feature order and names are stable.
- Unit tests cover important edge cases.
- Feature generation runtime is measured.

---

## 5.7 Phase 5 — Matching Baseline and Model Experiments

### Owner
Member 1.

### Supporting member
Member 4 for evaluation and aggregation.

### Tasks

- Implement a transparent weighted-rule baseline.
- Define the scoring interface.
- Determine a valid validation strategy.
- Test a simple ML model if appropriate.
- Tune thresholds using the agreed evaluation protocol.
- Analyze false positives and false negatives.
- Compare results by blocking strategy.
- Document all model decisions.

### Deliverables

```text
src/matching/
tests/test_matching.py
experiments/experiment_registry.csv
```

### Exit Criteria

- A baseline is available.
- Any improved model has a measurable comparison against the baseline.
- Threshold selection is documented.
- Aggregation behavior is tested.
- The team can explain the model's limitations.

---

## 5.8 Phase 6 — End-to-End Integration

### Owner
Member 4.

### Supporting members
All members.

### Tasks

- Connect data loading, preprocessing, blocking, features, matching, aggregation, and output writing.
- Add a configuration-driven pipeline.
- Add end-to-end smoke tests.
- Validate identifier integrity.
- Verify output row counts.
- Confirm that candidate and final outputs follow the official schema.
- Record runtime and memory.
- Test the pipeline from a clean environment.

### Expected Flow

```python
load_data()
validate_schema()
normalize_records()
generate_candidates()
validate_candidates()
build_features()
score_pairs()
aggregate_matches()
evaluate_results()
write_outputs()
```

### Deliverables

```text
src/pipeline.py
tests/test_outputs.py
README.md
```

### Exit Criteria

- One documented command runs the full pipeline.
- All output files pass schema validation.
- A clean checkout can reproduce the pipeline.
- Runtime and memory are recorded.

---

## 5.9 Phase 7 — Optimization and Error Analysis

### Owners
All members, coordinated by Members 1 and 4.

### Tasks

- Re-run the strongest blocking configurations.
- Analyze false positives and false negatives.
- Inspect large candidate blocks.
- Compare thresholds.
- Review normalization collisions.
- Check for data leakage.
- Evaluate performance on difficult data slices.
- Reduce unnecessary runtime and memory usage.
- Confirm that candidate generation remains scalable.

### Exit Criteria

- The selected approach is supported by experiment records.
- Known failure modes are documented.
- The final configuration is frozen.
- No undocumented changes remain in the pipeline.

---

## 5.10 Phase 8 — Final Submission and Documentation

### Owners
All members; Member 3 coordinates.

### Tasks

- Validate `candidate_pairs.tsv`.
- Validate final matching output.
- Confirm all required files and columns.
- Freeze the final Git commit.
- Tag the final version.
- Complete the methodology document.
- Document candidate generation and blocking decisions.
- Document feature engineering and model architecture.
- Document evaluation and limitations.
- Add team contribution details.
- Preserve final experiment results.

### Final Documentation Outline

1. Problem understanding.
2. Dataset and data-quality analysis.
3. Preprocessing and normalization.
4. Candidate generation and blocking.
5. Pairwise features.
6. Matching model.
7. Entity-level aggregation.
8. Evaluation methodology.
9. Experiments and ablations.
10. Scalability and complexity.
11. Error analysis.
12. Limitations.
13. Reproducibility instructions.
14. Team contributions.

### Exit Criteria

- Final pipeline execution is reproducible.
- Submission artifacts pass validation.
- The final document describes the implemented system accurately.
- Every major technical claim is supported by an experiment, code reference, or documented observation.

## 5.11 AI-Agent Execution Protocol

Before every agent session:

1. Read `README.md`.
2. Read this roadmap.
3. Read `docs/07_AI_AGENT_CONTEXT.md`.
4. Inspect the relevant source files.
5. Confirm the task and expected interface.
6. Implement the smallest complete change.
7. Add or update tests.
8. Run the relevant tests.
9. Report changed files and test results.
10. Update the AI-agent context file.

After every agent session, record:

- Agent/tool used.
- User request or task.
- Files changed.
- Tests executed.
- Test results.
- Decisions made.
- Remaining issues.
- Recommended next task.

Agents must not invent schemas, silently change interfaces, delete existing functionality, or claim that tests passed without running them.

## 5.12 Phase Status Table

| Phase | Owner | Status | Exit Criteria Met |
|---|---|---|---|
| Phase 0 — Repository setup | All | Not started | No |
| Phase 1 — EDA | Member 2 | Not started | No |
| Phase 2 — Normalization | Member 2 | Not started | No |
| Phase 3 — Blocking | Member 4 | Not started | No |
| Phase 4 — Features | Member 1 | Not started | No |
| Phase 5 — Matching | Member 1 | Not started | No |
| Phase 6 — Integration | Member 4 | Not started | No |
| Phase 7 — Optimization | All | Not started | No |
| Phase 8 — Submission | All | Not started | No |
