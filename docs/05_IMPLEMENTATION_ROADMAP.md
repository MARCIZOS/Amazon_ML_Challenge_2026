# 5. Implementation Roadmap

## Phase 0 — Repository and contracts

**Owners:** All members; coordination by Member 3.

Tasks:

- Create repository and branch conventions.
- Confirm dataset filenames and schemas.
- Confirm candidate-pair and final-output schemas.
- Add environment instructions.
- Create decision log, experiment registry, and agent context file.

Exit criteria:

- Every member can set up the repository.
- Shared interfaces and open questions are documented.
- A smoke test passes.

## Phase 1 — EDA

**Owner:** Member 2.

Tasks:

- Inspect row counts, columns, data types, and missingness.
- Check identifier uniqueness and duplicates.
- Analyze name/address formatting and Unicode.
- Identify reliable blocking keys.
- Create the data dictionary and EDA report.

Deliverables:

```text
notebooks/01_data_profile.ipynb
docs/DATA_DICTIONARY.md
docs/EDA_REPORT.md
```

Exit criteria:

- Data-quality risks are documented.
- Blocking and normalization decisions are evidence-based.

## Phase 2 — Normalization

**Owner:** Member 2.

Tasks:

- Implement reusable field-specific normalizers.
- Preserve raw and normalized values.
- Handle casing, whitespace, punctuation, Unicode, and nulls.
- Test edge cases.
- Analyze normalization collisions.

Exit criteria:

- Normalization is deterministic and tested.
- A normalization version is recorded.

## Phase 3 — Blocking baselines

**Owner:** Member 4.

Experiments:

| ID | Strategy | Purpose |
|---|---|---|
| B0 | Small-sample all-pairs | Debugging only |
| B1 | Exact normalized name | Simple baseline |
| B2 | Token-based name | Handle token order and partial overlap |
| B3 | Name plus country/region | Reduce incompatible candidates |
| B4 | Name plus address/location | Add contextual constraints |
| B5 | Multi-pass union | Improve candidate recall |
| B6 | Candidate reduction | Control extreme candidate groups |

Record:

- Total candidate pairs.
- Mean, median, and maximum candidates per entity.
- Candidate recall where available.
- Reduction ratio.
- Runtime and memory.

Exit criteria:

- A blocking strategy is selected using measured trade-offs.
- `candidate_pairs.tsv` is reproducibly generated and validated.

## Phase 4 — Feature engineering

**Owner:** Member 1.

Tasks:

- Define a stable feature schema.
- Implement name, address, and agreement features.
- Add missingness and conflict indicators.
- Check feature distributions.
- Detect leakage.
- Version feature configuration.

Exit criteria:

- Features can be regenerated consistently from candidate pairs.
- Unit tests pass.

## Phase 5 — Matching

**Owner:** Member 1; support from Member 4.

Tasks:

- Implement weighted-rule baseline.
- Define validation strategy.
- Test a simple ML model if justified.
- Tune thresholds.
- Analyze false positives and false negatives.
- Compare performance by blocking strategy.

Exit criteria:

- Baseline and improved models are compared.
- Threshold selection and aggregation rules are documented.

## Phase 6 — End-to-end integration

**Owner:** Member 4; support from all members.

Tasks:

- Connect loading, preprocessing, blocking, features, matching, aggregation, and output writing.
- Add configuration-driven execution.
- Add smoke tests and output validation.
- Verify identifier integrity and output schemas.
- Record runtime and memory.

Expected flow:

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

Exit criteria:

- One documented command runs the full pipeline.
- Outputs pass validation from a clean environment.

## Phase 7 — Optimization and error analysis

**Owners:** All members.

Tasks:

- Re-run strongest configurations.
- Analyze false positives, false negatives, and large candidate blocks.
- Check normalization collisions and leakage.
- Compare thresholds and data-quality slices.
- Reduce unnecessary runtime and memory.

Exit criteria:

- Final approach is supported by experiment records.
- Known failure modes and limitations are documented.

## Phase 8 — Final submission

**Owners:** All members; coordination by Member 3.

Tasks:

- Validate `candidate_pairs.tsv`.
- Validate final matching output.
- Freeze configuration and final commit.
- Tag the final version.
- Complete the methodology document.
- Document blocking, features, model, evaluation, scalability, limitations, and team contributions.

Exit criteria:

- Final run is reproducible.
- Submission artifacts are validated.
- Documentation matches the actual implementation.

## AI-agent execution protocol

Before every agent session:

1. Read `README.md`.
2. Read this roadmap.
3. Read `docs/07_AI_AGENT_CONTEXT.md`.
4. Inspect existing code.
5. Confirm the task and interfaces.
6. Implement the smallest complete change.
7. Add or update tests.
8. Run relevant tests.
9. Report changed files and results.
10. Update the context file.

After every session, record:

- Agent/tool.
- Task.
- Files changed.
- Tests run and results.
- Decisions.
- Remaining issues.
- Recommended next task.

Agents must not invent schemas, silently change interfaces, delete existing functionality, or claim tests passed without running them.
