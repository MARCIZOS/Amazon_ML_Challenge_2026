# Implementation Roadmap

## Phase 0 — Repository and Contract Setup

### Tasks
- Confirm dataset filenames and schemas.
- Define canonical column names.
- Define output schemas.
- Add environment setup instructions.
- Add code formatting and test commands.
- Create issue labels and branch naming convention.

### Exit criteria
- All members can run the repository.
- Input and output contracts are written.
- A smoke test passes.

## Phase 1 — EDA and Data Contract

### Tasks
- Profile row counts and columns.
- Measure missingness.
- Inspect duplicate identifiers.
- Inspect field length and character distributions.
- Identify country, language, transliteration, and formatting patterns.
- Produce a data dictionary.

### Exit criteria
- EDA report is committed.
- Normalization decisions are recorded.
- Known data-quality risks are listed.

## Phase 2 — Normalization

### Tasks
- Implement field-specific normalizers.
- Preserve raw values alongside normalized values.
- Add unit tests for edge cases.
- Check for collisions introduced by normalization.

### Exit criteria
- Normalization is deterministic.
- Tests cover punctuation, casing, whitespace, Unicode, nulls, and repeated tokens.
- A before/after quality report exists.

## Phase 3 — Blocking Baselines

### Experiments
- B0: small-sample all-pairs baseline for debugging only.
- B1: exact normalized-name blocking.
- B2: token-based name blocking.
- B3: name plus country or region blocking.
- B4: name plus address/location blocking.
- B5: multi-pass union of blocking strategies.
- B6: candidate reduction for extreme candidate sets.

### Metrics
- Total candidate pairs.
- Mean candidates per reference entity.
- Median candidates per reference entity.
- Maximum candidates per reference entity.
- Candidate recall, if labels are available.
- Reduction ratio.
- Runtime and peak memory.

### Exit criteria
- A blocking strategy is selected using measured trade-offs.
- `candidate_pairs.tsv` can be generated reproducibly.
- Candidate statistics are saved.

## Phase 4 — Feature Engineering

### Tasks
- Build pairwise similarity features.
- Add missingness and conflict indicators.
- Check feature distributions.
- Detect leakage or features unavailable at inference time.
- Save feature definitions and versions.

### Exit criteria
- Feature matrix can be regenerated.
- Feature tests and schema checks pass.
- Feature computation runtime is measured.

## Phase 5 — Matching Baselines

### Tasks
- Establish a transparent weighted-rule baseline.
- Compare with a simple statistical model if labels or reliable training construction are available.
- Tune thresholds using a validation protocol.
- Analyze false positives and false negatives.
- Evaluate performance by data-quality slice.

### Exit criteria
- Baseline and candidate models are compared.
- Threshold selection is documented.
- Error examples are reviewed manually.

## Phase 6 — End-to-End Integration

### Tasks
- Connect loading, preprocessing, blocking, features, scoring, aggregation, and output writing.
- Add configuration-driven execution.
- Add deterministic seeds where relevant.
- Add end-to-end smoke tests.
- Verify output row counts and identifier integrity.

### Exit criteria
- One command runs the full pipeline.
- Output files pass validation.
- Runtime and memory are recorded.

## Phase 7 — Final Optimization and Submission

### Tasks
- Re-run the strongest experiments.
- Confirm no accidental data leakage.
- Confirm candidate-pair generation is scalable.
- Validate all required artifacts.
- Freeze the final configuration.
- Prepare the final methodology document.
- Tag the final commit.

### Exit criteria
- Final run is reproducible.
- Final files are validated.
- Final document matches the implemented system.
