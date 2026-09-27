# 4. System Architecture

## 4.1 Objective

Build a scalable entity-resolution pipeline that maps records from source datasets to entities in a reference dataset.

The system separates candidate generation from candidate scoring because candidate generation must be both efficient and sufficiently high-recall.

## 4.2 Pipeline

```text
Raw datasets
    ↓
Data loading and schema validation
    ↓
Preprocessing and field normalization
    ↓
Multi-pass blocking
    ↓
Candidate deduplication and validation
    ↓
Candidate statistics
    ↓
Pairwise feature extraction
    ↓
Matching model or rules
    ↓
Thresholding and entity-level aggregation
    ↓
Evaluation and error analysis
    ↓
Submission artifacts
```

## 4.3 Components

### Data loading and validation

Responsibilities:

- Load all required datasets.
- Validate required columns and identifier fields.
- Check data types and malformed rows.
- Report missingness and duplicate identifiers.
- Fail early on schema errors.

### Preprocessing

Create canonical fields while preserving originals.

Example:

```text
original_name     → normalized_name
original_address  → normalized_address
original_country  → normalized_country
```

Potential operations:

- Case normalization.
- Whitespace and punctuation normalization.
- Unicode normalization.
- Field-specific tokenization.
- Missing-value standardization.

All transformations must be deterministic and tested.

### Multi-pass blocking

The blocking layer produces candidate pairs without comparing every record against every other record.

Potential passes:

1. Exact normalized-name blocking.
2. Token-based name blocking.
3. Name plus country or region.
4. Name plus address or location.
5. Additional dataset-supported blocking keys.

The final candidate set is generally the deduplicated union of the passes:

```text
Candidates =
    ExactName
    UNION TokenName
    UNION NameCountry
    UNION NameAddress
```

The blocking layer must report:

- Number of candidates per pass.
- Deduplicated candidate count.
- Mean, median, and maximum candidates per reference entity.
- Candidate recall where labels are available.
- Runtime and memory.

### Pairwise features

Possible features include:

- Exact field agreement.
- Character-level name similarity.
- Token-level name similarity.
- Address similarity.
- Country or region agreement.
- Length differences.
- Missingness indicators.
- Conflicting-field indicators.

The final feature set depends on the actual dataset schema.

### Matching

Development order:

1. Transparent weighted-rule baseline.
2. Simple statistical model if valid labels or a defensible training strategy exist.
3. More complex model only when it demonstrates a measurable benefit.

The configuration should define the model, feature version, threshold, and random seed.

### Aggregation

The aggregation layer:

- Applies match thresholds.
- Handles ambiguous candidates.
- Groups pairwise decisions by reference entity.
- Supports multiple matches if permitted.
- Handles entities with no accepted match if permitted.
- Writes the required output schema.

### Evaluation

Blocking and matching must be evaluated separately.

Blocking metrics:

- Candidate recall.
- Candidate count.
- Mean, median, and maximum candidates.
- Reduction ratio.
- Runtime and peak memory.

Matching metrics:

- Required challenge metric.
- False positives and false negatives.
- Threshold sensitivity.
- Performance by data-quality slice.
- Comparison across blocking strategies.

## 4.4 Data contracts

### Preprocessing

```text
Input: raw records
Output: stable IDs + original fields + normalized fields
```

### Blocking

```text
Input: normalized reference and source records
Output: deduplicated candidate pairs with valid IDs
```

### Features

```text
Input: candidate pairs and canonical records
Output: one feature vector per candidate pair
```

### Matching

```text
Input: pairwise feature matrix
Output: pairwise scores and decisions
```

### Output

```text
Input: accepted pairwise decisions
Output: validated candidate_pairs.tsv and final matching output
```

Exact column names must be finalized after inspecting the official challenge files.

## 4.5 Non-functional requirements

- Reproducibility: same data and configuration should reproduce the same output.
- Scalability: avoid unnecessary all-pairs comparisons.
- Testability: modules should have unit tests and the pipeline should have smoke tests.
- Observability: log row counts, candidate counts, accepted matches, runtime, and memory.
- Maintainability: use small modules and explicit interfaces.

## 4.6 Execution environment

The initial implementation will use local Python execution. SageMaker is optional and should only be introduced if measured data size, memory, runtime, or workflow needs justify it.
