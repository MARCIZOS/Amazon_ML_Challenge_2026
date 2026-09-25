# 4. System Architecture

## 4.1 Objective

The system is designed to resolve records from one or more source datasets to entities in a reference dataset.

The architecture separates candidate generation from candidate scoring. This is essential because the challenge evaluates not only the final matching output but also the quality and scalability of candidate generation.

## 4.2 High-Level Architecture

```text
                    ┌─────────────────────────┐
                    │      Raw Datasets       │
                    │ Reference + Sources     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Data Loading & Validation│
                    │ Schema, IDs, Data Types │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Preprocessing            │
                    │ Normalization            │
                    │ Canonical Representation│
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Multi-Pass Blocking      │
                    │ Exact / Token / Context │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Candidate Validation    │
                    │ Deduplication & Metrics │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Pairwise Features       │
                    │ Name / Address / Fields │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Matching Model          │
                    │ Rules or ML Scoring     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Thresholding &           │
                    │ Entity-Level Aggregation│
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Evaluation & Error      │
                    │ Analysis                 │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Submission Artifacts    │
                    │ candidate_pairs.tsv    │
                    │ matching_results.tsv   │
                    └─────────────────────────┘
```

## 4.3 Component Responsibilities

### 4.3.1 Data Loading and Validation

Input:
- Raw challenge datasets.

Responsibilities:
- Load files using a consistent interface.
- Validate required columns.
- Validate identifier uniqueness where required.
- Check data types.
- Report missing values and malformed records.
- Fail early when the input schema is invalid.

Output:
- Validated data frames or equivalent table structures.

### 4.3.2 Preprocessing and Normalization

The preprocessing layer creates canonical representations while preserving the original values.

Example:

```text
original_name       -> normalized_name
original_address    -> normalized_address
original_country    -> normalized_country
```

Potential operations include:

- Case normalization.
- Whitespace normalization.
- Punctuation handling.
- Unicode normalization.
- Field-specific tokenization.
- Missing-value standardization.
- Removal or standardization of known formatting noise.

Normalization must be deterministic and tested. It must not introduce uncontrolled collisions.

### 4.3.3 Multi-Pass Blocking

The blocking layer creates a manageable set of candidate pairs.

Possible blocking passes:

1. Exact normalized-name blocking.
2. Token-based name blocking.
3. Name plus country or region blocking.
4. Name plus address or location blocking.
5. Additional field-based blocks, if supported by the dataset.

The final candidate set should generally be the deduplicated union of the blocking passes.

Conceptually:

```text
Candidates =
    Block_Exact_Name
    UNION Block_Token_Name
    UNION Block_Name_Country
    UNION Block_Name_Address
```

The blocking system must:

- Avoid full all-pairs comparison on the complete dataset.
- Preserve candidate recall as much as possible.
- Record candidate counts.
- Track very large blocks.
- Support deterministic output.
- Make blocking strategies configurable.

Important principle:

> If a true match is excluded during blocking, the matching model cannot recover it later.

### 4.3.4 Candidate Validation

Before scoring candidate pairs:

- Remove duplicate pairs.
- Validate reference and source identifiers.
- Check that every candidate belongs to an allowed source dataset.
- Measure candidates per reference entity.
- Identify unusually large candidate groups.
- Save candidate statistics.
- Write the required `candidate_pairs.tsv` artifact in the correct schema.

### 4.3.5 Pairwise Feature Engineering

For each candidate pair, calculate features such as:

- Exact field agreement.
- Character-level name similarity.
- Token-level name similarity.
- Address similarity.
- Country or region agreement.
- Length differences.
- Missingness indicators.
- Conflicting-field indicators.

Feature generation must be consistent between experimentation and final inference.

### 4.3.6 Matching Model

The matching layer converts pairwise features into scores or match probabilities.

Development order:

1. Transparent weighted-rule baseline.
2. Simple statistical model, if training labels or a valid training strategy exist.
3. More complex model only if it provides a measurable benefit.

Model configuration should be externalized, including:

- Model type.
- Feature version.
- Threshold.
- Random seed.
- Any class-weight or calibration settings.

### 4.3.7 Thresholding and Aggregation

Pairwise scores must be converted into entity-level matches.

Responsibilities:

- Apply a documented threshold.
- Handle ambiguous candidates.
- Group matches by reference entity.
- Support multiple matches if the task permits them.
- Handle entities with no accepted candidate when permitted.
- Preserve the required output identifiers and structure.

The aggregation logic must be tested independently because errors at this stage can affect many records.

### 4.3.8 Evaluation

Evaluation should be divided into separate components.

#### Blocking evaluation

Measure:

- Candidate recall, if ground-truth matches are available.
- Total candidate pairs.
- Mean and median candidates per reference entity.
- Maximum candidates per reference entity.
- Reduction ratio.
- Runtime.
- Peak memory.

#### Matching evaluation

Measure:

- Required challenge metric.
- False positives.
- False negatives.
- Performance by data-quality slice.
- Threshold sensitivity.
- Performance by blocking strategy.

## 4.4 Data Flow Contracts

### Preprocessing contract

```text
Input:
    Raw records

Output:
    Original fields + normalized fields + stable identifiers
```

### Blocking contract

```text
Input:
    Reference records and source records with normalized fields

Output:
    Deduplicated candidate pairs with source and reference identifiers
```

### Feature contract

```text
Input:
    Candidate pairs and canonical records

Output:
    One feature vector per candidate pair
```

### Matching contract

```text
Input:
    Pairwise feature matrix

Output:
    Candidate-pair scores and match decisions
```

### Output contract

```text
Input:
    Accepted pairwise decisions

Output:
    Valid candidate-pair and final matching submission artifacts
```

The exact column names must be finalized after inspecting the challenge datasets and official submission requirements.

## 4.5 Non-Functional Requirements

### Reproducibility
The same data version and configuration should produce the same output, subject to documented randomness.

### Scalability
Candidate generation must avoid unnecessary all-pairs comparisons and must report candidate-set size and runtime.

### Testability
Each module should be testable independently, with end-to-end smoke tests for the complete pipeline.

### Observability
The pipeline should log:

- Input row counts.
- Number of normalized records.
- Number of candidate pairs per blocking pass.
- Deduplicated candidate count.
- Feature generation count.
- Number of accepted matches.
- Runtime and memory where practical.

### Maintainability
Business logic should be separated into small modules with clear interfaces and minimal hidden state.

## 4.6 Execution Model

The project will initially use local execution rather than making SageMaker a mandatory dependency.

SageMaker or another cloud execution environment may be reconsidered only if the dataset size, runtime, memory, or team workflow creates a demonstrated need.
