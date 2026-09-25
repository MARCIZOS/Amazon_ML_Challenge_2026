# System Architecture

## High-Level Pipeline

```text
Raw datasets
    |
    v
Data loading and schema validation
    |
    v
Field normalization and canonical representation
    |
    v
Exploratory profiling and quality checks
    |
    v
Multi-pass candidate generation / blocking
    |
    v
Candidate deduplication and candidate statistics
    |
    v
Pairwise feature extraction
    |
    v
Match scoring model or rules
    |
    v
Thresholding and ambiguity handling
    |
    v
Entity-level aggregation
    |
    v
Evaluation and error analysis
    |
    v
Submission and audit artifacts
```

## Design Boundaries

### Preprocessing layer
Input: raw records  
Output: canonical records and normalized fields

Responsibilities:
- Type coercion.
- Whitespace and punctuation handling.
- Unicode normalization.
- Case folding.
- Missing-value handling.
- Field-specific normalization.

### Blocking layer
Input: canonical records  
Output: candidate pairs

Responsibilities:
- Generate candidates through multiple blocking passes.
- Use the union of blocking passes.
- Deduplicate pairs.
- Avoid all-pairs comparison.
- Record candidate counts and coverage.

Important invariant:

> Blocking must not silently discard a plausible match merely because one blocking key is missing or noisy.

### Feature layer
Input: candidate pairs  
Output: pairwise feature matrix

Potential features:
- Exact agreement indicators.
- Character-level name similarity.
- Token-based name similarity.
- Address similarity.
- Phone/email agreement where available.
- Country or region agreement.
- Missingness indicators.
- Conflict indicators.

Features must be computed only from permitted input data.

### Matching layer
Input: pairwise features  
Output: pairwise scores or match probabilities

Responsibilities:
- Fit or apply the selected scoring approach.
- Keep threshold configuration separate from code.
- Support reproducible inference.
- Record model version and feature version.

### Aggregation layer
Input: scored candidate pairs  
Output: entity-level matches

Responsibilities:
- Group predictions by reference entity.
- Apply the selected threshold and ambiguity rules.
- Preserve the required output schema.
- Handle entities with no accepted match when the task permits it.

### Evaluation layer
Responsibilities:
- Evaluate blocking separately from matching.
- Report candidate recall where labels are available.
- Report candidate-set size statistics.
- Calculate the required matching metric.
- Perform slice-based error analysis.
