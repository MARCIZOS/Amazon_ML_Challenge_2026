# Source Code Layout

- `io/`: loading, schema checks, and output writing.
- `preprocessing/`: normalization and canonicalization.
- `blocking/`: candidate generation and candidate reduction.
- `features/`: pairwise feature extraction.
- `matching/`: scoring models and aggregation.
- `evaluation/`: blocking metrics, matching metrics, and error analysis.
- `pipeline.py`: orchestration entry point.

Keep modules focused. Avoid placing exploratory code inside production modules.
