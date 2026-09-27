# Business Entity Resolution Pipeline

High-accuracy, scalable, fully offline entity resolution pipeline for the Amazon ML Challenge 2026.

Designed to scale across multi-million record datasets, support unseen countries (including France), process Indian and multilingual scripts without external APIs, and optimize macro F0.5 (with singleton protection).

---

## 1. System Requirements & Setup

- **Python Version:** Python 3.10 – 3.12 (64-bit)
- **Dependencies:** Install the pinned dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- **External Resources:** Fully offline. **Zero** external API, web lookup, geocoding, or internet calls.
- **Model:** LightGBM Gradient Boosted Decision Trees (MIT License, ~4 MB model size, well below 8B parameter constraint).

---

## 2. Directory Layout & Configuration

The pipeline defaults to relative paths, with optional environment variable overrides:
- `DATASET_DIR` (default: `dataset/`)
- `PROCESSED_DIR` (default: `data/processed/`)
- `MODELS_DIR` (default: `models/`)
- `OUTPUT_DIR` (default: `output/`)
- `RESULTS_DIR` (default: `experiments/results/`)
- `VALIDATOR_PATH` (default: `dataset/utils/validate_submission.py`)

Place competition dataset files in `dataset/`:
```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

---

## 3. End-to-End Execution Guide

### Step 1: Preprocessing & Multilingual Normalization
Converts raw TSV files into compact columnar Parquet files with phonetic skeletons, legal form stripping, and script normalization:
```bash
python -m src.preprocessing.preprocess --data-dir dataset --out-dir data/processed
```

*(Optional: Learn native-script transliteration vocabulary from training matches)*
```bash
python -m src.preprocessing.translit_vocab --data-dir dataset
```

### Step 2: Build Compact Inverted Blocking Indexes
Builds country-partitioned 64-bit hashed blocking indexes:
```bash
python -m src.preprocessing.build_indexes --config configs/pipeline.yaml
```

### Step 3: Train Matcher & Tune Decision Thresholds
Generates training pairs against distractor candidates, extracts vectorized string similarity & group ranking features, trains the LightGBM classifier with early stopping, and tunes decision thresholds:
```bash
python -m src.train --config configs/pipeline.yaml
```

### Step 4: Test Scoring & Candidate Generation
Scores all test candidate pairs, saves `output/candidate_pairs.tsv` (exactly the candidate pairs evaluated by the model), and caches prediction scores:
```bash
python -m src.predict --config configs/pipeline.yaml
```

### Step 5: Format & Validate Final Submission
Generates `output/matching_results.tsv` using one-to-one assignment and singleton-aware thresholding, then executes the official submission validator:
```bash
python -m src.make_submission --config configs/pipeline.yaml --variant 1
```

Or run all stages in sequence:
```bash
python -m src.pipeline
```

---

## 4. Submission Validation
Verify submission validity against official challenge constraints:
```bash
python dataset/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Exit code 0 confirms valid submission format.
