# Preprocessing — how to run (Member 2)

Everything below runs from the **repository root**. `<DATASET_DIR>` is the
challenge `dataset/` folder (contains `train/` and `test/`), e.g.
`C:\Users\kusha\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset`.
Nothing under `data/` is committed (see `.gitignore`).

## Modules

| path | what it does |
|---|---|
| `src/io/data_loader.py` | safe TSV reading (tab, UTF-8, **quoting off**, empty strings kept), chunked iterator `iter_tsv`, `ParquetAppender`, `ground_truth_pairs` |
| `src/io/benchmark.py` | load-engine benchmark (pandas / chunked / pyarrow / polars / duckdb / parquet) with peak RAM per engine, plus normaliser throughput → `docs/BENCHMARK.md` |
| `src/preprocessing/normalize.py` | `Normalizer`: Unicode-safe canonicalisation → 14 new fields (see `DATA_DICTIONARY.md`); originals untouched |
| `src/preprocessing/translit_vocab.py` | learns Indian-script → English word map from training matches (co-occurrence + sound-alike) → `configs/translit_vocab.json` |
| `src/preprocessing/preprocess.py` | TSV → normalised Parquet for all splits/sources, multi-core, constant memory → `data/processed/` |
| `src/preprocessing/keys.py` | shared blocking-key definitions + stable 64-bit key hash |
| `src/preprocessing/build_indexes.py` | compact integer Parquet indexes per key (`data/processed/<split>/index/`), oversized keys dropped; `candidates()` reads them back for Member 4 |
| `src/preprocessing/blocking_keys.py` | blocking-key analysis: block sizes, oversized keys, duplicates, pair recall per key and greedy union → `docs/BLOCKING_KEYS.md` |
| `src/preprocessing/canonicalize.py` | Phase-1 baseline normaliser (kept for `src/pipeline.py`; superseded by `normalize.py`) |
| `configs/lexicons.json` | static word lists (legal forms, abbreviations, stop words, placeholders, US/IN states, FR regions/departments) — regenerate with `scripts/build_lexicons.py` |
| `scripts/eda.py` | the full-data EDA behind `EDA_REPORT.md` |
| `scripts/eval_normalization.py` | similarity of true / false pairs before vs after normalisation |
| `tests/test_preprocessing.py` | 37 unit tests |

## Run order

```bash
pip install -r requirements.txt
python -m pytest -q tests/test_preprocessing.py                                   # ~1 s

python scripts/eda.py --data-dir <DATASET_DIR>                                    # ~15 min, optional
python -m src.io.benchmark --file <DATASET_DIR>/train/train_source2.tsv           # 5-10 min
python -m src.preprocessing.translit_vocab --data-dir <DATASET_DIR>               # 5-10 min
python -m src.preprocessing.preprocess --data-dir <DATASET_DIR>                   # ~20 min (11 workers)
python -m src.preprocessing.blocking_keys --processed data/processed              # 15-30 min
python -m src.preprocessing.build_indexes --processed data/processed              # 15-30 min
```

On Windows use the Python that has the packages, e.g. `py -3.12 -m ...`.
Out of memory? `preprocess --chunksize 100000 --workers 2`; `blocking_keys --other-frac 0.5 --sample-s1 10000`.

## Measured on the full data (laptop, Ryzen 5 5600H 12 threads, 16 GB)

| step | result |
|---|---|
| EDA | 13.3 min, pandas, all train + test |
| preprocess | 24,229,173 rows in 1,134 s (18.9 min) with 11 workers; 3.1 GB Parquet (zstd) + 0.1 GB ground truth |

| file | rows | seconds | Parquet MB |
|---|---|---|---|
| train/source1 | 2,206,821 | 95.5 | 278.6 |
| train/source2 | 5,034,616 | 222.6 | 648.2 |
| train/source3 | 5,285,603 | 281.4 | 669.0 |
| test/source1 | 1,732,544 | 81.0 | 221.2 |
| test/source2 | 4,887,273 | 233.5 | 645.2 |
| test/source3 | 5,082,316 | 213.1 | 653.4 |

## Using the processed data

```python
import pandas as pd
s1 = pd.read_parquet("data/processed/train/source1.parquet")
s2 = pd.read_parquet("data/processed/train/source2.parquet",
                     columns=["entity_id", "country", "name_core", "addr_core", "addr_nums", "addr_state"])

from src.preprocessing.normalize import Normalizer
Normalizer().record("Nexnex t/a Raka Motors Pvt. Ltd.", "S/No 74/3, Satara, MH", "India")
```

## Rules followed

* Raw and normalised fields are kept side by side (D003).
* Non-ASCII text is never deleted: Indian scripts are mapped with the learned vocabulary or
  transliterated with `anyascii` (ISC licence); accents are folded only in derived fields.
* `country` is an open set — unknown labels fall back to all lexicons (France included).
* No external data or lookup services: lexicons are static lists; the vocabulary is learned
  only from the provided training matches.
* No model thresholds are set or changed here.
