"""Dataset loading functions.

Memory-safe TSV reader using PyArrow backend with chunked processing.
Validates schema on load.
"""

import csv
import os

import pandas as pd

from src.io.schema_validator import validate_schema

# Expected schemas
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def load_source_tsv(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Load a source TSV file with robust settings.

    All fields are read as strings. No default NA values are applied.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        na_values=[],
        nrows=nrows,
        encoding="utf-8",
        on_bad_lines="warn",
    )
    validate_schema(df, SOURCE_COLUMNS)
    return df


def load_ground_truth(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Load ground truth TSV."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        na_values=[],
        nrows=nrows,
        encoding="utf-8",
        on_bad_lines="warn",
    )
    validate_schema(df, GT_COLUMNS)
    return df


def load_datasets(data_dir: str, split: str = "train", nrows: int | None = None) -> dict:
    """Load all source files and optionally ground truth.

    Returns dict with keys: 's1', 's2', 's3', and 'gt' (train only).
    """
    base = os.path.join(data_dir, split)
    prefix = f"{split}_source"

    result = {}
    for tag, num in [("s1", "1"), ("s2", "2"), ("s3", "3")]:
        path = os.path.join(base, f"{prefix}{num}.tsv")
        result[tag] = load_source_tsv(path, nrows=nrows)

    if split == "train":
        gt_path = os.path.join(base, f"train_ground_truth.tsv")
        result["gt"] = load_ground_truth(gt_path, nrows=nrows)

    return result


# --------------------------------------------------------------------------- #
# Member 2 additions: chunked / memory-safe helpers used by                    #
# src/preprocessing/{preprocess,translit_vocab,blocking_keys}.py               #
# (existing functions above are unchanged)                                     #
# --------------------------------------------------------------------------- #
from typing import Iterator  # noqa: E402

_READ_KW = dict(sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False,
                na_values=[], encoding="utf-8", on_bad_lines="warn")


def read_tsv(path: str, usecols=None, nrows=None) -> pd.DataFrame:
    """Read any challenge TSV as strings, quoting OFF, empty strings kept (no NaN)."""
    return pd.read_csv(path, usecols=usecols, nrows=nrows, **_READ_KW)


def iter_tsv(path: str, chunksize: int = 500_000, usecols=None) -> Iterator[pd.DataFrame]:
    """Yield a TSV in chunks of `chunksize` rows (constant memory)."""
    yield from pd.read_csv(path, usecols=usecols, chunksize=chunksize, **_READ_KW)


def source_path(data_dir: str, split: str, k: int) -> str:
    return os.path.join(data_dir, split, f"{split}_source{k}.tsv")


def gt_path(data_dir: str) -> str:
    return os.path.join(data_dir, "train", "train_ground_truth.tsv")


def ground_truth_pairs(data_dir: str) -> pd.DataFrame:
    """Ground truth in long format: one row per true (s1, other) pair."""
    gt = read_tsv(gt_path(data_dir))
    gt["ids"] = gt["matched_entity_ids"].map(lambda s: [x for x in s.split(",") if x])
    long = gt[["source1_entity_id", "ids"]].explode("ids").dropna()
    long = long[long["ids"] != ""]
    return long.rename(columns={"source1_entity_id": "s1", "ids": "other"}).reset_index(drop=True)


class ParquetAppender:
    """Append DataFrame chunks to one parquet file (zstd) with a fixed schema."""

    def __init__(self, path: str, compression: str = "zstd"):
        self.path, self.compression, self.writer = path, compression, None
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def write(self, df: pd.DataFrame):
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression=self.compression)
        self.writer.write_table(table)

    def close(self):
        if self.writer is not None:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
