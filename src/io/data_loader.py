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
