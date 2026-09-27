"""Tests for evaluation, decision rules, output files and blocking/feature plumbing."""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.evaluation.metrics import evaluate, f05  # noqa: E402
from src.io.output_writer import (CAND_HEADER, MATCH_HEADER, check_submission,  # noqa: E402
                                  write_id_lists)
from src.matching.aggregation import decide_pairs, pairs_to_lists  # noqa: E402


def test_f05_problem_statement_example():
    # predicted 3, 2 correct, 2 true -> 0.714
    assert f05([2], [3], [2])[0] == pytest.approx(0.7143, abs=1e-3)


def test_f05_singletons():
    assert f05([0], [0], [0])[0] == 1.0      # correct empty
    assert f05([0], [1], [0])[0] == 0.0      # false merge on a singleton
    assert f05([0], [0], [3])[0] == 0.0      # predicted empty, had matches


def test_evaluate_macro_includes_all_s1():
    true = pd.DataFrame({"s1_id": ["A", "A", "B"], "other_id": ["S2-1", "S3-1", "S2-2"]})
    pred = pd.DataFrame({"s1_id": ["A"], "other_id": ["S2-1"]})
    m = evaluate(pred, true, ["A", "B", "C"])       # C is a singleton predicted empty
    fa = f05([1], [1], [2])[0]
    assert m["macro_f05"] == pytest.approx((fa + 0 + 1) / 3)
    assert m["false_merges_on_singletons"] == 0


def test_decide_one_to_one_threshold_relative():
    sc = pd.DataFrame({"s1_id": ["A", "B", "A", "A", "C"], "other_id": ["X", "X", "Y", "Z", "W"],
                       "p": [0.9, 0.95, 0.8, 0.3, 0.55]})
    kept = decide_pairs(sc, threshold=0.5, relative=0.0)
    assert set(zip(kept.s1_id, kept.other_id)) == {("B", "X"), ("A", "Y"), ("C", "W")}   # X goes to its best S1 only
    kept = decide_pairs(sc, threshold=0.2, relative=0.5, one_to_one=False)
    assert ("A", "Z") not in set(zip(kept.s1_id, kept.other_id))             # 0.3 < 0.5 * 0.9
    # single_min_p drops C because C only has 1 match with p=0.55 < 0.65
    kept = decide_pairs(sc, threshold=0.5, relative=0.0, single_min_p=0.65)
    assert ("C", "W") not in set(zip(kept.s1_id, kept.other_id))
    assert ("A", "Y") in set(zip(kept.s1_id, kept.other_id))


def test_output_files_pass_checks():
    s1 = ["S1-1", "S1-2", "S1-3"]
    with tempfile.TemporaryDirectory() as d:
        m, c = os.path.join(d, "m.tsv"), os.path.join(d, "c.tsv")
        write_id_lists(c, CAND_HEADER, s1, {"S1-1": ["S2-1", "S3-9", "S2-1"], "S1-2": ["S3-2"]})
        write_id_lists(m, MATCH_HEADER, s1, pairs_to_lists(
            pd.DataFrame({"s1_id": ["S1-1"], "other_id": ["S2-1"]}), s1))
        assert check_submission(m, c, s1) == []
        with open(m, encoding="utf-8") as f:
            lines = f.read().splitlines()
        assert lines[0] == "source1_entity_id\tmatched_entity_ids"
        assert lines[1] == "S1-1\tS2-1" and lines[3] == "S1-3\t"
        # a match outside the candidate set is reported
        write_id_lists(m, MATCH_HEADER, s1, {"S1-2": ["S2-7"]})
        assert any("not in candidate" in p for p in check_submission(m, c, s1))


def test_blocking_and_features_end_to_end():
    from src.blocking.candidates import CandidateGenerator
    from src.preprocessing.build_indexes import build
    from src.preprocessing.preprocess import normalize_df_parallel
    from src.scoring import build_pairs
    rows = {
        1: [("S1-1", "Raka Motors Pvt Ltd", "74 Kesarkar Peth, Satara, MH", "India"),
            ("S1-2", "Acme Corp", "12 Main Street, Dallas, TX", "US"),
            ("S1-3", "Lonely Bakery", "9 Nowhere Lane, Reno, NV", "US")],
        2: [("S2-1", "RAKA MOTORS PRIVATE LIMITED", "74 KESARKAR PETH, SATARA", "India"),
            ("S2-2", "Raka Motors", "74 Kesarkar Peth", "US")],
        3: [("S3-1", "Acme Corporation", "12 Main St, Dallas, Texas", "US")],
    }
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "train"))
        for k, rs in rows.items():
            df = pd.DataFrame(rs, columns=["entity_id", "business_name", "business_address", "country"])
            normalize_df_parallel(df, workers=1, vocab_path="").to_parquet(
                os.path.join(d, "train", f"source{k}.parquet"), index=False)
        keys = {"name_first": 50, "num_street": 50, "name_skel_pair": 50}
        build(d, ("train",), keys)
        gen = CandidateGenerator(d, "train", keys, pre_k=10, top_k=5)
        meta, X = build_pairs(gen, np.arange(3))
        pairs = set(zip(meta.s1_id, meta.other_id))
        assert ("S1-1", "S2-1") in pairs and ("S1-2", "S3-1") in pairs
        assert ("S1-1", "S2-2") not in pairs                     # never across countries
        assert not any(s == "S1-3" for s, _ in pairs)             # no candidates -> none
        assert len(X) == len(meta) and np.isfinite(X.select_dtypes("number").to_numpy()).all()
        assert X.loc[(meta.s1_id == "S1-1").to_numpy(), "name_tset"].max() > 0.9
