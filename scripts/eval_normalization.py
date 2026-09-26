"""Measure how much normalisation helps: similarity of TRUE matched pairs and of
non-matching pairs, before (lowercase only) vs after (normalised fields).

A good normaliser raises similarity for true matches while keeping
non-matches low. Works on the EDA sample_groups.tsv (group format).

  python scripts/eval_normalization.py --groups data/interim/eda_output/sample_groups.tsv [--vocab configs/translit_vocab.json]
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402

from src.io.data_loader import read_tsv  # noqa: E402
from src.preprocessing.normalize import Normalizer, normalize_frame  # noqa: E402


def tset(a, b):
    return fuzz.token_set_ratio(a, b) / 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--vocab", default=None, help="translit vocab json ('' = none)")
    args = ap.parse_args()

    df = read_tsv(args.groups)
    nz = Normalizer(vocab_path=args.vocab if args.vocab is not None else None)
    df = normalize_frame(df, nz)
    df["raw_name"] = df["business_name"].str.lower()
    df["raw_addr"] = df["business_address"].str.lower()
    rec = df.set_index("entity_id")
    s1 = df[df.entity_id.str.startswith("S1-")]
    others = df[~df.entity_id.str.startswith("S1-")]
    s1_of = dict(zip(s1.group, s1.entity_id))

    pos = [(s1_of[g], e) for g, e in zip(others.group, others.entity_id) if g in s1_of]
    random.seed(0)
    ids = list(others.entity_id)
    neg = []
    for a, _ in pos:
        b = random.choice(ids)
        if rec.at[b, "group"] != rec.at[a, "group"] and rec.at[a, "country"] == rec.at[b, "country"]:
            neg.append((a, b))

    def score(pairs, label):
        rows = []
        for a, b in pairs:
            A, B = rec.loc[a], rec.loc[b]
            name_after = max(tset(A.name_core, B.name_core),
                             tset(A.name_core, B.name_alt) if B.name_alt else 0)
            rows.append({
                "label": label,
                "native": int(B.name_translit),
                "name_before": tset(A.raw_name, B.raw_name),
                "name_after": name_after,
                "name_key": tset(A.name_key, B.name_key),
                "addr_before": tset(A.raw_addr, B.raw_addr),
                "addr_after": tset(A.addr_core, B.addr_core),
                "addr_key": tset(A.addr_key, B.addr_key),
                "nums_overlap": float(bool(set(A.addr_nums.split()) & set(B.addr_nums.split()))),
                "same_state": float(A.addr_state != "" and A.addr_state == B.addr_state),
                "legal_eq": float(A.name_legal == B.name_legal),
            })
        return rows

    res = pd.DataFrame(score(pos, "match") + score(neg, "non-match"))
    cols = ["name_before", "name_after", "name_key", "addr_before", "addr_after", "addr_key",
            "nums_overlap", "same_state", "legal_eq"]
    pd.set_option("display.width", 200)
    print("\nMean similarity (true matches should go UP, non-matches stay LOW)")
    print(res.groupby("label")[cols].mean().round(3).T.to_string())
    print("\nTrue matches where the other name is in an Indian script:")
    print(res[(res.label == "match") & (res.native == 1)][cols].mean().round(3).to_string())
    m = res[res.label == "match"]
    print("\nShare of true matches with name similarity >= 0.9:  before "
          f"{(m.name_before >= .9).mean():.3f}  after {(m.name_after >= .9).mean():.3f}")
    print(f"pairs: {len(pos)} matches, {len(neg)} non-matches")


if __name__ == "__main__":
    main()
