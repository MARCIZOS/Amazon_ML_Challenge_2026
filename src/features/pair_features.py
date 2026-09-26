"""Vectorised pairwise features (Member 1).

All string similarities use rapidfuzz.process.cpdist (element-wise, C++, all
cores), so tens of millions of pairs are feasible. Inputs are the normalised
fields written by src.preprocessing (see docs/DATA_DICTIONARY.md). No country
one-hot is used, so the model transfers to France (unseen in training).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

S1_COLS = ["name_core", "name_key", "name_compact", "name_legal", "addr_core", "addr_key",
           "addr_nums", "addr_state", "addr_missing"]
O_COLS = ["name_core", "name_key", "name_compact", "name_alt", "name_legal", "name_translit",
          "addr_core", "addr_key", "addr_nums", "addr_state", "addr_missing"]


def _l(x):
    return x.tolist() if isinstance(x, np.ndarray) else list(x)


def _cp(a, b, scorer, scale=100.0):
    a, b = _l(a), _l(b)
    if not a:
        return np.empty(0, np.float32)
    return (cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / scale).astype(np.float32)


def fast_similarity(n1, n2, a1, a2):
    """Cheap name / address token-set similarity used for candidate re-ranking."""
    return _cp(n1, n2, fuzz.token_set_ratio), _cp(a1, a2, fuzz.token_set_ratio)


def _set_features(x1, x2):
    """jaccard, any-overlap, first-equal, both-present for space-separated token strings."""
    n = len(x1)
    jac = np.zeros(n, np.float32)
    anyo = np.zeros(n, np.int8)
    first = np.zeros(n, np.int8)
    both = np.zeros(n, np.int8)
    for i, (a, b) in enumerate(zip(x1, x2)):
        if a and b:
            sa, sb = a.split(), b.split()
            both[i] = 1
            first[i] = sa[0] == sb[0]
            A, B = set(sa), set(sb)
            inter = len(A & B)
            anyo[i] = inter > 0
            jac[i] = inter / len(A | B)
    return jac, anyo, first, both


def _len_tokens(x):
    return np.fromiter((len(s.split()) if s else 0 for s in x), dtype=np.int16, count=len(x))


def pair_features(s1: dict, o: dict, n_keys, pre_score, src) -> pd.DataFrame:
    """s1 / o: dicts of aligned arrays (one entry per candidate pair)."""
    f = {}
    n1, n2 = s1["name_core"], o["name_core"]
    f["name_ratio"] = _cp(n1, n2, fuzz.ratio)
    f["name_tset"] = _cp(n1, n2, fuzz.token_set_ratio)
    f["name_tsort"] = _cp(n1, n2, fuzz.token_sort_ratio)
    f["name_partial"] = _cp(n1, n2, fuzz.partial_ratio)
    f["name_jw"] = _cp(n1, n2, JaroWinkler.normalized_similarity, scale=1.0)
    f["name_key_tset"] = _cp(s1["name_key"], o["name_key"], fuzz.token_set_ratio)
    f["name_key_ratio"] = _cp(s1["name_key"], o["name_key"], fuzz.ratio)
    f["compact_ratio"] = _cp(s1["name_compact"], o["name_compact"], fuzz.ratio)
    f["compact_partial"] = _cp(s1["name_compact"], o["name_compact"], fuzz.partial_ratio)
    has_alt = np.fromiter((bool(x) for x in o["name_alt"]), dtype=bool, count=len(n1))
    f["alt_tset"] = np.where(has_alt, _cp(n1, o["name_alt"], fuzz.token_set_ratio), -1).astype(np.float32)
    jac, anyo, first, _ = _set_features(n1, n2)
    f["name_jacc"], f["name_any"], f["name_first_eq"] = jac, anyo, first
    f["name_exact"] = (np.asarray(n1, dtype=object) == np.asarray(n2, dtype=object)).astype(np.int8)
    l1, l2 = _len_tokens(n1), _len_tokens(n2)
    f["name_len_s1"], f["name_len_o"] = l1, l2
    f["name_len_diff"] = np.abs(l1 - l2).astype(np.int16)

    lg1, lg2 = s1["name_legal"], o["name_legal"]
    ljac, lany, _, lboth = _set_features(lg1, lg2)
    f["legal_jacc"], f["legal_both"] = ljac, lboth
    f["legal_conflict"] = ((lboth == 1) & (lany == 0)).astype(np.int8)
    f["o_translit"] = np.asarray(o["name_translit"], dtype=np.int8)

    a1, a2 = s1["addr_core"], o["addr_core"]
    f["addr_tset"] = _cp(a1, a2, fuzz.token_set_ratio)
    f["addr_ratio"] = _cp(a1, a2, fuzz.ratio)
    f["addr_partial"] = _cp(a1, a2, fuzz.partial_ratio)
    f["addr_key_tset"] = _cp(s1["addr_key"], o["addr_key"], fuzz.token_set_ratio)
    ajac, _, _, _ = _set_features(a1, a2)
    f["addr_jacc"] = ajac
    njac, nany, nfirst, nboth = _set_features(s1["addr_nums"], o["addr_nums"])
    f["nums_jacc"], f["nums_any"], f["nums_first_eq"], f["nums_both"] = njac, nany, nfirst, nboth
    f["nums_conflict"] = ((nboth == 1) & (nany == 0)).astype(np.int8)
    st1 = np.asarray(s1["addr_state"], dtype=object)
    st2 = np.asarray(o["addr_state"], dtype=object)
    both_st = (st1 != "") & (st2 != "")
    f["state_both"] = both_st.astype(np.int8)
    f["state_eq"] = (both_st & (st1 == st2)).astype(np.int8)
    f["state_conflict"] = (both_st & (st1 != st2)).astype(np.int8)
    f["addr_missing_s1"] = np.asarray(s1["addr_missing"], dtype=np.int8)
    f["addr_missing_o"] = np.asarray(o["addr_missing"], dtype=np.int8)

    f["n_keys"] = np.asarray(n_keys, dtype=np.int16)
    f["pre_score"] = np.asarray(pre_score, dtype=np.float32)
    f["src3"] = (np.asarray(src) == 3).astype(np.int8)
    return pd.DataFrame(f)


def add_group_features(df: pd.DataFrame, group_col: str = "s1_row") -> pd.DataFrame:
    """Ranks / gaps within the candidate list of each S1 record."""
    g = df.groupby(group_col, sort=False)
    for c in ("name_tset", "addr_tset", "pre_score"):
        mx = g[c].transform("max")
        df[f"{c}_gap"] = (mx - df[c]).astype(np.float32)
        df[f"{c}_rank"] = g[c].rank(ascending=False, method="min").astype(np.float32)
    df["n_cands"] = g[group_col].transform("size").astype(np.int16)
    return df


FEATURES = None  # filled lazily by feature_names()


def feature_names(df: pd.DataFrame) -> list[str]:
    drop = {"s1_row", "cand", "s1_id", "other_id", "label"}
    return [c for c in df.columns if c not in drop]
