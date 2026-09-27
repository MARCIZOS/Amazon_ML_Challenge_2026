"""Blocking-key definitions shared by blocking_keys.py (analysis) and
build_indexes.py (compact on-disk indexes).

Every key function takes one normalised record (a namedtuple with the columns
it lists in KEY_COLUMNS) and returns a list of key strings. Keys are always
prefixed with the country label by `explode_keys`, so records are never
compared across countries (100% of true pairs share the country label).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLS = ["entity_id", "country", "name_core", "name_key", "name_compact", "name_alt",
        "addr_core", "addr_key", "addr_nums", "addr_state"]


def k_name_tokens(r):            # every name word (incl. the pre-marker trade name)
    return r.name_core.split() + r.name_alt.split()


def k_name_first(r):             # first word of the name
    return r.name_core.split()[:1]


def k_name_skel_tokens(r):       # phonetic skeleton of every distinctive name word (>= 3 chars)
    return [t for t in r.name_key.split() if len(t) >= 3]


def k_name_skel_pair(r):         # pairs among the first three skeleton words, order-free
    t = sorted(r.name_key.split()[:3])
    if len(t) >= 2:
        return [a + "_" + b for i, a in enumerate(t) for b in t[i + 1:]]
    return t[:1]


def k_name_prefix(r):            # first 4 characters of the compact name (fixed length for true equality)
    return [r.name_compact[:4]] if len(r.name_compact) >= 4 else ([r.name_compact] if len(r.name_compact) >= 3 else [])


def k_addr_tokens(r):            # every address word (phonetic)
    return [t for t in r.addr_key.split() if len(t) >= 3]


def k_num_street(r):             # house number (incl. single digit) + address word
    # Support 1+ digit numbers for European (France) and Indian addresses (e.g. 5 rue ...)
    nums = [n for n in r.addr_nums.split() if len(n) >= 1][:3]
    words = [t for t in r.addr_key.split() if len(t) >= 3][:6]
    return [n + "|" + w for n in nums for w in words]


def k_state_name_skel(r):        # state + first name skeleton word
    t = r.name_key.split()
    return [r.addr_state + "|" + t[0]] if (t and r.addr_state) else []


KEYS = {
    "name_tokens": k_name_tokens,
    "name_first": k_name_first,
    "name_skel_tokens": k_name_skel_tokens,
    "name_skel_pair": k_name_skel_pair,
    "name_prefix5": k_name_prefix,
    "addr_tokens": k_addr_tokens,
    "num_street": k_num_street,
    "state_name_skel": k_state_name_skel,
}

# columns each key needs (plus entity_id + country, always read)
KEY_COLUMNS = {
    "name_tokens": ["name_core", "name_alt"],
    "name_first": ["name_core"],
    "name_skel_tokens": ["name_key"],
    "name_skel_pair": ["name_key"],
    "name_prefix5": ["name_compact"],
    "addr_tokens": ["addr_key"],
    "num_street": ["addr_nums", "addr_key"],
    "state_name_skel": ["addr_state", "name_key"],
}


def stable_hash(strings) -> np.ndarray:
    """Stable 64-bit hash (same value on every run / machine), vectorised.

    Python's built-in hash() is salted per process, so it must not be used for
    anything written to disk.
    """
    arr = np.asarray(strings, dtype=object)
    if arr.size == 0:
        return np.empty(0, dtype=np.uint64)
    return pd.util.hash_array(arr, categorize=False)


def explode_keys(df: pd.DataFrame, fn, id_map: dict | None = None) -> pd.DataFrame:
    """One row per (record, distinct key): columns rid (int32), key (uint64).

    rid is id_map[entity_id] when a map is given, else the row position in df.
    Keys are 'country#key' strings hashed with stable_hash.
    """
    rids, keys = [], []
    for pos, r in enumerate(df.itertuples(index=False)):
        ks = set(fn(r))
        if ks:
            pre = r.country + "#"
            rid = id_map[r.entity_id] if id_map is not None else pos
            for k in ks:
                rids.append(rid)
                keys.append(pre + k)
    return pd.DataFrame({"rid": np.asarray(rids, dtype=np.int32), "key": stable_hash(keys)})


def key_strings(df: pd.DataFrame, fn, wanted: set) -> dict:
    """Recover readable key strings for a few hashes (for reports)."""
    out = {}
    for r in df.itertuples(index=False):
        ks = [r.country + "#" + k for k in fn(r)]
        for k, h in zip(ks, stable_hash(ks)):
            h = int(h)
            if h in wanted and h not in out:
                out[h] = k
        if len(out) == len(wanted):
            break
    return out
