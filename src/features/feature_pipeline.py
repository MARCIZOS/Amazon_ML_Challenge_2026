"""Pairwise feature extraction for candidate pairs.

Computes lightweight string similarity features using rapidfuzz.
"""

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


def compute_pair_features(
    name1: str, addr1: str,
    name2: str, addr2: str,
) -> dict:
    """Compute similarity features for a candidate pair.

    Args:
        name1, addr1: Normalized name and address of S1 entity.
        name2, addr2: Normalized name and address of S2/S3 entity.

    Returns:
        Dict of feature name -> float value.
    """
    # Name features
    name_ratio = fuzz.ratio(name1, name2) / 100.0 if (name1 and name2) else 0.0
    name_tset = fuzz.token_set_ratio(name1, name2) / 100.0 if (name1 and name2) else 0.0
    name_tsort = fuzz.token_sort_ratio(name1, name2) / 100.0 if (name1 and name2) else 0.0
    name_partial = fuzz.partial_ratio(name1, name2) / 100.0 if (name1 and name2) else 0.0
    name_jw = JaroWinkler.similarity(name1, name2) if (name1 and name2) else 0.0

    # Name token overlap
    t1 = set(name1.split()) if name1 else set()
    t2 = set(name2.split()) if name2 else set()
    name_jaccard = len(t1 & t2) / len(t1 | t2) if (t1 or t2) else 0.0

    # Address features
    addr_ratio = fuzz.ratio(addr1, addr2) / 100.0 if (addr1 and addr2) else 0.0
    addr_tset = fuzz.token_set_ratio(addr1, addr2) / 100.0 if (addr1 and addr2) else 0.0
    addr_tsort = fuzz.token_sort_ratio(addr1, addr2) / 100.0 if (addr1 and addr2) else 0.0

    # Address token overlap
    a1 = set(addr1.split()) if addr1 else set()
    a2 = set(addr2.split()) if addr2 else set()
    addr_jaccard = len(a1 & a2) / len(a1 | a2) if (a1 or a2) else 0.0

    return {
        "name_ratio": name_ratio,
        "name_tset": name_tset,
        "name_tsort": name_tsort,
        "name_partial": name_partial,
        "name_jw": name_jw,
        "name_jaccard": name_jaccard,
        "addr_ratio": addr_ratio,
        "addr_tset": addr_tset,
        "addr_tsort": addr_tsort,
        "addr_jaccard": addr_jaccard,
    }


def build_features(candidate_pairs, reference_data, source_data):
    """Compute features for all candidate pairs.

    Args:
        candidate_pairs: dict of s1_id -> set of s2s3_ids
        reference_data: dict of entity_id -> (name_norm, addr_norm)
        source_data: dict of entity_id -> (name_norm, addr_norm)

    Returns:
        List of (s1_id, s2s3_id, features_dict)
    """
    results = []
    for s1_id, s2s3_ids in candidate_pairs.items():
        n1, a1 = reference_data.get(s1_id, ("", ""))
        for s2s3_id in s2s3_ids:
            n2, a2 = source_data.get(s2s3_id, ("", ""))
            feats = compute_pair_features(n1, a1, n2, a2)
            results.append((s1_id, s2s3_id, feats))
    return results
