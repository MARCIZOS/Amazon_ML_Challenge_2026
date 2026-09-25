"""Entity-level match aggregation and output formatting."""

from collections import defaultdict


def aggregate_matches(
    scored_pairs: list[tuple[str, str, bool, float]],
    all_s1_ids: set[str],
) -> dict[str, list[str]]:
    """Aggregate pairwise match decisions into per-entity match lists.

    Args:
        scored_pairs: List of (s1_id, s2s3_id, is_match, score)
        all_s1_ids: Complete set of S1 entity IDs (ensures every S1 appears)

    Returns:
        Dict mapping s1_id -> sorted list of matched s2s3 entity_ids.
        Includes empty lists for singletons.
    """
    matches = defaultdict(list)

    for s1_id, s2s3_id, is_match, score in scored_pairs:
        if is_match:
            matches[s1_id].append((s2s3_id, score))

    # Build final result ensuring every S1 ID is present
    result = {}
    for s1_id in all_s1_ids:
        if s1_id in matches:
            # Sort by score descending, take IDs
            sorted_matches = sorted(matches[s1_id], key=lambda x: -x[1])
            # Deduplicate (shouldn't happen but safety)
            seen = set()
            unique_ids = []
            for mid, _ in sorted_matches:
                if mid not in seen:
                    seen.add(mid)
                    unique_ids.append(mid)
            result[s1_id] = unique_ids
        else:
            result[s1_id] = []

    return result
