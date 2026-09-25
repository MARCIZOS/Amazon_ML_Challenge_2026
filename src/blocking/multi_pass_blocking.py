"""Multi-pass blocking for candidate pair generation.

Country-aware blocking with multiple passes. Candidates are the
deduplicated union across all passes. Per Decision D002.

Blocking strategy (baseline):
  Pass 1: Exact normalized name match (within country)
  Pass 2: Exact sorted-token name key (within country)
  Pass 3: First name token + first 3 chars of address (within country)

Each pass builds an inverted index over S2+S3 records, then looks up
each S1 record. This is memory-efficient: indexes are per-country and
per-pass, discarded after use.
"""

from collections import defaultdict

from tqdm import tqdm


def _build_index(records: list[tuple[str, str]], key_fn) -> dict[str, list[str]]:
    """Build inverted index: blocking_key -> list of entity_ids."""
    index = defaultdict(list)
    for entity_id, key_value in records:
        if key_value:
            bk = key_fn(key_value) if key_fn else key_value
            if bk:
                index[bk].append(entity_id)
    return index


def generate_candidates(
    s1_df,
    s2s3_df,
    max_candidates_per_entity: int = 200,
) -> dict[str, set[str]]:
    """Generate candidate pairs using multi-pass blocking within country.
    
    Args:
        s1_df: pandas DataFrame of S1 records
        s2s3_df: pandas DataFrame of S2/S3 records
    """
    candidates = defaultdict(set)
    
    # Process S1 by country
    countries = s1_df["country"].unique()
    
    for country in tqdm(countries, desc="Blocking by country"):
        s1_country = s1_df[s1_df["country"] == country]
        s2s3_country = s2s3_df[s2s3_df["country"] == country]
        
        if s2s3_country.empty:
            continue

        # Convert to list of namedtuples for fast iteration without huge dict overhead
        s1_iter = list(s1_country.itertuples(index=False))
        s2s3_iter = list(s2s3_country.itertuples(index=False))

        # ---- Pass 1: Exact normalized name ----
        name_index = defaultdict(list)
        for row in s2s3_iter:
            if row.name_norm:
                name_index[row.name_norm].append(row.entity_id)

        for row in s1_iter:
            if row.name_norm and row.name_norm in name_index:
                candidates[row.entity_id].update(name_index[row.name_norm])

        # ---- Pass 2: Sorted-token name key ----
        key_index = defaultdict(list)
        for row in s2s3_iter:
            if row.name_key:
                key_index[row.name_key].append(row.entity_id)

        for row in s1_iter:
            if row.name_key and row.name_key in key_index:
                candidates[row.entity_id].update(key_index[row.name_key])

        # ---- Pass 3: First token match ----
        ft_index = defaultdict(list)
        for row in s2s3_iter:
            ft = row.first_token
            if ft and len(ft) >= 3:
                ft_index[ft].append(row.entity_id)

        for row in s1_iter:
            ft = row.first_token
            if ft and len(ft) >= 3 and ft in ft_index:
                block = ft_index[ft]
                if len(block) <= 500:
                    candidates[row.entity_id].update(block)

        # ---- Pass 4: Postcode match ----
        pc_index = defaultdict(list)
        for row in s2s3_iter:
            if row.postcode:
                pc_index[row.postcode].append(row.entity_id)

        for row in s1_iter:
            if row.postcode and row.postcode in pc_index:
                block = pc_index[row.postcode]
                if len(block) <= 500:
                    candidates[row.entity_id].update(block)

        # Cleanup per-country indexes
        del name_index, key_index, ft_index, pc_index

    # Apply cap per entity
    if max_candidates_per_entity:
        for s1_id in candidates:
            if len(candidates[s1_id]) > max_candidates_per_entity:
                candidates[s1_id] = set(sorted(candidates[s1_id])[:max_candidates_per_entity])

    return dict(candidates)


