# Phase 0 — Inspection Report

## System Resources
| Resource | Value |
|---|---|
| CPU | AMD Ryzen 5 5600H, 6C/12T |
| RAM | 16 GB total, ~4 GB free |
| Python | 3.14.3 |
| Key packages | pandas 2.2.2, pyarrow 23.0.1, rapidfuzz 3.14.6, scikit-learn 1.8.0, unidecode 1.4.0, numpy 2.2.6 |
| Missing | duckdb (installing), pyyaml (installing) |

## Dataset Summary

| File | Rows | Size | Notes |
|---|---|---|---|
| train_source1.tsv | 2,206,821 | 200 MB | US + India, all fields populated |
| train_source2.tsv | 5,034,616 | 467 MB | 169K empty addresses, 15% non-ASCII names, 25K dup name+addr |
| train_source3.tsv | 5,285,603 | 480 MB | 176K empty addresses, 11.5% non-ASCII names, 19K dup name+addr |
| train_ground_truth.tsv | 2,206,821 | 121 MB | 1:1 with S1 |
| test_source1.tsv | 1,732,544 | 167 MB | US + India + **France** (259K) |
| test_source2.tsv | 4,887,273 | 486 MB | US + India + France (703K) |
| test_source3.tsv | 5,082,316 | 483 MB | US + India + France (732K) |

**Total data: ~2.4 GB on disk. Cannot fit all in memory simultaneously on 16GB RAM.**

### Schema (all files)
- `entity_id`, `business_name`, `business_address`, `country`
- Ground truth: `source1_entity_id`, `matched_entity_ids` (comma-separated)
- Tab-separated, UTF-8
- No duplicate entity_ids within any source
- S1 has zero empty fields; S2/S3 have ~3% empty addresses

### Key Ground Truth Statistics
- **Singleton rate: 5.58%** (123K entities with no match)
- Mean matches per S1: 3.46 (non-singletons: 3.67)
- Max matches for one S1: 11
- **S2/S3 records map to at most 1 S1** (many-to-one = 0) — critical constraint!
- S2: 73.4% matched, S3: 74.6% matched; rest are distractors
- Country is always equal in matched pairs (100%)

### Key Similarity Statistics (from EDA)
| Feature | True Match | Hard Neg (same postcode) | Random Neg |
|---|---|---|---|
| name_exact | 0.257 | 0.000 | 0.000 |
| name_token_set | 0.908 | 0.334 | 0.361 |
| name_jaccard | 0.645 | 0.016 | 0.031 |
| first_tok_equal | 0.731 | 0.003 | 0.001 |
| addr_token_set | 0.874 | 0.482 | 0.359 |

### Blocking Signal Recall (from EDA)
- Same country: 100% ← **must block by country**
- Share any name token: 86.16%
- Same first name token: 73.07%
- Exact normalised name: 25.75%
- Same postcode (when present): 71.49% (but only 5-12% of records have postcodes)

## Memory Strategy

Loading all ~12M rows × 4 columns into pandas ≈ 3-4GB. With normalized fields ≈ 6-8GB.
**Strategy: Use DuckDB for disk-backed blocking index construction. Process S1 in chunks.**

## Existing Codebase Status

**All src/ modules are empty stubs.** The repository has good structure but zero working code:
- `src/config.py` — `NotImplementedError`
- `src/pipeline.py` — placeholder print
- `src/io/data_loader.py` — `NotImplementedError`
- `src/blocking/multi_pass_blocking.py` — `NotImplementedError`
- `src/features/feature_pipeline.py` — `NotImplementedError`
- `src/matching/model.py` — `NotImplementedError`
- `src/matching/aggregation.py` — `NotImplementedError`
- `src/preprocessing/canonicalize.py` — returns data unchanged
- `src/io/schema_validator.py` — basic column check (works)
- `src/io/output_writer.py` — basic `to_csv` (works)
- Tests: 1 smoke test (passes), 2 schema validator tests

## Phase 1 Implementation Plan

### Approach: Conservative precision-first baseline
F0.5 penalizes false merges heavily. The baseline should match only when very confident.

### Pipeline:
1. **Data loading** — PyArrow-backed pandas reader with chunked processing
2. **Normalization** — Unicode NFKD → unidecode (for blocking keys only), lowercase, strip legal suffixes, normalize whitespace. Keep original fields.
3. **Blocking** — Country partition + multi-pass:
   - Pass 1: Exact normalized name (catches 25.7% of pairs, very high precision)
   - Pass 2: First-3-chars-of-name + first-token (catches more, still safe)
   - Pass 3: Name n-gram overlap (careful, controlled explosion)
4. **Candidate scoring** — For each blocked pair:
   - name_token_set_ratio (rapidfuzz)
   - name_ratio (rapidfuzz)
   - address_token_set_ratio (rapidfuzz)
5. **Match decision** — Conservative threshold: name_token_set ≥ 0.70 AND (addr_token_set ≥ 0.50 OR addr is empty)
6. **Aggregation** — Per-S1 entity, collect all matched S2/S3 IDs
7. **Output** — TSV with proper formatting

### Memory-safe execution:
- Build S2/S3 normalized lookup tables per-country using DuckDB or dict-of-dicts
- Process S1 entities in chunks of ~100K
- Build blocking indexes per-country to limit block sizes
