# Baseline Characterization & Error Analysis (Phase 1)

**Date:** 2026-09-27  
**Audited Run:** `experiments/results/M20260927_1454/`  
**Test Score (Public Leaderboard):** 0.915344 macro F0.5  

---

## 1. Baseline Performance Summary

The performance of the 0.915344 baseline run on the 60,000 S1 validation set is summarized below:

| Metric | Baseline Value | Interpretation |
|---|---|---|
| **Blocking Recall** | **0.914337** (91.43%) | **189,681 / 207,452** true pairs captured |
| **Candidates per S1 (mean / max)** | **39.20 / 40** | Full candidate budget saturated (top_k=40) |
| **Reduction Ratio** | **0.9999962** | 2,351,837 pairs evaluated vs ~60,000 × 10.3M |
| **S1 without Candidates** | 84 (0.14%) | Entities with no key overlap |
| **Macro F0.5 (Overall)** | **0.930511** | Baseline score on validation |
| **Macro F0.5 (Singletons)** | **0.930597** | 3,314 singleton S1 entities (5.52% share) |
| **Macro F0.5 (Non-Singletons)** | **0.930506** | 56,686 non-singleton S1 entities |
| **Micro Precision / Recall** | **0.986139 / 0.856352** | High precision, significant recall loss |
| **False Merges on Singletons** | **230** | Costly: each false merge drops S1 score from 1.0 to 0.0 |
| **Ceiling with Perfect Matcher** | **0.964740** | Theoretical maximum F0.5 on current candidate set |

---

## 2. Root Cause: Blocking Gap vs Matching Gap

> [!IMPORTANT]
> **The primary ceiling is Blocking Recall.**  
> Even with a mathematically perfect matcher that achieves 100% precision and keeps 100% of available true pairs, the macro F0.5 ceiling is capped at **0.964740**.
> To hit the target of **0.98+**, blocking recall MUST be boosted to **≥ 0.985+** while simultaneously tightening candidate efficiency.

The secondary gap is **Matching Recall & Singleton False Merges**:
- The matcher drops recall from 91.43% down to 85.64% (a 5.79 percentage point loss).
- 230 singleton S1 records were erroneously matched with distractor candidates, completely zeroing out their score.

---

## 3. Feature Importance Analysis (Top 20 Features)

From `experiments/results/M20260927_1454/feature_importance.csv`:

| Rank | Feature | Importance Gain | Category / Function |
|---|---|---|---|
| 1 | `pre_score` | 5,592,118 | Cheap blocking similarity (fast_similarity + shared keys) |
| 2 | `nums_jacc` | 2,432,810 | Jaccard overlap of address numbers (crucial differentiator) |
| 3 | `pre_score_rank` | 2,142,882 | Relative rank of candidate within the S1's candidate pool |
| 4 | `name_partial` | 1,267,633 | Rapidfuzz partial ratio on core name |
| 5 | `pre_score_gap` | 907,919 | Difference between candidate pre_score and top candidate |
| 6 | `nums_conflict` | 647,057 | Both have numbers, but zero overlap (strong mismatch signal) |
| 7 | `name_tsort` | 518,827 | Token sort ratio on core name (handles word reordering) |
| 8 | `addr_jacc` | 439,570 | Address token Jaccard similarity |
| 9 | `legal_conflict` | 193,586 | Conflicting legal entity types (e.g., pvt vs ltd) |
| 10 | `addr_tset` | 192,356 | Address token-set ratio |
| 11 | `name_len_o` | 181,395 | Token length of candidate name |
| 12 | `addr_partial` | 162,834 | Partial ratio on address tokens |
| 13 | `compact_ratio` | 133,121 | Ratio on whitespace-stripped name |
| 14 | `state_conflict` | 124,052 | Conflicting admin area / state codes |
| 15 | `name_jacc` | 123,139 | Name token Jaccard similarity |
| 16 | `name_jw` | 122,919 | Jaro-Winkler similarity on name |
| 17 | `addr_key_tset` | 115,412 | Phonetic skeleton token-set ratio on address |
| 18 | `addr_ratio` | 110,050 | Full address string ratio |
| 19 | `name_len_diff` | 107,270 | Absolute difference in token counts |
| 20 | `legal_jacc` | 100,995 | Jaccard overlap of legal forms |

**Takeaway:** Group/relative features (`pre_score_rank`, `pre_score_gap`) and address number exactness (`nums_jacc`, `nums_conflict`) dominate the model's decision tree.

---

## 4. Error Taxonomy & Systematic Failure Modes

From exhaustive examination of `errors.tsv`:

1. **Branch / Numeric Distinction (False Merges):**
   - *Example:* `peak desert four` vs `peak desert twelve` at `styles bridge rd collinsville` (p=0.99 false merge).
   - *Cause:* Identical street address and high token-sort ratio dominate, while the distinct number token (`four` vs `twelve`) is treated as a minor mismatch.
   - *Fix:* Dedicated word-number conflict feature and number-token identity features.

2. **Empty / Sparse Address Distractors (False Merges):**
   - *Example:* `east silver constructions` with full address vs candidate with empty address `""` (p=0.899 false merge).
   - *Cause:* When address is missing, address features are neutral/imputed, allowing moderate name matches to pass the global threshold.
   - *Fix:* Penalize address missing when candidate name is not 100% exact, and add strict name agreement gates when address is absent.

3. **Honorific / Prefix Word Differences (Missed Matches in Blocking & Matching):**
   - *Example:* `bharat impex` vs `shri bharat impex`, `ayan foundation` vs `sri ayan foundation`, `visual bakery` vs `shri visual bakery`.
   - *Cause:* First word blocking key fails because `shri`/`sri` is the first token of the candidate but stripped or absent from S1.
   - *Fix:* Ensure honorifics are thoroughly stripped in `name_core` or add multi-token skeleton keys that include all non-stop tokens.

4. **Single-Token and Short Names (Missed Matches in Blocking):**
   - *Example:* `alois brothers` vs `alois baothers`, `lolinon` vs `lolinon center`.
   - *Cause:* `name_skel_pair` requires at least two skeleton tokens; when one record has only one skeleton token, it falls back to a single token, which fails to match multi-token records.
   - *Fix:* Add individual high-selectivity token keys (capped at moderate block size, e.g. 300) so any shared distinctive name token generates candidates.

5. **False Merges on Singletons (230 occurrences):**
   - *Cause:* Singletons that have a distractor candidate with score slightly above threshold (e.g. 0.61) get falsely merged.
   - *Fix:* Dual-threshold / margin decision logic: single candidate matches must pass a higher confidence bar (e.g., score >= 0.70 or clear margin) to avoid destroying singleton precision.
