# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

> Numbers marked **[from report]** must be copied from `experiments/results/<exp_id>/report.md`
> of the run that produced the submitted files. Do not fill them from memory.

---

## 1. Executive Summary
A four-stage pipeline: Unicode-safe normalisation that keeps multilingual text (Indian scripts
are mapped to English with a vocabulary learned from training matches), multi-key
country-partitioned blocking with capped block sizes, a LightGBM pair classifier on ~50
string-similarity features, and a precision-oriented decision step (one-to-one assignment of each
Source 2/3 record, probability and relative thresholds tuned for macro F0.5 including singletons).

---

## 2. Methodology

### 2.1 Problem Analysis
Full-data EDA (`docs/EDA_REPORT.md`):
- 2.2M / 5.0M / 5.3M train records (S1/S2/S3); 1.73M / 4.89M / 5.08M test; test adds France (15% of test S1).
- 5.58% of S1 are singletons; mean 3.46 matches; max 11.
- 100% of true pairs share the country label; no S2/S3 record belongs to two S1 (0 of 7.64M).
- ~26% of S2/S3 records are distractors.
- No postcodes exist in the data (the naive regex captured house numbers).
- Noise: Indian-script names (15% of S2 names), accent injection, digit-for-letter substitutions
  (6eneral), shuffled/added/dropped legal forms, trade-name markers (t/a, dba, aka, née — real name
  after the marker), URL/ID decorations, replaced names or initials, empty or placeholder addresses
  (~3%), abbreviations (Rd/Road, St/SAINT), state codes vs names, French region vs department.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier (+ constrained assignment)
**Core Innovation:** learned native-script → English vocabulary from training matches
(co-occurrence + phonetic alignment), phonetic skeleton keys shared by blocking and features, and
decision rules that exploit the one-to-one structure of the ground truth.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used** (all prefixed by country; per-key cap on block size, larger values dropped):
  name skeleton word pairs (1000), first name word (500), 5-char compact-name prefix (500),
  house number + address word (500), state + first name skeleton (1000), address words (200).
- Union over keys → top `pre_k`=100 per S1 by number of shared keys → re-rank with name/address
  token-set similarity → top `top_k`=40 per S1. These top-40 are `candidate_pairs.tsv`.
- **Candidate pairs generated:** [from report / data/interim/predict_test.json]
- **Blocking recall (validation S1):** [from report]   **Reduction ratio:** [from report]
- **How true matches are protected:** keys on name *and* address (replaced names, empty
  addresses), phonetic skeletons (typos, transliteration), recall measured on held-out train S1.

---

## 4. Matching Model

**Features used** (`src/features/pair_features.py`):
- Name: ratio, token-set, token-sort, partial, Jaro-Winkler on cleaned core name; token-set/ratio on
  phonetic skeleton; compact-name ratio/partial; trade-name alias similarity; Jaccard; first-word
  equality; exact match; token counts; legal-form Jaccard / conflict; transliterated flag.
- Address: token-set, ratio, partial, skeleton token-set, Jaccard; house-number Jaccard / any / first
  equal / conflict; state equal / conflict; missing-address flags.
- Other: number of shared blocking keys, re-rank score, source (S2/S3), ranks and gaps within the
  S1's candidate list. No country one-hot (France is unseen in training).

**Model type:** LightGBM binary classifier (MIT licence), early stopping on a disjoint S1 set.
**Threshold selection method:** grid over (threshold, relative-to-best) with one-to-one resolution,
maximising macro F0.5 (singletons included) on held-out train S1.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, validation):** [from report]  — precision [from report], recall [from report]
- **Ceiling with a perfect matcher on the same candidates:** [from report]
- **Leaderboard (public) per variant:** [record after upload]
- **Common false positives (wrong merges):** [from errors.tsv]
- **Common false negatives (missed matches):** [from errors.tsv]

---

## 6. Conclusion
[2–3 sentences after results are in.]

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` — see `README.md` and `docs/PIPELINE.md`.
Entry points: `python -m src.preprocessing.preprocess --data-dir <dataset>`, then
`python -m src.pipeline` (indexes → train → predict → submission). Outputs:
`output/matching_results.tsv`, `output/candidate_pairs.tsv`.

### B. Additional Results
`experiments/results/<exp_id>/` (tuning grid, feature importance, error examples),
`docs/EDA_REPORT.md`, `docs/BLOCKING_KEYS.md`, `docs/BENCHMARK.md`.
