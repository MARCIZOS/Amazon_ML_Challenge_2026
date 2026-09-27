# Decision Log

| ID | Date | Decision | Alternatives | Reason | Owner | Status |
|---|---|---|---|---|---|---|
| D001 | | Use local execution initially | SageMaker, hybrid | Reduce setup overhead and accelerate iteration | Team | Proposed |
| D002 | | Use multi-pass blocking | Single key, all-pairs | Balance candidate recall and computational cost | Member 4 | Proposed |
| D003 | 2026-09-26 | Preserve raw and normalized fields | Normalize in place | Debugging and reproducibility | Member 2 | Accepted - implemented in `src/preprocessing/normalize.py` |
| D004 | | Use configuration-driven thresholds | Hard-coded values | Easier experimentation | Member 1 | Proposed |
| D005 | 2026-09-26 | Do not use postcodes for blocking or features | Postcode regex | EDA: the data has no postcodes; the regex matched house numbers (0% US ZIPs, 0% Indian PINs) | Member 2 | Accepted |
| D006 | 2026-09-26 | Transliterate Indian scripts with `anyascii` + a vocabulary learned from training matches | `unidecode` only | `unidecode` gives 'praaivett'; learned vocab maps 'प्राइवेट' -> 'private'; Indian-script name similarity of true pairs 0.15 -> 0.83; anyascii is ISC-licensed | Member 2 | Accepted |
| D007 | 2026-09-26 | Store normalised data as Parquet in `data/processed/` (originals + 14 fields) | Re-normalise in every run | Normalising 24.2M rows takes ~19 min; Parquet reload takes seconds and is ~6x smaller | Member 2 | Accepted |
| D008 | 2026-09-26 | Block within country only | Cross-country blocking | 100% of true pairs share the country label | Member 2 / Member 4 | Proposed |
| D009 | 2026-09-26 | Each S2/S3 record maps to at most one S1 | Allow many-to-many | 0 of 7.64M matched S2/S3 records belong to more than one S1 | Member 2 / Member 1 | Proposed |
| D010 | 2026-09-26 | Candidates = union of 6 country-prefixed keys (name_skel_pair, name_first, name_prefix5, num_street, state_name_skel, addr_tokens) with per-key block caps, top-100 by shared keys, re-ranked to top-40 by cheap name/address similarity | Postcode / first-token only (Phase 1) | No postcodes in data; union of name- and address-based keys covers replaced names and missing addresses; top-K bounds cost | Member 4 | Accepted (tune K from report) |
| D011 | 2026-09-26 | LightGBM binary matcher on ~50 pair features, no country one-hot | Rule thresholds (Phase 1) | Learns name/address trade-offs; MIT licence; country-agnostic features so France (unseen) is not out-of-distribution | Member 1 | Accepted |
| D012 | 2026-09-26 | Decision = one-to-one per S2/S3 record, then p >= threshold and p >= relative x best p per S1; tuned for macro F0.5 on held-out train S1 | Fixed 0.5 cut-off | Metric is precision-weighted and includes singletons; one-to-one holds for 100% of train GT | Member 1 / Member 4 | Accepted |
| D013 | 2026-09-26 | Score the test set once, derive 5 leaderboard variants from the validation grid | Retrain per submission | 5 submissions/day; variants cost seconds | Member 4 | Accepted |
