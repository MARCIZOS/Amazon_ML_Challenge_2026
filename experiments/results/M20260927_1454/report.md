# Experiment M20260927_1454

- Date: 2026-09-27T15:09:38  |  commit: `6e487eb`  |  runtime: 14.9 min
- Train S1 used to fit: 150,000 (5,881,871 candidate pairs, 472,447 positive); early-stop S1: 15,000; validation S1: 60,000
- Blocking keys: {'name_skel_pair': 1000, 'name_first': 500, 'name_prefix5': 500, 'num_street': 500, 'state_name_skel': 1000, 'addr_tokens': 200}; pre_k=100, top_k=40

## Blocking (validation S1)

| metric | value |
|---|---|
| blocking_recall | 0.9143368104428976 |
| true_pairs | 207452 |
| true_pairs_in_candidates | 189681 |
| candidate_pairs | 2351837 |
| candidates_per_s1_mean | 39.19728333333333 |
| candidates_per_s1_max | 40 |
| s1_without_candidates | 84 |
| reduction_ratio | 0.9999962018942298 |

## Matching (validation S1, best decision)

threshold = 0.6, relative = 0.7, one-to-one = True

| metric | value |
|---|---|
| n_s1 | 60000 |
| macro_f05 | 0.9305113265526137 |
| macro_f05_singletons | 0.9305974652987327 |
| macro_f05_non_singletons | 0.9305062906741843 |
| singleton_share | 0.055233333333333336 |
| pairs_pred | 180149 |
| pairs_true | 207452 |
| pairs_tp | 177652 |
| micro_precision | 0.9861392513974543 |
| micro_recall | 0.8563523128241713 |
| macro_precision | 0.9843676921309421 |
| macro_recall | 0.8575317109787702 |
| false_merges_on_singletons | 230 |

Ceiling with a perfect matcher on these candidates: macro F0.5 = 0.9647

## Submission variants

| variant | threshold | relative | valid macro F0.5 |
|---|---|---|---|
| v1_best | 0.6 | 0.7 | 0.9305113265526137 |
| v2_stricter | 0.65 | 0.7 | 0.9305045887696278 |
| v3_looser | 0.55 | 0.7 | 0.9297277662141238 |
| v4_rel | 0.6 | 0.5 | 0.9298093548722502 |
| v5_precision | 0.7 | 0.7 | 0.9301659999815798 |

## Top of the tuning grid

| threshold | relative | macro_f05 | micro_precision | micro_recall | false_merges_on_singletons |
|---|---|---|---|---|---|
| 0.6000 | 0.7000 | 0.9305 | 0.9861 | 0.8564 | 230 |
| 0.6500 | 0.7000 | 0.9305 | 0.9865 | 0.8561 | 194 |
| 0.6500 | 0.3000 | 0.9303 | 0.9845 | 0.8600 | 194 |
| 0.6500 | 0.0000 | 0.9303 | 0.9845 | 0.8600 | 194 |
| 0.6500 | 0.5000 | 0.9303 | 0.9845 | 0.8600 | 194 |
| 0.7000 | 0.0000 | 0.9302 | 0.9867 | 0.8556 | 175 |
| 0.7000 | 0.5000 | 0.9302 | 0.9867 | 0.8556 | 175 |
| 0.7000 | 0.3000 | 0.9302 | 0.9867 | 0.8556 | 175 |
| 0.7000 | 0.7000 | 0.9302 | 0.9867 | 0.8556 | 175 |
| 0.6000 | 0.0000 | 0.9298 | 0.9811 | 0.8654 | 230 |
