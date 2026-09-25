"""Evaluation script for Phase 2 validation.

Computes:
- Per-entity macro F0.5
- Precision and recall
- Singleton performance
- Blocking recall / recall ceiling
- Candidate stats (number of candidates, reduction ratio)
- False-merge and Missed-match examples
"""

import argparse
import time

import pandas as pd


def parse_id_list(id_string):
    """Parse comma-separated IDs, handling NaN and empty strings."""
    if pd.isna(id_string) or not str(id_string).strip():
        return set()
    return set(x.strip() for x in str(id_string).split(",") if x.strip())


def compute_metrics(gt_df, pred_df, cand_df, s2_count=5034616, s3_count=5285603):
    """Compute all evaluation metrics."""
    # Build dictionaries for fast lookup
    gt_dict = {
        s1: parse_id_list(mids)
        for s1, mids in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"])
    }
    
    pred_dict = {
        s1: parse_id_list(mids)
        for s1, mids in zip(pred_df["source1_entity_id"], pred_df.get("matched_entity_ids", [""] * len(pred_df)))
    }
    
    cand_dict = {}
    if cand_df is not None:
        cand_dict = {
            s1: parse_id_list(mids)
            for s1, mids in zip(cand_df["source1_entity_id"], cand_df.get("candidate_entity_ids", [""] * len(cand_df)))
        }

    # Intersect S1 IDs to only evaluate on what was predicted
    s1_ids = [s1 for s1 in pred_dict.keys() if s1 in gt_dict]
    if not s1_ids:
        print("No intersecting S1 IDs found between prediction and ground truth.")
        return
        
    print(f"Evaluating on {len(s1_ids)} S1 entities...")

    # Per-entity F0.5 components
    macro_f05_sum = 0.0
    macro_precision_sum = 0.0
    macro_recall_sum = 0.0
    
    global_true_pos = 0
    global_false_pos = 0
    global_false_neg = 0

    # Singleton stats
    true_singletons = 0
    predicted_singletons = 0
    correct_singletons = 0
    
    # Blocking stats
    blocking_true_pos = 0
    global_positives = 0
    total_candidates = 0

    false_merges = []
    missed_matches = []

    for s1 in s1_ids:
        true_ids = gt_dict[s1]
        pred_ids = pred_dict.get(s1, set())
        
        # Singleton logic
        is_true_singleton = len(true_ids) == 0
        is_pred_singleton = len(pred_ids) == 0
        
        if is_true_singleton:
            true_singletons += 1
        if is_pred_singleton:
            predicted_singletons += 1
        if is_true_singleton and is_pred_singleton:
            correct_singletons += 1
            
        global_positives += len(true_ids)

        # TP, FP, FN for this entity
        tp = len(true_ids & pred_ids)
        fp = len(pred_ids - true_ids)
        fn = len(true_ids - pred_ids)
        
        global_true_pos += tp
        global_false_pos += fp
        global_false_neg += fn
        
        # Save some examples
        if fp > 0 and len(false_merges) < 5:
            false_merges.append((s1, pred_ids - true_ids))
        if fn > 0 and len(missed_matches) < 5:
            missed_matches.append((s1, true_ids - pred_ids))

        # Precision, Recall, F0.5 per entity
        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if is_true_singleton and is_pred_singleton else 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if is_true_singleton and is_pred_singleton else 0.0)
        
        if precision + recall > 0:
            # F0.5 weights precision twice as much as recall
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        else:
            f05 = 0.0
            
        macro_precision_sum += precision
        macro_recall_sum += recall
        macro_f05_sum += f05
        
        # Blocking stats
        if cand_dict:
            cands = cand_dict.get(s1, set())
            total_candidates += len(cands)
            blocking_true_pos += len(true_ids & cands)

    # Averages
    n = len(s1_ids)
    macro_f05 = macro_f05_sum / n
    macro_precision = macro_precision_sum / n
    macro_recall = macro_recall_sum / n
    
    # Global metrics
    global_precision = global_true_pos / (global_true_pos + global_false_pos) if (global_true_pos + global_false_pos) > 0 else 0
    global_recall = global_true_pos / (global_true_pos + global_false_neg) if (global_true_pos + global_false_neg) > 0 else 0

    print("\n--- MATCHING METRICS ---")
    print(f"Per-entity Macro F0.5: {macro_f05:.4f}")
    print(f"Per-entity Macro Precision: {macro_precision:.4f}")
    print(f"Per-entity Macro Recall: {macro_recall:.4f}")
    print(f"Global Micro Precision: {global_precision:.4f}")
    print(f"Global Micro Recall: {global_recall:.4f}")
    
    print("\n--- SINGLETON PERFORMANCE ---")
    print(f"True singletons: {true_singletons} ({(true_singletons/n)*100:.1f}%)")
    print(f"Predicted singletons: {predicted_singletons}")
    print(f"Correctly predicted singletons: {correct_singletons} ({(correct_singletons/max(1,true_singletons))*100:.1f}% of true singletons)")

    if cand_dict:
        blocking_recall = blocking_true_pos / global_positives if global_positives > 0 else 1.0
        # Total possible pairs = n * (s2_count + s3_count)
        # We don't have exactly the pool size, but we approximate max possible pairs
        max_possible_pairs = n * (s2_count + s3_count)
        reduction_ratio = 1.0 - (total_candidates / max_possible_pairs)
        
        print("\n--- BLOCKING METRICS ---")
        print(f"Blocking Recall (Recall Ceiling): {blocking_recall:.4f} ({blocking_true_pos}/{global_positives} true matches covered by candidates)")
        print(f"Total Candidates Generated: {total_candidates}")
        print(f"Candidates per S1 (mean): {total_candidates/n:.1f}")
        print(f"Candidate Reduction Ratio: {reduction_ratio:.6f}")
        
    print("\n--- EXAMPLES ---")
    print("False Merges (Predicted match that is incorrect):")
    for s1, fps in false_merges:
        print(f"  S1: {s1} incorrectly matched to {fps}")
        
    print("Missed Matches (True match that was not predicted):")
    for s1, fns in missed_matches:
        print(f"  S1: {s1} missed matches {fns}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default=r"c:\Users\kusha\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_ground_truth.tsv")
    parser.add_argument("--pred", default="output/matching_results.tsv")
    parser.add_argument("--cand", default="output/candidate_pairs.tsv")
    args = parser.parse_args()

    t0 = time.time()
    print("Loading data...")
    gt_df = pd.read_csv(args.gt, sep="\t", dtype=str)
    pred_df = pd.read_csv(args.pred, sep="\t", dtype=str)
    try:
        cand_df = pd.read_csv(args.cand, sep="\t", dtype=str)
    except FileNotFoundError:
        cand_df = None
        
    compute_metrics(gt_df, pred_df, cand_df)
    print(f"\nEvaluation completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
