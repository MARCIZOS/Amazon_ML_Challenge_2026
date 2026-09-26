"""Phase-1 rule-based baseline pipeline (kept for comparison; superseded by src/pipeline.py).

Execution flow:
1. Load data
2. Canonicalize records
3. Block (generate candidates)
4. Score candidate pairs
5. Aggregate matches
6. Write outputs
"""

import argparse
import os
import time

import pandas as pd
from tqdm import tqdm

from src.io.data_loader import load_datasets
from src.preprocessing.canonicalize import canonicalize_records
from src.blocking.multi_pass_blocking import generate_candidates
from src.features.feature_pipeline import compute_pair_features
from src.matching.model import RuleBasedMatcher
from src.matching.aggregation import aggregate_matches


def write_results(matches: dict[str, list[str]], candidates: dict[str, set[str]], out_dir: str):
    """Write submission TSV files."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Write matching_results.tsv
    matching_path = os.path.join(out_dir, "matching_results.tsv")
    print(f"Writing {matching_path}...")
    with open(matching_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, match_list in matches.items():
            f.write(f"{s1_id}\t{','.join(match_list)}\n")
            
    # Write candidate_pairs.tsv
    candidate_path = os.path.join(out_dir, "candidate_pairs.tsv")
    print(f"Writing {candidate_path}...")
    with open(candidate_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, cand_list in candidates.items():
            f.write(f"{s1_id}\t{','.join(cand_list)}\n")


def run_pipeline(data_dir: str, split: str, out_dir: str, max_samples: int = None, max_s1_samples: int = None):
    t0 = time.time()
    
    print(f"Loading datasets from {data_dir} ({split} split)...")
    datasets = load_datasets(data_dir, split, nrows=max_samples)
    
    s1_df = datasets["s1"]
    s2_df = datasets["s2"]
    s3_df = datasets["s3"]
    
    if max_s1_samples and len(s1_df) > max_s1_samples:
        s1_df = s1_df.head(max_s1_samples)
    
    print(f"Loaded S1: {len(s1_df)}, S2: {len(s2_df)}, S3: {len(s3_df)}")

    print("Canonicalizing records...")
    s1_norm = canonicalize_records(s1_df)
    s2_norm = canonicalize_records(s2_df)
    s3_norm = canonicalize_records(s3_df)
    
    # Build lookup dictionaries for fast feature extraction
    print("Building lookup dictionaries...")
    s1_lookup = dict(zip(s1_norm["entity_id"], zip(s1_norm["name_norm"], s1_norm["addr_norm"])))
    
    s2s3_norm = pd.concat([s2_norm, s3_norm], ignore_index=True)
    s2s3_lookup = dict(zip(s2s3_norm["entity_id"], zip(s2s3_norm["name_norm"], s2s3_norm["addr_norm"])))
    
    print("Generating candidates via multi-pass blocking...")
    candidates = generate_candidates(s1_norm, s2s3_norm)
    total_candidates = sum(len(c) for c in candidates.values())
    print(f"Generated {total_candidates} candidate pairs.")
    
    print("Scoring candidate pairs...")
    matcher = RuleBasedMatcher(name_threshold=0.65, addr_threshold=0.45)
    scored_pairs = []
    
    # Process with progress bar
    s1_ids = list(candidates.keys())
    for s1_id in tqdm(s1_ids, desc="Scoring pairs"):
        n1, a1 = s1_lookup[s1_id]
        
        for s2s3_id in candidates[s1_id]:
            n2, a2 = s2s3_lookup[s2s3_id]
            
            features = compute_pair_features(n1, a1, n2, a2)
            is_match, score = matcher.predict(s1_id, s2s3_id, features, a1, a2)
            scored_pairs.append((s1_id, s2s3_id, is_match, score))
            
    print("Aggregating matches...")
    all_s1_ids = set(s1_df["entity_id"].tolist())
    matches = aggregate_matches(scored_pairs, all_s1_ids)
    
    total_matches = sum(len(m) for m in matches.values())
    singletons = sum(1 for m in matches.values() if not m)
    print(f"Found {total_matches} total matches.")
    print(f"Singleton S1 entities (no matches): {singletons} ({(singletons/len(all_s1_ids)):.2%})")
    
    print("Writing outputs...")
    write_results(matches, candidates, out_dir)
    
    print(f"Pipeline completed in {time.time() - t0:.1f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="Entity Resolution Baseline Pipeline")
    parser.add_argument("--data-dir", default=r"c:\Users\kusha\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--max-samples", type=int, default=None, help="Process subset of ALL data for testing")
    parser.add_argument("--max-s1-samples", type=int, default=None, help="Process subset of S1 data only")
    args = parser.parse_args()
    
    run_pipeline(args.data_dir, args.split, args.out_dir, args.max_samples, args.max_s1_samples)


if __name__ == "__main__":
    main()
