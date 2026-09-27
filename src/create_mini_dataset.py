import os
import pandas as pd

def parse_id_list(id_string):
    if pd.isna(id_string) or not str(id_string).strip():
        return set()
    return set(x.strip() for x in str(id_string).split(",") if x.strip())

def main():
    data_dir = r"c:\Users\kusha\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train"
    out_dir = "mini_dataset"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading Ground Truth...")
    gt_df = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t", dtype=str)
    
    # Take first 10,000 S1 entities
    n_samples = 10000
    mini_gt = gt_df.head(n_samples)
    
    # Collect all needed S2 and S3 IDs
    needed_s2 = set()
    needed_s3 = set()
    
    for _, row in mini_gt.iterrows():
        mids = parse_id_list(row["matched_entity_ids"])
        for mid in mids:
            if mid.startswith("S2-"):
                needed_s2.add(mid)
            elif mid.startswith("S3-"):
                needed_s3.add(mid)
                
    print(f"Need {len(needed_s2)} S2 records and {len(needed_s3)} S3 records for true matches.")
    
    print("Loading S1...")
    needed_s1 = set(mini_gt["source1_entity_id"])
    s1_full = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", dtype=str)
    s1_df = s1_full[s1_full["entity_id"].isin(needed_s1)]
    
    print("Loading S2...")
    # Load in chunks and filter
    s2_chunks = []
    chunk_idx = 0
    for chunk in pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t", dtype=str, chunksize=500000):
        # Keep true matches + 2% random noise
        mask = chunk["entity_id"].isin(needed_s2) | (chunk.index % 50 == 0)
        s2_chunks.append(chunk[mask])
        chunk_idx += 1
        print(f"S2 chunk {chunk_idx} processed")
    s2_df = pd.concat(s2_chunks)
    
    print("Loading S3...")
    s3_chunks = []
    chunk_idx = 0
    for chunk in pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t", dtype=str, chunksize=500000):
        mask = chunk["entity_id"].isin(needed_s3) | (chunk.index % 50 == 0)
        s3_chunks.append(chunk[mask])
        chunk_idx += 1
        print(f"S3 chunk {chunk_idx} processed")
    s3_df = pd.concat(s3_chunks)
    
    print(f"Mini dataset sizes: S1: {len(s1_df)}, S2: {len(s2_df)}, S3: {len(s3_df)}")
    
    # Save
    print("Saving...")
    mini_gt.to_csv(os.path.join(out_dir, "train_ground_truth.tsv"), sep="\t", index=False)
    s1_df.to_csv(os.path.join(out_dir, "train_source1.tsv"), sep="\t", index=False)
    s2_df.to_csv(os.path.join(out_dir, "train_source2.tsv"), sep="\t", index=False)
    s3_df.to_csv(os.path.join(out_dir, "train_source3.tsv"), sep="\t", index=False)
    print("Done!")

if __name__ == "__main__":
    main()
