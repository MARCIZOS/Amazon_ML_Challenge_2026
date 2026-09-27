#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Business Entity Resolution: EDA script.

Run from the repository root:

    python scripts/eda.py --data-dir <DATASET_DIR>

Optional flags:
    --data-dir dataset      path to the dataset folder (default: dataset)
    --out data/interim/eda_output  where the report, charts and samples are written
    --skip-test             skip the test-set profile (faster, less RAM)
    --pair-sample 100000    how many matched pairs to score for similarity

Needs: pandas, numpy, matplotlib. Strongly recommended: rapidfuzz, unidecode
    pip install rapidfuzz unidecode

Outputs (all small, safe to share):
    data/interim/eda_output/eda_report.md        full text report
    data/interim/eda_output/*.png                charts
    data/interim/eda_output/sample_groups.tsv    ~3,000 S1 entities with their matched S2/S3 records
    data/interim/eda_output/sample_test.tsv      random test rows per source/country (incl. France)
    data/interim/eda_output/hard_negatives.tsv   same-postcode but NON-matching pairs
    data/interim/eda_output/stats.json           key numbers in machine-readable form
"""

import argparse
import csv
import gc
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    HAVE_RF = True
except ImportError:
    import difflib
    HAVE_RF = False

try:
    from unidecode import unidecode
except ImportError:
    def unidecode(s):
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

random.seed(42)
np.random.seed(42)

# chart colours (validated categorical palette, light surface)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SRC_COL = {"S1": C_BLUE, "S2": C_ORANGE, "S3": C_AQUA}
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b",
    "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

# --------------------------------------------------------------------------- utils
REPORT = []
STATS = {}


def log(s=""):
    print(s, flush=True)
    REPORT.append(s)


def h(title, level=2):
    log("")
    log("#" * level + " " + title)
    log("")


def table(df, floatfmt="{:.3f}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: floatfmt.format(v) if pd.notna(v) else "")
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
    log("\n".join(lines))


def read_tsv(path, nrows=None):
    t = time.time()
    df = pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                     keep_default_na=False, na_values=[], nrows=nrows,
                     encoding="utf-8", on_bad_lines="warn")
    print(f"  loaded {os.path.basename(path)}: {len(df):,} rows in {time.time()-t:.1f}s", flush=True)
    return df


PUNCT = re.compile(r"[^\w\s]")
WS = re.compile(r"\s+")


def norm(s):
    s = unidecode(str(s)).lower()
    s = s.replace("&", " and ")
    s = PUNCT.sub(" ", s)
    return WS.sub(" ", s).strip()


# postcode-like tokens: India PIN (6 digits, maybe "560 001"), US ZIP(+4), FR (5 digits)
PC_RE = re.compile(r"(?<!\d)(\d{3}\s?\d{3}|\d{5}(?:-\d{4})?)(?!\d)")


def postcode(addr):
    m = PC_RE.findall(str(addr))
    if not m:
        return ""
    return m[-1].replace(" ", "").split("-")[0]


LEGAL = ["pvt", "private", "ltd", "limited", "llc", "inc", "incorporated", "corp",
         "corporation", "co", "company", "llp", "lp", "plc", "pllc",
         "sarl", "sas", "sa", "eurl", "sci", "snc"]
LANDMARK = ["near", "opp", "opposite", "behind", "beside", "next to", "above", "below"]


def sim_ratio(a, b):
    if HAVE_RF:
        return fuzz.ratio(a, b) / 100
    return difflib.SequenceMatcher(None, a, b).ratio()


def sim_tset(a, b):
    if HAVE_RF:
        return fuzz.token_set_ratio(a, b) / 100
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def sim_jw(a, b):
    if HAVE_RF:
        return JaroWinkler.similarity(a, b)
    return sim_ratio(a, b)


def jacc(a, b):
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def savefig(fig, out, name):
    p = os.path.join(out, name)
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log(f"![{name}]({name})")


# --------------------------------------------------------------------------- profile
def profile_source(df, tag, split):
    """Basic column profile of one source file."""
    n = len(df)
    prof = {"rows": n, "columns": list(df.columns)}
    prof["dup_ids"] = int(df["entity_id"].duplicated().sum())
    prof["bad_prefix"] = int((~df["entity_id"].str.startswith(tag + "-")).sum())
    for c in ["business_name", "business_address", "country"]:
        prof[f"empty_{c}"] = int((df[c].str.strip() == "").sum())
    prof["country_counts"] = df["country"].value_counts().to_dict()
    nm = df["business_name"].str.len()
    ad = df["business_address"].str.len()
    prof["name_len_median"] = float(nm.median())
    prof["addr_len_median"] = float(ad.median())
    prof["name_tokens_median"] = float(df["business_name"].str.split().str.len().median())
    prof["addr_tokens_median"] = float(df["business_address"].str.split().str.len().median())
    prof["non_ascii_name_frac"] = float(df["business_name"].map(lambda s: not s.isascii()).mean())
    prof["non_ascii_addr_frac"] = float(df["business_address"].map(lambda s: not s.isascii()).mean())
    # exact duplicate (name, address) rows within a source
    prof["dup_name_addr_rows"] = int(df.duplicated(["business_name", "business_address"]).sum())
    return prof


def text_patterns(df, tag, sample=200_000):
    """Postcode presence, legal suffixes, landmarks - by country."""
    d = df.sample(min(sample, len(df)), random_state=1)
    nn = d["business_name"].map(norm)
    na = d["business_address"].map(norm)
    ntoks = nn.str.split()
    out = {}
    for cty, idx in d.groupby("country").groups.items():
        sub_a = d.loc[idx, "business_address"]
        sub_na = na.loc[idx]
        sub_nt = ntoks.loc[idx]
        pcs = sub_a.map(postcode)
        out[cty] = {
            "n": len(idx),
            "has_postcode": float((pcs != "").mean()),
            "postcode_6digit": float((pcs.str.len() == 6).mean()),
            "postcode_5digit": float((pcs.str.len() == 5).mean()),
            "has_legal_suffix": float(sub_nt.map(lambda t: bool(t) and t[-1] in LEGAL).mean()),
            "has_landmark": float(sub_na.map(lambda s: any(f" {w} " in f" {s} " for w in LANDMARK)).mean()),
            "addr_has_digit": float(sub_a.str.contains(r"\d").mean()),
            "name_has_digit": float(d.loc[idx, "business_name"].str.contains(r"\d").mean()),
            "name_all_caps": float(d.loc[idx, "business_name"].map(lambda s: s.isupper()).mean()),
        }
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out", default=os.path.join("data", "interim", "eda_output"))
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--pair-sample", type=int, default=100_000)
    ap.add_argument("--nrows", type=int, default=None, help="debug: only read first N rows per file")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tr = os.path.join(args.data_dir, "train")
    te = os.path.join(args.data_dir, "test")
    t0 = time.time()

    log("# Amazon ML Challenge 2026 - Entity Resolution EDA")
    log(f"rapidfuzz available: {HAVE_RF}  (install it for faster, better similarity scores)")

    # ---------------------------------------------------------------- load train
    h("1. Train files - shape and quality")
    S = {}
    for tag in ["S1", "S2", "S3"]:
        S[tag] = read_tsv(os.path.join(tr, f"train_source{tag[1]}.tsv"), args.nrows)
    gt = read_tsv(os.path.join(tr, "train_ground_truth.tsv"), args.nrows)

    prof = {tag: profile_source(S[tag], tag, "train") for tag in S}
    STATS["train_profile"] = prof
    keys = ["rows", "dup_ids", "bad_prefix", "empty_business_name", "empty_business_address",
            "empty_country", "dup_name_addr_rows", "name_len_median", "addr_len_median",
            "name_tokens_median", "addr_tokens_median", "non_ascii_name_frac", "non_ascii_addr_frac"]
    table(pd.DataFrame({t: {k: prof[t][k] for k in keys} for t in prof}))
    log("")
    log("Columns: " + ", ".join(S["S1"].columns))

    h("Country distribution (train)", 3)
    cc = pd.DataFrame({t: prof[t]["country_counts"] for t in prof}).fillna(0).astype(int)
    table(cc)

    # ---------------------------------------------------------------- ground truth
    h("2. Ground truth structure")
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
    gt["ids"] = gt["matched_entity_ids"].map(lambda s: [x for x in s.split(",") if x.strip()])
    gt["n_match"] = gt["ids"].str.len()
    gt["n_s2"] = gt["ids"].map(lambda l: sum(x.startswith("S2-") for x in l))
    gt["n_s3"] = gt["ids"].map(lambda l: sum(x.startswith("S3-") for x in l))

    s1_ids = set(S["S1"]["entity_id"])
    gt_ids = set(gt["source1_entity_id"])
    log(f"- GT rows: {len(gt):,} | S1 rows: {len(S['S1']):,}")
    log(f"- S1 ids missing from GT: {len(s1_ids - gt_ids):,} | GT ids not in S1: {len(gt_ids - s1_ids):,}")
    sing = float((gt["n_match"] == 0).mean())
    log(f"- **Singleton rate (S1 with no match): {sing:.2%}**  <- predicting empty for all gives this F0.5 as a floor")
    log(f"- Mean matches per S1: {gt['n_match'].mean():.2f} | per non-singleton: {gt.loc[gt.n_match>0,'n_match'].mean():.2f}")
    log(f"- Max matches for one S1: {gt['n_match'].max()}")
    STATS["singleton_rate"] = sing

    vc = gt["n_match"].clip(upper=10).value_counts().sort_index()
    vc.index = [str(i) if i < 10 else "10+" for i in vc.index]
    h("Matches per S1 entity", 3)
    table(pd.DataFrame({"count": vc, "share": vc / vc.sum()}))

    comp = pd.crosstab(gt["n_s2"].clip(upper=3), gt["n_s3"].clip(upper=3))
    comp.index.name = "n_S2 \\ n_S3"
    h("Composition: # S2 matches (rows) x # S3 matches (cols), capped at 3", 3)
    table(comp)

    # which S2/S3 records ever get matched; is the mapping many-to-one?
    exploded = gt[["source1_entity_id", "ids"]].explode("ids").dropna()
    exploded = exploded[exploded["ids"] != ""]
    per_rec = exploded.groupby("ids")["source1_entity_id"].nunique()
    log("")
    log(f"- Total matched (S1, S2/S3) pairs: {len(exploded):,}")
    log(f"- S2/S3 records linked to >1 S1: {int((per_rec>1).sum()):,} "
        f"({(per_rec>1).mean():.3%})  <- if ~0, enforce 'each S2/S3 record -> at most one S1'")
    for tag in ["S2", "S3"]:
        matched = exploded["ids"].str.startswith(tag + "-").sum()
        total = len(S[tag])
        log(f"- {tag}: {matched:,} of {total:,} records matched to some S1 ({matched/max(1,total):.1%}); "
            f"the rest are distractors")
    missing = set(exploded["ids"]) - set(S["S2"]["entity_id"]) - set(S["S3"]["entity_id"])
    log(f"- GT ids not found in S2/S3 files: {len(missing):,}")
    STATS["multi_s1_records"] = int((per_rec > 1).sum())

    # singleton rate by country
    s1c = S["S1"].set_index("entity_id")["country"]
    gt["country"] = gt["source1_entity_id"].map(s1c)
    byc = gt.groupby("country").agg(n=("n_match", "size"),
                                    singleton_rate=("n_match", lambda x: (x == 0).mean()),
                                    mean_matches=("n_match", "mean"))
    h("By country", 3)
    table(byc)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(vc.index, vc.values, color=C_BLUE, width=0.7)
    ax.set_title("Matches per Source-1 entity (train)", loc="left")
    ax.set_xlabel("number of S2/S3 matches"); ax.set_ylabel("S1 entities")
    savefig(fig, args.out, "matches_per_s1.png")

    # ---------------------------------------------------------------- text patterns
    h("3. Text patterns by source and country (200k-row samples)")
    tp = {tag: text_patterns(S[tag], tag) for tag in S}
    STATS["text_patterns_train"] = tp
    rows = {}
    for tag, d in tp.items():
        for cty, v in d.items():
            rows[f"{tag}/{cty}"] = v
    table(pd.DataFrame(rows).T)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for tag in S:
        axes[0].hist(S[tag]["business_name"].str.len().clip(upper=100), bins=50, histtype="step",
                     linewidth=2, color=SRC_COL[tag], label=tag, density=True)
        axes[1].hist(S[tag]["business_address"].str.len().clip(upper=250), bins=50, histtype="step",
                     linewidth=2, color=SRC_COL[tag], label=tag, density=True)
    axes[0].set_title("Name length (chars)", loc="left"); axes[1].set_title("Address length (chars)", loc="left")
    axes[0].legend(frameon=False)
    savefig(fig, args.out, "length_by_source.png")

    # top name tokens
    h("Most common name tokens (normalised), per source", 3)
    for tag in S:
        c = Counter()
        for s in S[tag]["business_name"].sample(min(200_000, len(S[tag])), random_state=2):
            c.update(norm(s).split())
        log(f"- **{tag}**: " + ", ".join(f"{w} ({n})" for w, n in c.most_common(30)))
    h("Most common address tokens (normalised), per source", 3)
    for tag in S:
        c = Counter()
        for s in S[tag]["business_address"].sample(min(200_000, len(S[tag])), random_state=3):
            c.update(t for t in norm(s).split() if not t.isdigit())
        log(f"- **{tag}**: " + ", ".join(f"{w} ({n})" for w, n in c.most_common(30)))

    # ---------------------------------------------------------------- pair analysis
    h("4. How similar are true matches vs non-matches?")
    rec = {}
    for tag in S:
        d = S[tag]
        rec.update(zip(d["entity_id"], zip(d["business_name"], d["business_address"], d["country"])))

    pos = exploded.sample(min(args.pair_sample, len(exploded)), random_state=4)
    pos_pairs = [(a, b) for a, b in zip(pos["source1_entity_id"], pos["ids"]) if a in rec and b in rec]

    # random negatives from same country
    by_cty = defaultdict(list)
    for tag in ["S2", "S3"]:
        for i, c in zip(S[tag]["entity_id"], S[tag]["country"]):
            by_cty[c].append(i)
    true_set = set(zip(exploded["source1_entity_id"], exploded["ids"]))
    neg_pairs = []
    s1_list = S["S1"]["entity_id"].tolist()
    while len(neg_pairs) < len(pos_pairs) // 2:
        a = random.choice(s1_list)
        cands = by_cty.get(rec[a][2]) or []
        if not cands:
            continue
        b = random.choice(cands)
        if (a, b) not in true_set:
            neg_pairs.append((a, b))

    # hard negatives: share a postcode, not matched
    log("Building postcode index over S2+S3 (for block sizes and hard negatives)...")
    pc_index = defaultdict(list)
    for tag in ["S2", "S3"]:
        for i, a in zip(S[tag]["entity_id"], S[tag]["business_address"]):
            p = postcode(a)
            if p:
                pc_index[p].append(i)
    block_sizes = np.array([len(v) for v in pc_index.values()])
    hard = []
    s1_sample = random.sample(s1_list, min(20_000, len(s1_list)))
    for a in s1_sample:
        p = postcode(rec[a][1])
        if not p or p not in pc_index:
            continue
        cands = pc_index[p]
        for b in random.sample(cands, min(5, len(cands))):
            if (a, b) not in true_set:
                hard.append((a, b))
        if len(hard) >= len(pos_pairs) // 2:
            break

    def feats(pairs, label):
        out = []
        for a, b in pairs:
            n1, a1, c1 = rec[a]; n2, a2, c2 = rec[b]
            nn1, nn2, na1, na2 = norm(n1), norm(n2), norm(a1), norm(a2)
            p1, p2 = postcode(a1), postcode(a2)
            t1, t2 = nn1.split(), nn2.split()
            out.append({
                "s1": a, "other": b, "label": label,
                "name_exact": nn1 == nn2,
                "name_ratio": sim_ratio(nn1, nn2),
                "name_tset": sim_tset(nn1, nn2),
                "name_jw": sim_jw(nn1, nn2),
                "name_jacc": jacc(nn1, nn2),
                "first_tok_eq": bool(t1 and t2 and t1[0] == t2[0]),
                "addr_ratio": sim_ratio(na1, na2),
                "addr_tset": sim_tset(na1, na2),
                "addr_jacc": jacc(na1, na2),
                "pc_both": bool(p1 and p2),
                "pc_eq": bool(p1 and p1 == p2),
                "country_eq": c1 == c2,
                "src": b[:2],
            })
        return out

    log(f"Scoring {len(pos_pairs):,} positive, {len(neg_pairs):,} random-negative, {len(hard):,} hard-negative pairs...")
    P = pd.DataFrame(feats(pos_pairs, "match") + feats(neg_pairs, "random_neg") + feats(hard, "hard_neg_same_pc"))
    fcols = ["name_exact", "name_ratio", "name_tset", "name_jw", "name_jacc", "first_tok_eq",
             "addr_ratio", "addr_tset", "addr_jacc", "pc_both", "pc_eq", "country_eq"]
    summ = P.groupby("label")[fcols].mean().T
    h("Mean feature value by pair type", 3)
    table(summ)
    h("Matched pairs: S1-S2 vs S1-S3", 3)
    table(P[P.label == "match"].groupby("src")[fcols].mean().T)
    h("Matched pairs by country", 3)
    P["country"] = P["s1"].map(lambda i: rec[i][2])
    table(P[P.label == "match"].groupby("country")[fcols].mean().T)

    q = P[P.label == "match"][["name_ratio", "name_tset", "addr_tset"]].quantile([.01, .05, .1, .25, .5])
    q.index.name = "quantile"
    h("Low tail of similarity for TRUE matches (how noisy do matches get?)", 3)
    table(q)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, col, title in zip(axes, ["name_tset", "name_ratio", "addr_tset"],
                              ["Name token-set sim", "Name char ratio", "Address token-set sim"]):
        for lab, colr in [("match", C_BLUE), ("hard_neg_same_pc", C_ORANGE), ("random_neg", "#9a9992")]:
            v = P.loc[P.label == lab, col]
            if len(v):
                ax.hist(v, bins=40, range=(0, 1), histtype="step", linewidth=2, color=colr,
                        density=True, label=lab)
        ax.set_title(title, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    savefig(fig, args.out, "similarity_pos_vs_neg.png")

    # ---------------------------------------------------------------- blocking
    h("5. Blocking signals - what fraction of true pairs would each key catch?")
    m = P[P.label == "match"]
    tok_first = m["first_tok_eq"].mean()
    log(f"- Same postcode (when S1 has one): {m.loc[m.s1.map(lambda i: postcode(rec[i][1]) != ''), 'pc_eq'].mean():.2%}")
    log(f"- Same postcode (all pairs): {m['pc_eq'].mean():.2%}")
    log(f"- Same first name token: {tok_first:.2%}")
    log(f"- Share any name token: {(m['name_jacc'] > 0).mean():.2%}")
    log(f"- Same country label: {m['country_eq'].mean():.2%}")
    log(f"- Exact normalised name: {m['name_exact'].mean():.2%}")
    log("")
    log(f"- Postcode blocks over S2+S3: {len(block_sizes):,} distinct codes; size median {np.median(block_sizes):.0f}, "
        f"p90 {np.percentile(block_sizes,90):.0f}, p99 {np.percentile(block_sizes,99):.0f}, max {block_sizes.max():,}")
    STATS["postcode_block_sizes"] = {"n": int(len(block_sizes)), "median": float(np.median(block_sizes)),
                                     "p90": float(np.percentile(block_sizes, 90)),
                                     "p99": float(np.percentile(block_sizes, 99)), "max": int(block_sizes.max())}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(np.log10(block_sizes), bins=50, color=C_BLUE)
    ax.set_title("Postcode block size over S2+S3 (log10 records per code)", loc="left")
    ax.set_xlabel("log10(block size)"); ax.set_ylabel("postcodes")
    savefig(fig, args.out, "postcode_block_sizes.png")

    # ---------------------------------------------------------------- examples
    h("6. Examples")
    ex = gt[gt.n_match > 0].sample(min(15, (gt.n_match > 0).sum()), random_state=5)
    for _, r in ex.iterrows():
        a = r["source1_entity_id"]
        log(f"- **{a}** [{rec[a][2]}] {rec[a][0]}  |  {rec[a][1]}")
        for b in r["ids"][:6]:
            if b in rec:
                log(f"    - {b} [{rec[b][2]}] {rec[b][0]}  |  {rec[b][1]}")
    h("Singletons (no match)", 3)
    for a in gt[gt.n_match == 0]["source1_entity_id"].sample(min(8, (gt.n_match == 0).sum()), random_state=6):
        log(f"- {a} [{rec[a][2]}] {rec[a][0]}  |  {rec[a][1]}")
    h("Hardest true matches (lowest name similarity)", 3)
    for _, r in m.nsmallest(12, "name_tset").iterrows():
        log(f"- {rec[r.s1][0]}  |  {rec[r.s1][1]}")
        log(f"    - {rec[r.other][0]}  |  {rec[r.other][1]}   (name_tset={r.name_tset:.2f}, addr_tset={r.addr_tset:.2f})")

    # ---------------------------------------------------------------- samples to share
    samp_ids = gt["source1_entity_id"].sample(min(3000, len(gt)), random_state=7)
    rows = []
    gmap = gt.set_index("source1_entity_id")["ids"]
    for a in samp_ids:
        n, ad, c = rec[a]
        rows.append({"group": a, "entity_id": a, "business_name": n, "business_address": ad, "country": c})
        for b in gmap[a]:
            if b in rec:
                n, ad, c = rec[b]
                rows.append({"group": a, "entity_id": b, "business_name": n, "business_address": ad, "country": c})
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "sample_groups.tsv"), sep="\t", index=False,
                              quoting=csv.QUOTE_NONE, escapechar="\\")
    hn = P[P.label == "hard_neg_same_pc"].head(2000)
    pd.DataFrame([{"s1": a, "s1_name": rec[a][0], "s1_addr": rec[a][1],
                   "other": b, "other_name": rec[b][0], "other_addr": rec[b][1]}
                  for a, b in zip(hn.s1, hn.other)]).to_csv(
        os.path.join(args.out, "hard_negatives.tsv"), sep="\t", index=False,
        quoting=csv.QUOTE_NONE, escapechar="\\")

    del rec, P, pc_index, S
    gc.collect()

    # ---------------------------------------------------------------- test set
    if not args.skip_test:
        h("7. Test files - shape and country mix")
        T = {}
        test_samples = []
        for tag in ["S1", "S2", "S3"]:
            d = read_tsv(os.path.join(te, f"test_source{tag[1]}.tsv"), args.nrows)
            T[tag] = profile_source(d, tag, "test")
            T[tag]["patterns"] = text_patterns(d, tag)
            for cty, g in d.groupby("country"):
                test_samples.append(g.sample(min(300, len(g)), random_state=8))
            del d
            gc.collect()
        STATS["test_profile"] = T
        table(pd.DataFrame({t: {k: T[t][k] for k in keys} for t in T}))
        h("Country distribution (test)", 3)
        table(pd.DataFrame({t: T[t]["country_counts"] for t in T}).fillna(0).astype(int))
        h("Text patterns (test)", 3)
        rows = {f"{t}/{c}": v for t in T for c, v in T[t]["patterns"].items()}
        table(pd.DataFrame(rows).T)
        pd.concat(test_samples).to_csv(os.path.join(args.out, "sample_test.tsv"), sep="\t", index=False,
                                       quoting=csv.QUOTE_NONE, escapechar="\\")

    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(STATS, f, indent=1, default=str)
    log("")
    log(f"_Done in {(time.time()-t0)/60:.1f} min._")
    with open(os.path.join(args.out, "eda_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
    print(f"\nReport written to {os.path.join(args.out, 'eda_report.md')}")


if __name__ == "__main__":
    main()
