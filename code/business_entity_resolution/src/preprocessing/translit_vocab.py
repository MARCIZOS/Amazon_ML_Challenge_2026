"""Learn a native-script -> English word vocabulary from the TRAINING matches.

Why: ~15% of Source-2/3 names (and many state names in addresses) are English
words written in Indian scripts, e.g. 'प्राइवेट लिमिटेड' = 'private limited'.
Generic transliteration gives 'praivet', close but not equal. The training
ground truth pairs every such record with its clean English Source-1 record, so
we can align tokens and learn the exact mapping. Only the provided training
data is used - no external resource.

Algorithm
  for every matched (S1, S2/S3) pair where the S2/S3 text contains native script
    for each native token k: count the pair, and for every English candidate w
      (S1 words and adjacent word bigrams) that sounds at least a little like k
      (sound_score >= --min-sim) count the co-occurrence (k, w)
  keep k -> argmax_w  P(w|k) + 0.5 * sound(k, w)   with P(w|k) >= --min-share
  and k seen in >= --min-count pairs

Usage (from project root)
  full training data (run on the machine that has the dataset):
    python -m src.preprocessing.translit_vocab --data-dir <DATASET_DIR>
  EDA sample groups file (quick test):
    python -m src.preprocessing.translit_vocab --groups data/interim/eda_output/sample_groups.tsv --min-count 1
Writes configs/translit_vocab.json (picked up automatically by Normalizer).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from functools import lru_cache

from rapidfuzz import fuzz

from src.io.data_loader import read_tsv, iter_tsv
from src.preprocessing.normalize import (DEFAULT_VOCAB, _to_ascii, has_non_latin,
                                         native_key, skeleton, unicode_canonical)

INDIC_RE = r"[ऀ-෿]"          # Devanagari .. Sinhala blocks (all Indian scripts)
SPLIT = re.compile(r"[\s,;:()\[\]{}|/\\]+")
ASCII_TOK = re.compile(r"[a-z]+")


def english_candidates(text: str) -> list[str]:
    toks = ASCII_TOK.findall(_to_ascii(unicode_canonical(text)).lower())
    toks = [t for t in toks if len(t) >= 2]
    return toks + [a + " " + b for a, b in zip(toks, toks[1:])]


def native_tokens(text: str) -> list[str]:
    out = []
    for tok in SPLIT.split(unicode_canonical(text)):
        if tok and has_non_latin(tok):
            k = native_key(tok)
            if k:
                out.append(k)
    return out


@lru_cache(maxsize=500_000)
def learn_key(s: str) -> str:
    """Phonetic key tuned for Indian-script transliterations: voiced and
    unvoiced consonants are merged (Tamil writes k/g, t/d, p/b with one letter),
    'rr' -> 't' (Malayalam/Tamil retroflex), nasal 'm' before a consonant -> 'n'."""
    s = re.sub(r"[^a-z]", "", s.lower())
    s = s.replace("rr", "t").replace("ph", "f").replace("x", "ks")
    s = re.sub(r"c(?=[eiy])", "s", s)
    s = s.translate(str.maketrans("gdbvwfzjqc", "ktppppsckk"))
    s = re.sub(r"m(?=[^aeiouy m])", "n", s)
    s = re.sub(r"^y(?=[aeiou])", "", s)
    if not s:
        return s
    first, rest = s[0], re.sub(r"[aeiouyh]", "", s[1:])
    out = []
    for ch in first + rest:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


@lru_cache(maxsize=2_000_000)
def sound_score(native: str, cand: str) -> float:
    """0..1 sound-alike score between a native token and an English candidate."""
    t = re.sub(r"[^a-z]", "", _to_ascii(native).lower())
    c = cand.replace(" ", "")
    if not t or not c:
        return 0.0
    return 0.5 * fuzz.ratio(learn_key(t), learn_key(c)) / 100 + 0.5 * fuzz.ratio(t, c) / 100


class VocabLearner:
    """Co-occurrence + sound-alike alignment.

    For each native token k we count in how many matched pairs each English
    candidate w (S1 word or word bigram) appears together with k, keeping only
    candidates that sound at least a little alike (>= min_sim). The final
    mapping maximises  P(w | k) + sound_weight * sound(k, w)  subject to
    P(w | k) >= min_share. Co-occurrence stops frequent look-alikes (e.g.
    'care') from stealing a word ('future') that only sometimes appears with it.
    """

    def __init__(self, min_sim=0.4, sound_weight=0.5):
        self.n = Counter()                   # pairs containing native token k
        self.co = defaultdict(Counter)       # k -> candidate -> pairs
        self.min_sim = min_sim
        self.sound_weight = sound_weight
        self.pairs_seen = 0

    def add_pair(self, s1_text: str, other_text: str):
        nat = set(native_tokens(other_text))
        if not nat:
            return
        cands = set(english_candidates(s1_text))
        if not cands:
            return
        self.pairs_seen += 1
        for k in nat:
            self.n[k] += 1
            for c in cands:
                if sound_score(k, c) >= self.min_sim:
                    self.co[k][c] += 1

    def build(self, min_count=3, min_share=0.5) -> dict:
        vocab = {}
        for k, n in self.n.items():
            if n < min_count or k not in self.co:
                continue
            best, best_s = None, -1.0
            for c, m in self.co[k].items():
                p = m / n
                if p < min_share:
                    continue
                sc = p + self.sound_weight * sound_score(k, c)
                if " " in c:                 # prefer single words unless the bigram
                    sc -= 0.03               # sounds clearly better ('tamil nadu')
                if sc > best_s:
                    best, best_s = c, sc
            if best is not None:
                vocab[k] = best
        return vocab


# ----------------------------------------------------------------------------- inputs
def pairs_from_groups(path):
    """sample_groups.tsv format: group, entity_id, business_name, business_address, country."""
    df = read_tsv(path)
    s1 = df[df["entity_id"].str.startswith("S1-")].set_index("group")
    other = df[~df["entity_id"].str.startswith("S1-")]
    for g, n, a in zip(other["group"], other["business_name"], other["business_address"]):
        if g in s1.index:
            yield s1.at[g, "business_name"], n, s1.at[g, "business_address"], a


def pairs_from_dataset(data_dir, chunksize=500_000):
    """Stream the full training data, keeping only records with Indian script."""
    tr = os.path.join(data_dir, "train")
    gt = read_tsv(os.path.join(tr, "train_ground_truth.tsv"))
    other_to_s1 = {}
    for s1_id, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        for x in ids.split(","):
            if x:
                other_to_s1[x] = s1_id
    del gt
    native_rows = []
    for k in (2, 3):
        for ch in iter_tsv(os.path.join(tr, f"train_source{k}.tsv"), chunksize=chunksize):
            mask = (ch["business_name"].str.contains(INDIC_RE, regex=True)
                    | ch["business_address"].str.contains(INDIC_RE, regex=True))
            native_rows.append(ch.loc[mask, ["entity_id", "business_name", "business_address"]])
            print(f"  source{k}: kept {int(mask.sum()):,} native-script rows", flush=True)
    import pandas as pd
    nat = pd.concat(native_rows, ignore_index=True)
    nat["s1"] = nat["entity_id"].map(other_to_s1)
    nat = nat.dropna(subset=["s1"])
    need = set(nat["s1"])
    s1_rec = {}
    for ch in iter_tsv(os.path.join(tr, "train_source1.tsv"), chunksize=chunksize):
        ch = ch[ch["entity_id"].isin(need)]
        s1_rec.update(zip(ch["entity_id"], zip(ch["business_name"], ch["business_address"])))
    for s1_id, n, a in zip(nat["s1"], nat["business_name"], nat["business_address"]):
        if s1_id in s1_rec:
            sn, sa = s1_rec[s1_id]
            yield sn, n, sa, a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", help="dataset folder containing train/")
    src.add_argument("--groups", help="sample_groups.tsv from the EDA")
    ap.add_argument("--out", default=DEFAULT_VOCAB)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--min-share", type=float, default=0.5,
                    help="min share of the token's pairs in which the English word co-occurs")
    ap.add_argument("--min-sim", type=float, default=0.4, help="min sound-alike score of a candidate")
    args = ap.parse_args()

    t0 = time.time()
    learner = VocabLearner(min_sim=args.min_sim)
    pairs = pairs_from_groups(args.groups) if args.groups else pairs_from_dataset(args.data_dir)
    for s1_name, o_name, s1_addr, o_addr in pairs:
        learner.add_pair(s1_name, o_name)
        learner.add_pair(s1_addr, o_addr)
    vocab = learner.build(args.min_count, args.min_share)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"_comment": "native-script token -> English word, learned from training matches "
                               "by src/preprocessing/translit_vocab.py",
                   "pairs_used": learner.pairs_seen, "vocab": dict(sorted(vocab.items()))},
                  f, ensure_ascii=False, indent=0)
    print(f"learned {len(vocab):,} entries from {learner.pairs_seen:,} text pairs "
          f"in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
