"""Submission writers and checks (Member 4).

Both files: TAB separated, header row, one row per test Source 1 entity (in
test_source1 order), comma-separated ids with no quoting, empty when none.
"""
from __future__ import annotations

import os
import subprocess
import sys

MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CAND_HEADER = ("source1_entity_id", "candidate_entity_ids")


def write_tsv(data, path):
    """Legacy helper."""
    data.to_csv(path, sep="\t", index=False)


def write_id_lists(path: str, header, s1_ids, lists: dict):
    """Write one row per id in s1_ids; lists[s1] is an iterable of ids (deduplicated here)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s in s1_ids:
            ids = list(dict.fromkeys(lists.get(s, ())))
            f.write(f"{s}\t{','.join(ids)}\n")


class StreamingIdListWriter:
    """Append rows chunk by chunk (used for candidate_pairs.tsv on 1.7M S1)."""

    def __init__(self, path, header):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8", newline="\n")
        self.f.write("\t".join(header) + "\n")
        self.rows = 0

    def write(self, s1_ids, lists: dict):
        for s in s1_ids:
            ids = list(dict.fromkeys(lists.get(s, ())))
            self.f.write(f"{s}\t{','.join(ids)}\n")
            self.rows += 1

    def close(self):
        self.f.close()


def read_id_lists(path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s, _, ids = line.rstrip("\n").partition("\t")
            out[s] = [x for x in ids.split(",") if x]
    return out


def check_submission(matching_path, candidate_path, s1_ids) -> list[str]:
    """Our own checks (mirrors the official rules). Returns a list of problems."""
    problems = []
    s1_ids = list(s1_ids)
    s1_set = set(s1_ids)
    for path, header in ((matching_path, MATCH_HEADER), (candidate_path, CAND_HEADER)):
        if path is None:
            continue
        with open(path, encoding="utf-8") as f:
            if f.readline().rstrip("\n").split("\t") != list(header):
                problems.append(f"{os.path.basename(path)}: wrong header")
        lists = read_id_lists(path)
        if len(lists) != len(s1_set) or set(lists) != s1_set:
            problems.append(f"{os.path.basename(path)}: S1 rows {len(lists)} vs expected {len(s1_set)}")
        bad_prefix = sum(1 for v in lists.values() for x in v if not (x.startswith("S2-") or x.startswith("S3-")))
        dups = sum(1 for v in lists.values() if len(v) != len(set(v)))
        if bad_prefix:
            problems.append(f"{os.path.basename(path)}: {bad_prefix} ids are not S2-/S3-")
        if dups:
            problems.append(f"{os.path.basename(path)}: {dups} rows with duplicate ids")
    if matching_path and candidate_path:
        m, c = read_id_lists(matching_path), read_id_lists(candidate_path)
        outside = sum(1 for s, v in m.items() for x in v if x not in set(c.get(s, ())))
        if outside:
            problems.append(f"{outside} matched ids are not in candidate_pairs.tsv")
    return problems


def run_official_validator(validator_path, matching_path, candidate_path, test_dir) -> int:
    """Run utils/validate_submission.py from the student_resource; returns its exit code."""
    if not validator_path or not os.path.exists(validator_path):
        print(f"official validator not found at {validator_path} - skipped")
        return -1
    cmd = [sys.executable, validator_path, "--matching", matching_path, "--test-dir", test_dir]
    if candidate_path:
        cmd += ["--candidate", candidate_path]
    print("running:", " ".join(cmd))
    res = subprocess.run(cmd)
    return res.returncode
