"""End-to-end driver: indexes -> train -> predict -> submission.

  python -m src.pipeline                    # everything, variant 1 submission
  python -m src.pipeline --skip-indexes     # indexes already built
  python -m src.pipeline --stage predict    # only test scoring + submission

Stages (each is also runnable on its own):
  indexes    python -m src.preprocessing.build_indexes --config configs/pipeline.yaml
  train      python -m src.train
  predict    python -m src.predict
  submission python -m src.make_submission --variant N

Prerequisite: python -m src.preprocessing.preprocess --data-dir <DATASET_DIR>
(the Phase-1 rule baseline lives in src/pipeline_baseline.py)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

STAGES = ["indexes", "train", "predict", "submission"]


def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    t = time.time()
    code = subprocess.run(cmd).returncode
    print(f"<<< exit {code} after {(time.time() - t) / 60:.1f} min", flush=True)
    if code != 0:
        raise SystemExit(f"stage failed: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--stage", choices=STAGES, default=None, help="start from this stage")
    ap.add_argument("--skip-indexes", action="store_true")
    ap.add_argument("--variant", type=int, default=1)
    a = ap.parse_args()
    start = STAGES.index(a.stage) if a.stage else (1 if a.skip_indexes else 0)
    py = sys.executable
    for st in STAGES[start:]:
        if st == "indexes":
            run([py, "-m", "src.preprocessing.build_indexes", "--config", a.config])
        elif st == "train":
            run([py, "-m", "src.train", "--config", a.config])
        elif st == "predict":
            run([py, "-m", "src.predict", "--config", a.config])
        else:
            run([py, "-m", "src.make_submission", "--config", a.config, "--variant", str(a.variant)])


if __name__ == "__main__":
    main()
