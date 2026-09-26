"""Memory / runtime benchmark of TSV loading engines + normalisation throughput.

Each engine runs in a FRESH subprocess so its peak memory (RSS) is measured
in isolation. Every engine does the same job: read one source file with all
columns as strings (quoting off), then compute rows-per-country and the mean
name length - enough to force the data to actually be materialised.

Usage (from project root, on the machine with the data):
    python -m src.io.benchmark --file <DATASET_DIR>/train/train_source2.tsv
    python -m src.io.benchmark --file <DATASET_DIR>/train/train_source2.tsv --engines pandas_c,polars
Writes docs/BENCHMARK.md (and prints the table).

Engines that are not installed are skipped (pip install polars duckdb pyarrow psutil).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

ENGINES = ["pandas_c", "pandas_chunked", "pyarrow", "pyarrow_to_pandas_arrow",
           "polars", "polars_lazy", "duckdb", "parquet_read"]

# --------------------------------------------------------------------- engine bodies
CODE = {
"pandas_c": r"""
import csv, pandas as pd
df = pd.read_csv(PATH, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False, na_values=[])
rows = len(df); cc = df["country"].value_counts().to_dict(); ml = df["business_name"].str.len().mean()
""",
"pandas_chunked": r"""
import csv, pandas as pd
from collections import Counter
rows = 0; cc = Counter(); tot = 0
for ch in pd.read_csv(PATH, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False,
                      na_values=[], chunksize=500_000):
    rows += len(ch); cc.update(ch["country"].value_counts().to_dict()); tot += ch["business_name"].str.len().sum()
ml = tot / rows; cc = dict(cc)
""",
"pyarrow": r"""
import pyarrow as pa, pyarrow.csv as pc, pyarrow.compute as pcm
t = pc.read_csv(PATH, parse_options=pc.ParseOptions(delimiter="\t", quote_char=False),
                convert_options=pc.ConvertOptions(column_types={c: pa.string() for c in
                    ["entity_id","business_name","business_address","country"]}, strings_can_be_null=False))
rows = t.num_rows
vc = pcm.value_counts(t["country"]); cc = {d["values"]: d["counts"] for d in vc.to_pylist()}
ml = pcm.mean(pcm.utf8_length(t["business_name"])).as_py()
""",
"pyarrow_to_pandas_arrow": r"""
import pandas as pd, pyarrow as pa, pyarrow.csv as pc
t = pc.read_csv(PATH, parse_options=pc.ParseOptions(delimiter="\t", quote_char=False),
                convert_options=pc.ConvertOptions(column_types={c: pa.string() for c in
                    ["entity_id","business_name","business_address","country"]}, strings_can_be_null=False))
df = t.to_pandas(types_mapper=pd.ArrowDtype); del t
rows = len(df); cc = df["country"].value_counts().to_dict(); ml = float(df["business_name"].str.len().mean())
""",
"polars": r"""
import polars as pl
df = pl.read_csv(PATH, separator="\t", quote_char=None, infer_schema_length=0)
rows = df.height
cc = dict(df.group_by("country").len().iter_rows())
ml = df["business_name"].str.len_chars().mean()
""",
"polars_lazy": r"""
import polars as pl
lf = pl.scan_csv(PATH, separator="\t", quote_char=None, infer_schema_length=0)
res = lf.group_by("country").agg(pl.len().alias("n"), pl.col("business_name").str.len_chars().sum().alias("s")).collect()
rows = int(res["n"].sum()); cc = dict(zip(res["country"], res["n"])); ml = res["s"].sum() / rows
""",
"duckdb": r"""
import duckdb
con = duckdb.connect()
q = f"SELECT country, count(*) n, sum(length(business_name)) s FROM read_csv('{PATH}', delim='\t', quote='', escape='', header=true, all_varchar=true) GROUP BY 1"
res = con.execute(q).fetchall()
rows = sum(r[1] for r in res); cc = {r[0]: r[1] for r in res}; ml = sum(r[2] for r in res) / rows
""",
"parquet_read": r"""
import pandas as pd
df = pd.read_parquet(PARQUET)
rows = len(df); cc = df["country"].value_counts().to_dict(); ml = df["business_name"].str.len().mean()
""",
}

RUNNER = r"""
import json, sys, time
PATH = {path!r}; PARQUET = {parquet!r}
t0 = time.time()
{body}
print("__RESULT__" + json.dumps({{"seconds": time.time() - t0, "rows": int(rows),
      "countries": {{str(k): int(v) for k, v in cc.items()}}, "mean_name_len": float(ml)}}))
"""


def _peak_rss(proc, out):
    try:
        import psutil
    except ImportError:
        out["peak_mb"] = None
        return
    peak = 0
    try:
        p = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                rss = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
                peak = max(peak, rss)
            except psutil.Error:
                pass
            time.sleep(0.05)
    except psutil.Error:
        pass
    out["peak_mb"] = peak / 2**20


def run_engine(name, path, parquet):
    code = RUNNER.format(path=os.path.abspath(path).replace("\\", "/"),
                         parquet=os.path.abspath(parquet).replace("\\", "/") if parquet else "",
                         body=CODE[name])
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    mem = {}
    th = threading.Thread(target=_peak_rss, args=(proc, mem))
    th.start()
    stdout, stderr = proc.communicate()
    th.join()
    wall = time.time() - t0
    res = next((json.loads(l[10:]) for l in stdout.splitlines() if l.startswith("__RESULT__")), None)
    if res is None:
        err = (stderr.strip().splitlines() or ["unknown error"])[-1]
        return {"engine": name, "ok": False, "error": err[:150]}
    res.update(engine=name, ok=True, wall_seconds=wall, peak_mb=mem.get("peak_mb"))
    return res


def write_parquet(path, parquet):
    import pyarrow as pa
    import pyarrow.csv as pc
    import pyarrow.parquet as pq
    t0 = time.time()
    t = pc.read_csv(path, parse_options=pc.ParseOptions(delimiter="\t", quote_char=False),
                    convert_options=pc.ConvertOptions(strings_can_be_null=False,
                                                      column_types={c: pa.string() for c in
                                                                    ["entity_id", "business_name",
                                                                     "business_address", "country"]}))
    pq.write_table(t, parquet, compression="zstd")
    return time.time() - t0


def bench_normalize(path, n_rows, workers):
    """Throughput of the normaliser on the first n_rows of the file."""
    from src.io.data_loader import read_tsv
    from src.preprocessing.preprocess import normalize_df_parallel
    df = read_tsv(path, nrows=n_rows)
    out = {}
    for w in sorted({1, workers}):
        t0 = time.time()
        normalize_df_parallel(df, workers=w)
        dt = time.time() - t0
        out[w] = len(df) / dt
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="a source TSV, e.g. train_source2.tsv")
    ap.add_argument("--engines", default=",".join(ENGINES))
    ap.add_argument("--normalize-rows", type=int, default=100_000, help="0 to skip")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join("docs", "BENCHMARK.md"))
    args = ap.parse_args()

    size_mb = os.path.getsize(args.file) / 2**20
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    parquet, pq_secs = None, None
    if "parquet_read" in engines:
        parquet = os.path.splitext(args.file)[0] + ".bench.parquet"
        try:
            pq_secs = write_parquet(args.file, parquet)
        except Exception as e:  # pyarrow missing
            print("parquet conversion skipped:", e)
            engines.remove("parquet_read")

    results = []
    for e in engines:
        print(f"running {e} ...", flush=True)
        r = run_engine(e, args.file, parquet)
        results.append(r)
        if r["ok"]:
            print(f"   {r['seconds']:.1f}s, peak {r['peak_mb'] or float('nan'):.0f} MB, rows {r['rows']:,}")
        else:
            print("   FAILED:", r["error"])

    rows_ok = {r["rows"] for r in results if r["ok"]}
    lines = [f"# Loading benchmark", "",
             f"File: `{os.path.basename(args.file)}` ({size_mb:,.0f} MB). "
             f"Machine: {os.cpu_count()} logical CPUs, Python {sys.version.split()[0]}.", "",
             "| engine | ok | load+aggregate s | wall s (incl. startup) | peak RAM MB | rows |",
             "|---|---|---|---|---|---|"]
    for r in results:
        if r["ok"]:
            pk = f"{r['peak_mb']:,.0f}" if r["peak_mb"] else "n/a (pip install psutil)"
            lines.append(f"| {r['engine']} | yes | {r['seconds']:.1f} | {r['wall_seconds']:.1f} | {pk} | {r['rows']:,} |")
        else:
            lines.append(f"| {r['engine']} | **no** | | | | {r['error']} |")
    lines.append("")
    lines.append("All engines returned the same row count: " + ("**yes**" if len(rows_ok) == 1 else f"**NO** {rows_ok}"))
    if parquet:
        lines.append(f"\nTSV -> Parquet (zstd) conversion took {pq_secs:.1f}s; parquet size "
                     f"{os.path.getsize(parquet) / 2**20:,.0f} MB vs {size_mb:,.0f} MB TSV.")
        os.remove(parquet)

    if args.normalize_rows:
        print("benchmarking normalisation ...", flush=True)
        thr = bench_normalize(args.file, args.normalize_rows, args.workers)
        lines += ["", "## Normalisation throughput", "",
                  f"First {args.normalize_rows:,} rows of the file.", "",
                  "| worker processes | rows / second |", "|---|---|"]
        lines += [f"| {w} | {v:,.0f} |" for w, v in thr.items()]
        best = max(thr.values())
        lines.append(f"\nEstimated time to normalise all ~24M train+test rows at the best rate: "
                     f"**{24e6 / best / 60:.0f} min**.")

    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print("written to", args.out)


if __name__ == "__main__":
    main()
