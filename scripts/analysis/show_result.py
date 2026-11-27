"""Print the metrics of the newest benchopt result under <repo>/outputs.

Usage: show_result.py <competition_repo> [run_log]
Falls back to grepping the run log if the parquet cannot be read.
"""
import glob, os, re, sys

repo = sys.argv[1]
log = sys.argv[2] if len(sys.argv) > 2 else None
# benchopt results live under an "outputs" folder; the data/ tree (symlinked to replay_work)
# holds NeuralBench cache parquets (events tables) that are NOT results.
files = sorted((f for f in glob.glob(os.path.join(repo, "**", "outputs", "**", "*.parquet"), recursive=True)
                if "/data/" not in f and "/cache/" not in f),
               key=os.path.getmtime)
ok = False
if files:
    try:
        from benchopt.results import read_results
        df = read_results(files[-1])
        print("result file:", files[-1])
        n = 0
        for c in df.columns:
            if c.startswith("objective_") and c not in ("objective_name", "objective_value"):
                print(f"  {c[len('objective_'):]:<26} {df[c].iloc[0]}"); n += 1
        ok = n > 0   # a parquet without metric columns is not a result
    except Exception as e:  # noqa: BLE001
        print("could not read parquet:", repr(e))
if not ok and log and os.path.exists(log):
    pat = re.compile(r"(top[15]_acc\S*|n_candidates)\W+([0-9.]+)")
    for line in open(log, errors="replace"):
        for m in pat.finditer(line):
            print("  (log)", m.group(1), m.group(2))
if not ok and not (log and os.path.exists(log)):
    print("no result found")
