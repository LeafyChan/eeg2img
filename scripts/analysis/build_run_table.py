"""Build run_table.tsv: one row per NeuralBench run folder.

Usage: build_run_table.py <eeg2img_new_dir> <out.tsv>

Columns: hash8, folder, seed, batch_size, lr, n_epochs, dataset, log_file,
top5, top5_subj, top1_subj. Everything is extracted with tolerant regexes from
the saved config file inside each run folder and from the job logs; a value
that cannot be found is written as '?'. Reads files only (no imports of
torch / neuralbench), so it is safe on the login node.
"""
import glob, os, re, sys

H, OUT = sys.argv[1], sys.argv[2]
RUNS = os.path.join(H, "outputs", "neuralbench.main.Experiment.run,1")
LOGS = glob.glob(os.path.join(RUNS, "logs", "*", "*", "*_log.out"))

# 0.4.0 logs print metrics in a box table: names are cut at ~36 chars
# ("top5_acc_subjec..."), there is no colon, and each subject-agg row appears twice.
# The colon forms are kept for older (0.3.1) logs.
SCORE = {
    "top5": [r"test/full_retrieval/top5_acc\s*[│|]\s*([0-9.]+)",
             r"test/full_retrieval/top5_acc:\s*([0-9.]+)"],
    "top5_subj": [r"test/full_retrieval/top5_acc_subjec\S*\s*[│|]\s*([0-9.]+)",
                  r"test/full_retrieval/top5_acc_subject-agg:\s*([0-9.]+)"],
    "top1_subj": [r"test/full_retrieval/top1_acc_subjec\S*\s*[│|]\s*([0-9.]+)",
                  r"test/full_retrieval/top1_acc_subject-agg:\s*([0-9.]+)"],
}


def read(path):
    try:
        return open(path, errors="replace").read().replace("\r", "\n")
    except OSError:
        return ""


def first(pattern, text, flags=0):
    pats = pattern if isinstance(pattern, list) else [pattern]
    for pat in pats:
        m = re.search(pat, text, flags)
        if m:
            return m.group(1)
    return "?"


def biggest(pattern, text):
    """Largest integer matched (config has the training batch size AND the
    image-extractor batch size 4; the training one is the larger)."""
    v = [int(x) for x in re.findall(pattern, text)]
    return str(max(v)) if v else "?"


log_text = {p: read(p) + read(p.replace("_log.out", "_log.err")) for p in LOGS}
rows = []
for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
    name = os.path.basename(d)
    if name == "logs" or not os.path.isdir(d):
        continue
    h = re.findall(r"[0-9a-f]{8}", name)
    h8 = h[-1] if h else name[-8:]
    cfg = ""
    for f in ("config.yaml", "config.yml", "config.json"):
        cfg = read(os.path.join(d, f))
        if cfg:
            break
    row = {
        "hash8": h8, "folder": name,
        "seed": first([r"seed=(\d+)"], name) if "seed=" in name else first(r"(?m)^seed:\s*(\d+)", cfg),
        "batch_size": biggest(r"\bbatch_size\W+(\d+)", cfg),
        "lr": first(r"\blr\W+([0-9.eE-]+)", cfg),
        "n_epochs": first(r"\bn_epochs\W+(\d+)", cfg),
        "dataset": first(r"(xu20\d\d\w*|gifford\w*|grootswagers\w*)", cfg, re.I),
        "log_file": "?", "top5": "?", "top5_subj": "?", "top1_subj": "?",
    }
    # newest log that mentions this hash (a rerun of the same seed reuses the hash)
    hits = sorted((p for p, t in log_text.items() if h8 in t), key=os.path.getmtime, reverse=True)
    if hits:
        p = hits[0]; t = log_text[p]
        row["log_file"] = os.path.relpath(p, H)
        for k, pat in SCORE.items():
            row[k] = first(pat, t)
    rows.append(row)

cols = ["hash8", "folder", "seed", "batch_size", "lr", "n_epochs", "dataset",
        "log_file", "top5", "top5_subj", "top1_subj"]
with open(OUT, "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in rows:
        f.write("\t".join(str(r[c]) for c in cols) + "\n")
print(f"{len(rows)} run folders, {len(LOGS)} logs scanned -> {OUT}")
