import json, glob, os, sys, numpy as np, pandas as pd
D = sys.argv[1]; specs = sys.argv[2:]          # label=dirA,dirB  (several dirs = embedding ensemble)
def nrm(x): return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
def items(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*info.jsonl")):
        for l in open(f):
            e = json.loads(l)
            fn = os.path.join(d, e.get("filename", ""))
            if "dtype" in e and os.path.isfile(fn):
                out[e["#key"]] = np.array(np.memmap(fn, dtype=e["dtype"], mode="r", offset=e["offset"], shape=tuple(e["shape"])))
    return out
def load(name):
    it = items(f"{D}/{name}"); ks = sorted(k for k in it if k.startswith("y_pred"))
    P = np.concatenate([it[k] for k in ks]); T = np.concatenate([it["y_true_" + k.split("_")[-1]] for k in ks])
    return P, T, pd.read_csv(glob.glob(f"{D}/{name}/data/metadata-*.csv")[0])
Y = META = None; runs = {}
for sp in specs:
    label, dirs = sp.split("=")
    ps = []
    for dn in dirs.split(","):
        P, T, M = load(dn)
        if Y is None: Y, META = T, M
        elif not np.array_equal(Y, T): sys.exit("y_true differs in " + dn)
        ps.append(nrm(P))
    runs[label] = nrm(sum(ps))
col = next(c for c in ("subject_id", "subject") if c in META.columns)
subj = pd.factorize(META[col])[0]
Gt, tidx = np.unique(Y, axis=0, return_inverse=True); tidx = tidx.ravel(); gn = nrm(Gt).astype(np.float32)
gid = subj * 100000 + tidx; u, inv = np.unique(gid, return_inverse=True)
tg = np.zeros(len(u), int); tg[inv] = tidx
def ranks(P):
    M = np.zeros((len(u), P.shape[1])); np.add.at(M, inv, P.astype(np.float64))
    S = nrm(M).astype(np.float32) @ gn.T
    return (S > S[np.arange(len(u)), tg][:, None]).sum(1)
N = len(Gt); print(f"queries {len(u)} | gallery {N} | chance top5 {5 / N:.4f} | windows {len(Y)}")
R = {k: ranks(v) for k, v in runs.items()}
for k, r in R.items():
    print(f"{k:<12} top1 {(r < 1).mean():.4f} top5 {(r < 5).mean():.4f} ({int((r < 5).sum())} hits) top50 {(r < 50).mean():.4f} pct {1 - r.mean() / N:.4f}")
ref = list(R)[0]; rng = np.random.default_rng(0); idx = rng.integers(0, len(u), (2000, len(u)))
print(f"\npaired differences vs {ref} (95% bootstrap CI over queries)")
for k, r in R.items():
    if k == ref: continue
    for nm, a, b in (("top5", (r < 5) * 1.0, (R[ref] < 5) * 1.0), ("top50", (r < 50) * 1.0, (R[ref] < 50) * 1.0), ("pct", 1 - r / N, 1 - R[ref] / N)):
        d = a - b; bs = d[idx].mean(1)
        print(f"{k:<12} {nm:<6} diff {d.mean():+.4f} CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
