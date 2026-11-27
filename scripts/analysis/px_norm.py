import json, glob, os, sys, numpy as np, pandas as pd
D = sys.argv[1]; names = sys.argv[2:]
def nrm(x): return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
def items(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*info.jsonl")):
        for l in open(f):
            e = json.loads(l); fn = os.path.join(d, e.get("filename", ""))
            if "dtype" in e and os.path.isfile(fn):
                out[e["#key"]] = np.array(np.memmap(fn, dtype=e["dtype"], mode="r", offset=e["offset"], shape=tuple(e["shape"])))
    return out
def load(name):
    it = items(f"{D}/{name}"); ks = sorted(k for k in it if k.startswith("y_pred"))
    P = np.concatenate([it[k] for k in ks]); T = np.concatenate([it["y_true_" + k.split("_")[-1]] for k in ks])
    return P, T, pd.read_csv(glob.glob(f"{D}/{name}/data/metadata-*.csv")[0])
L = [load(n) for n in names]; Y, META = L[0][1], L[0][2]
for P, T, M in L[1:]:
    if not np.array_equal(Y, T): sys.exit("y_true differs: stop")
Ps = [x[0].astype(np.float32) for x in L]
col = next(c for c in ("subject_id", "subject") if c in META.columns); subj = pd.factorize(META[col])[0]
Gt, tidx = np.unique(Y, axis=0, return_inverse=True); tidx = tidx.ravel(); gn = nrm(Gt).astype(np.float32)
gid = subj * 100000 + tidx; u, inv = np.unique(gid, return_inverse=True); tg = np.zeros(len(u), int); tg[inv] = tidx
def ranks(P):
    M = np.zeros((len(u), P.shape[1])); np.add.at(M, inv, P.astype(np.float64))
    S = nrm(M).astype(np.float32) @ gn.T
    return (S > S[np.arange(len(u)), tg][:, None]).sum(1)
def view(P, a): return P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12) ** (1 - a)   # a=0 unit norm, a=1 raw
N = len(Gt)
for n, P in zip(names, Ps):
    nr = np.linalg.norm(P, axis=1); cs = (nrm(P) * nrm(Y)).sum(1)
    print(f"{n}: norm mean {nr.mean():.3f} cv {nr.std() / nr.mean():.3f} | corr(norm, cos to target) {np.corrcoef(nr, cs)[0, 1]:+.3f}")
rng = np.random.default_rng(0); idx = rng.integers(0, len(u), (2000, len(u))); base = {}
print(f"\nqueries {len(u)} | gallery {N}  (alpha 0 = unit-norm outputs, 1 = raw outputs)")
for a in (0, 0.5, 1, 2):
    rs = [ranks(view(P, a)) for P in Ps]; re_ = ranks(sum(view(P, a) for P in Ps))
    m5 = np.mean([(r < 5).sum() for r in rs]); m50 = np.mean([(r < 50).mean() for r in rs]); mp = np.mean([1 - r.mean() / N for r in rs])
    print(f"alpha={a:<4} singles: top5 hits {m5:6.1f} top50 {m50:.4f} pct {mp:.4f} | ens: top5 hits {int((re_ < 5).sum())} top50 {(re_ < 50).mean():.4f} pct {1 - re_.mean() / N:.4f}")
    if a == 0: base = {"ens": re_}
    else:
        for nm, x, y in (("top5", (re_ < 5) * 1.0, (base["ens"] < 5) * 1.0), ("top50", (re_ < 50) * 1.0, (base["ens"] < 50) * 1.0), ("pct", 1 - re_ / N, 1 - base["ens"] / N)):
            d = x - y; bs = d[idx].mean(1)
            print(f"      ens vs alpha 0  {nm:<5} diff {d.mean():+.4f} CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
