import json, glob, os, sys, numpy as np, pandas as pd
CACHE = sys.argv[1]; D = sys.argv[2]; seeds = sys.argv[3:]
def nrm(x): return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

def read_items(d):
    out = {}
    for f in glob.glob(os.path.join(d, "**", "*info.jsonl"), recursive=True):
        base = os.path.dirname(f)
        for l in open(f):
            if not l.strip(): continue
            e = json.loads(l)
            fn = os.path.join(base, e.get("filename", ""))
            if "dtype" not in e or not os.path.isfile(fn): continue
            a = np.array(np.memmap(fn, dtype=e["dtype"], mode="r", offset=e["offset"], shape=tuple(e["shape"])))
            out[str(e.get("#key", f))] = a
    return out

def load_preds(d):
    P, T = {}, {}
    for k, a in read_items(d).items():
        if k.startswith("y_pred"): P[k.split("_")[-1]] = a
        elif k.startswith("y_true"): T[k.split("_")[-1]] = a
    ks = sorted(P)
    return np.concatenate([P[k] for k in ks]), np.concatenate([T[k] for k in ks])

preds, Y, META = {}, None, None
for s in seeds:
    p, t = load_preds(f"{D}/{s}")
    m = pd.read_csv(glob.glob(f"{D}/{s}/data/metadata-*.csv")[0])
    print(s, "pred", p.shape, "true", t.shape)
    if Y is None: Y, META = t, m
    elif not np.array_equal(Y, t): sys.exit("y_true differs between seeds: stop")
    preds[s] = p
col = next(c for c in ("subject_id", "subject") if c in META.columns)
subj = pd.factorize(META[col])[0]
Gt, tidx = np.unique(Y, axis=0, return_inverse=True); tidx = tidx.ravel()
items = read_items(CACHE)
vecs = {k: v.reshape(-1) for k, v in items.items() if v.size == Y.shape[1]}
print("cache items:", len(items), "| 1536-d items:", len(vecs)); print("example keys:", list(vecs)[:3])
ak = [k for k in vecs if "nm000134" in k]
use = ak if len(ak) > 1000 else list(vecs)
print("gallery source keys:", len(use), "(Alljoined paths)" if ak else "(WARNING: all cache items; duplicates possible)")
Gal = np.stack([vecs[k] for k in use]).astype(np.float32)
S0 = nrm(Gt) @ nrm(Gal).T; hit = np.unique(S0.argmax(1)[S0.max(1) > 0.9999])
print("test targets found in cache:", int((S0.max(1) > 0.9999).sum()), "of", len(Gt))
keep = np.setdiff1d(np.arange(len(Gal)), hit)
G = np.concatenate([Gt.astype(np.float32), Gal[keep]]); gn = nrm(G).astype(np.float32)
print("gallery size:", len(G), "| windows:", len(Y), "| subjects:", len(np.unique(subj)))
gid = subj * 100000 + tidx

def score(P, sel):
    u, inv = np.unique(gid[sel], return_inverse=True)
    M = np.zeros((len(u), P.shape[1])); np.add.at(M, inv, P[sel].astype(np.float64))
    tg = np.zeros(len(u), int); tg[inv] = tidx[sel]
    S = nrm(M).astype(np.float32) @ gn.T; ts = S[np.arange(len(u)), tg]
    return (S > ts[:, None]).sum(1)

def summ(r):
    n, N = len(r), len(G)
    return f"top5 {(r < 5).mean():.5f} ({int((r < 5).sum())}/{n} hits) top50 {(r < 50).mean():.4f} top500 {(r < 500).mean():.4f} pct {1 - r.mean() / N:.4f}"

def subsample(sel0, k, rng):
    sel0 = rng.permutation(sel0); g = gid[sel0]; o = np.argsort(g, kind="stable"); gs = g[o]
    return sel0[o][(np.arange(len(sel0)) - np.searchsorted(gs, gs)) < k]
mem = {s: nrm(p) for s, p in preds.items()}
ens = nrm(sum(mem.values()))
rng = np.random.default_rng(0); perm = rng.permutation(len(Y)); A, B = perm[: len(Y) // 2], perm[len(Y) // 2:]
c = ens[A].mean(0)

allsel = np.arange(len(Y))
print("\n== full test, large gallery (subject-aggregated); chance top5 = %.5f, top500 = %.4f" % (5 / len(G), 500 / len(G)))
for s in seeds: print(f"{s}: {summ(score(mem[s], allsel))}")
print(f"ENSEMBLE({len(seeds)}): {summ(score(ens, allsel))}")
print("\n== repeat sweep, mean of 4 draws (top500 / pct): k | single-seed-mean | ensemble")
for k in (1, 3, 5, 10, 20):
    sm, em = [], []
    for _ in range(4):
        sel = subsample(allsel, k, rng)
        rs = [score(mem[s], sel) for s in seeds]; re_ = score(ens, sel)
        sm.append([np.mean([(r < 500).mean() for r in rs]), np.mean([1 - r.mean() / len(G) for r in rs])])
        em.append([(re_ < 500).mean(), 1 - re_.mean() / len(G)])
    sm, em = np.mean(sm, 0), np.mean(em, 0)
    print(f"k={k:<3} {sm[0]:.4f} / {sm[1]:.4f}   {em[0]:.4f} / {em[1]:.4f}")
print("\n== centering test: constant c from half A of the windows, scored on half B")
for alpha in (0.0, 0.25, 0.5, 0.75, 1.0): print(f"alpha={alpha}: {summ(score(ens - alpha * c, B))}")
