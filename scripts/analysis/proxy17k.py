import json, glob, sys, numpy as np, pandas as pd
D = sys.argv[1]; seeds = sys.argv[2:]
def nrm(x): return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
def load(d):
    ents = [json.loads(l) for l in open(glob.glob(f"{d}/*info.jsonl")[0]) if l.strip()]
    P, T = {}, {}
    for e in ents:
        k = e.get('#key', '')
        if 'dtype' not in e or not (k.startswith('y_pred') or k.startswith('y_true')): continue
        a = np.array(np.memmap(f"{d}/{e['filename']}", dtype=e['dtype'], mode='r', offset=e['offset'], shape=tuple(e['shape'])))
        (P if k.startswith('y_pred') else T)[k.split('_')[-1]] = a
    ks = sorted(P); return np.concatenate([P[k] for k in ks]), np.concatenate([T[k] for k in ks])

Z = np.load("/path/to/home/aj_emb.npz"); X = Z["X"].astype(np.float32)
Y = None; E = 0
for s in seeds:
    p, t = load(f"{D}/{s}"); 
    if Y is None:
        Y = t; m = pd.read_csv(glob.glob(f"{D}/{s}/data/metadata-*.csv")[0])
    E = E + nrm(p)
E = nrm(E)
col = next(c for c in ("subject_id", "subject") if c in m.columns)
subj = pd.factorize(m[col])[0]
G, tidx = np.unique(Y, axis=0, return_inverse=True); tidx = tidx.ravel()
sim = nrm(G) @ nrm(X).T; match = sim.argmax(1)
print("test images matched:", len(set(match)), "of", len(G), "min cos", sim.max(1).min())
mask = np.ones(len(X), bool); mask[match] = False
Gall = np.concatenate([X[match], X[mask]]); Gn = nrm(Gall)   # first 200 = targets, in G order
print("gallery:", len(Gall), "| windows:", len(Y), "| subjects:", len(np.unique(subj)))
chat = nrm(Gall.mean(0, keepdims=True))[0]
gid = subj * 1000 + tidx

def score(Q, sel):
    g = gid[sel]; u, inv = np.unique(g, return_inverse=True)
    M = np.zeros((len(u), Q.shape[1]), np.float64); np.add.at(M, inv, Q[sel].astype(np.float64))
    tg = np.zeros(len(u), int); tg[inv] = tidx[sel]
    S = nrm(M) @ Gn.T; ts = S[np.arange(len(u)), tg]
    r = (S > ts[:, None]).sum(1); return (r < 5).mean(), (r < 1).mean()
def subsample(k, rng):
    N = len(Y); perm = rng.permutation(N); g = gid[perm]
    o = np.argsort(g, kind='stable'); gs = g[o]
    pos = np.arange(N) - np.searchsorted(gs, gs); return perm[o][pos < k]

methods = {"base": E}
for a in (0.25, 0.5, 1.0, 2.0): methods[f"sub{a}"] = E - a * chat
for b in (0.5, 1.0): methods[f"proj{b}"] = E - b * (E @ chat)[:, None] * chat
ks = (1, 3, 5, 10, None); draws = 5
res = {}
for name, Q in methods.items():
    row = []
    for k in ks:
        if k is None: row.append(score(Q, np.arange(len(Y)))[0])
        else:
            rng = np.random.default_rng(0); row.append(np.mean([score(Q, subsample(k, rng))[0] for _ in range(draws)]))
    res[name] = row
print("\nTop-5 subj-agg on 16.7k gallery (chance %.5f)" % (5 / len(Gall)))
print("method      k=1       k=3       k=5       k=10      full   | ratio to base at k=3 / 5 / full")
b = res["base"]
for n, r in res.items():
    print(f"{n:<10} " + " ".join(f"{v:.5f}  " for v in r) + f"| {r[1]/b[1]:.2f} {r[2]/b[2]:.2f} {r[4]/b[4]:.2f}")
