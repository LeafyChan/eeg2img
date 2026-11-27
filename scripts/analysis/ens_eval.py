"""Offline seed-ensemble scorer for NeuralBench test predictions.

Usage: python ens_eval.py <preds_dir> <seed_dir> [<seed_dir> ...]
Mirrors tracks/image_decoding/objective.py of the competition repo:
raw predictions are averaged per (subject, image), THEN L2-normalised and
ranked against the unique target embeddings.
"""
import json, glob, sys, numpy as np, pandas as pd

D = sys.argv[1]; seeds = sys.argv[2:]


def load(d):
    ents = [json.loads(l) for l in open(glob.glob(f"{d}/*info.jsonl")[0]) if l.strip()]
    P, T = {}, {}
    for e in ents:
        k = e.get('#key', '')
        if 'dtype' not in e or not (k.startswith('y_pred') or k.startswith('y_true')):
            continue
        a = np.array(np.memmap(f"{d}/{e['filename']}", dtype=e['dtype'], mode='r',
                               offset=e['offset'], shape=tuple(e['shape'])))
        (P if k.startswith('y_pred') else T)[k.split('_')[-1]] = a
    ks = sorted(P)
    return np.concatenate([P[k] for k in ks]), np.concatenate([T[k] for k in ks])


def nrm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


preds = {}; Y = None; META = None
for s in seeds:
    p, t = load(f"{D}/{s}")
    m = pd.read_csv(glob.glob(f"{D}/{s}/data/metadata-*.csv")[0])
    print(s, "pred", p.shape, "true", t.shape, "meta", m.shape)
    if Y is None:
        Y, META = t, m
    else:
        print("  y_true identical:", np.array_equal(Y, t), "| metadata identical:", m.equals(META))
    preds[s] = p
col = next(c for c in ("subject_id", "subject") if c in META.columns)
subj = pd.factorize(META[col])[0]
G, tidx = np.unique(Y, axis=0, return_inverse=True)
tidx = tidx.ravel(); Gn = nrm(G)
print("gallery:", len(G), "| windows:", len(Y), "| subject groups:", len(np.unique(subj)))
gid_all = subj * 1000 + tidx


def score(pred, sel):
    g = gid_all[sel]; u, inv = np.unique(g, return_inverse=True)
    M = np.zeros((len(u), pred.shape[1]), np.float64)
    np.add.at(M, inv, pred[sel].astype(np.float64))
    tg = np.zeros(len(u), int); tg[inv] = tidx[sel]
    S = nrm(M) @ Gn.T; ts = S[np.arange(len(u)), tg]
    r = (S > ts[:, None]).sum(1)
    return (r < 5).mean(), (r < 1).mean(), len(u)


def single(pred):
    S = nrm(pred) @ Gn.T; ts = S[np.arange(len(pred)), tidx]
    return ((S > ts[:, None]).sum(1) < 5).mean()


def subsample(k, rng):
    N = len(Y); perm = rng.permutation(N); g = gid_all[perm]
    o = np.argsort(g, kind='stable'); gs = g[o]
    pos = np.arange(N) - np.searchsorted(gs, gs)
    return perm[o][pos < k]


allsel = np.arange(len(Y))
members = {s: nrm(p) for s, p in preds.items()}
ens = nrm(sum(members.values()))
print("\n== full test: top5 subj-agg / top1 subj-agg / queries / single-window top5")
for s in seeds:
    a = score(preds[s], allsel)
    print(f"{s} raw : {a[0]:.4f} {a[1]:.4f} {a[2]}   single {single(preds[s]):.4f}")
e = score(ens, allsel)
print(f"ENSEMBLE ({len(seeds)}): {e[0]:.4f} {e[1]:.4f} {e[2]}   single {single(ens):.4f}")
print("mean of singles (raw): %.4f" % np.mean([score(preds[s], allsel)[0] for s in seeds]))

print("\n== repeat sweep, mean of 5 draws (top5 / top1)")
print("k      single-seed-mean      ensemble")
for k in (1, 5, 10, 20, 40):
    rng = np.random.default_rng(0); sm, em = [], []
    for _ in range(5):
        sel = subsample(k, rng)
        sm.append([np.mean([score(members[s], sel)[i] for s in seeds]) for i in (0, 1)])
        em.append(score(ens, sel)[:2])
    sm = np.mean(sm, 0); em = np.mean(em, 0)
    print(f"{k:<6} {sm[0]:.4f} / {sm[1]:.4f}      {em[0]:.4f} / {em[1]:.4f}")
