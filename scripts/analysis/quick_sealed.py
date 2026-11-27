import importlib.util, pathlib, random, torch
HERE = pathlib.Path(".").resolve()
spec = importlib.util.spec_from_file_location("s", HERE / "submission.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
TC, TI = m.TRAIN_CH, m.THINGS_IDX
def load(n, ch=None):
    meta = dict(n_chans=n, n_outputs=1536, sfreq=120.0, submission_dir=HERE, device="cpu")
    if ch is not None: meta["ch_names"] = ch
    return m.Solver().load_model(meta)
torch.manual_seed(0); X = torch.randn(8, 32, 120)
mod = load(32, TC); a = mod.predict(X)
print("members", len(mod.members), "shape", tuple(a.shape), "finite", bool(torch.isfinite(a).all()))
print("batch-independent:", torch.allclose(a, torch.cat([mod.predict(X[:3]), mod.predict(X[3:])]), atol=1e-4))
print("no state between calls:", torch.allclose(a, mod.predict(X), atol=1e-6))
perm = list(range(32)); random.Random(1).shuffle(perm)
print("shuffled names == orig:", torch.allclose(a, load(32, [TC[i] for i in perm]).predict(X[:, perm]), atol=1e-5))
X63 = torch.randn(8, 63, 120); X63[:, TI] = X
print("63-channel path == name path:", torch.allclose(a, load(63).predict(X63), atol=1e-5))
