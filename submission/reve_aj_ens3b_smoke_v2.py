import importlib.util, pathlib, random, torch

HERE = pathlib.Path(".").resolve()

def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

new = load_mod(HERE / "submission.py", "sub_new")
old = load_mod(HERE.parent / "reve_aj_ens3" / "submission.py", "sub_old")
TRAIN_CH, THINGS_IDX = new.TRAIN_CH, new.THINGS_IDX

def load(mod, n, ch=None):
    meta = dict(n_chans=n, n_outputs=1536, sfreq=120.0, submission_dir=HERE, device="cpu")
    if ch is not None:
        meta["ch_names"] = ch
    return mod.Solver().load_model(meta)

torch.manual_seed(0)
X63 = torch.randn(8, 63, 120)
X32 = X63[:, THINGS_IDX]

print("canon keeps training names:", all(new._canon(n) == n for n in TRAIN_CH))
m_new = load(new, 32, TRAIN_CH)
a_new = m_new.predict(X32)
a_old = load(old, 32, TRAIN_CH).predict(X32)
print("shape", tuple(a_new.shape), "finite", bool(torch.isfinite(a_new).all()))
print("new == old on exact path:", torch.allclose(a_new, a_old, atol=1e-6))
print("63-channel warm-up path == sealed path:", torch.allclose(a_new, load(new, 63).predict(X63), atol=1e-5))
perm = list(range(32)); random.Random(1).shuffle(perm)
print("shuffled names == original:", torch.allclose(a_new, load(new, 32, [TRAIN_CH[i] for i in perm]).predict(X32[:, perm]), atol=1e-5))
alt = [n.replace("AFz", "Afz").lower() for n in TRAIN_CH]
print("Afz / lower-case ok:", torch.allclose(a_new, load(new, 32, alt).predict(X32), atol=1e-5))
parts = torch.cat([m_new.predict(X32[:3]), m_new.predict(X32[3:])])
print("batch-size independent:", torch.allclose(a_new, parts, atol=1e-4))
print("no state between calls:", torch.allclose(a_new, m_new.predict(X32), atol=1e-6))

repl = {"FCz": "F7", "AFz": "F3", "F5": "F4", "F1": "F8", "F2": "PO9", "F6": "PO10", "CP3": "O9", "CP4": "O10"}
ch_alt = [repl.get(n, n) for n in TRAIN_CH]
try:
    b = load(new, 32, ch_alt).predict(X32)
    cos = torch.nn.functional.cosine_similarity(a_new, b, dim=-1).mean().item()
    print("fallback runs: shape", tuple(b.shape), "finite", bool(torch.isfinite(b).all()), "mean cos vs exact", round(cos, 3))
except Exception as e:
    print("fallback FAILED:", type(e).__name__, str(e)[:200])
try:
    load(new, 32, ["XX%d" % i for i in range(32)])
    print("garbage names did NOT raise  <-- BAD")
except ValueError:
    print("garbage names raise: OK")
