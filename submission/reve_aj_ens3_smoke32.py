import torch, pathlib, random
from submission import Solver, TRAIN_CH, THINGS_IDX
D = pathlib.Path(".").resolve()
def load(n, ch=None):
    meta = dict(n_chans=n, n_outputs=1536, sfreq=120.0, submission_dir=D, device="cpu")
    if ch is not None: meta["ch_names"] = ch
    return Solver().load_model(meta)
torch.manual_seed(0)
X63 = torch.randn(4, 63, 120); X32 = X63[:, THINGS_IDX]
a = load(32, TRAIN_CH).predict(X32)      # sealed path, names in training order
b = load(63).predict(X63)                # warm-up path, no names
print("shape", tuple(a.shape), "finite", bool(torch.isfinite(a).all()))
print("warm-up path == sealed path:", torch.allclose(a, b, atol=1e-5))
perm = list(range(32)); random.Random(1).shuffle(perm)
c = load(32, [TRAIN_CH[i] for i in perm]).predict(X32[:, perm])
print("shuffled channels and names == original:", torch.allclose(a, c, atol=1e-5))
alt = [n.replace("AFz", "Afz").lower() for n in TRAIN_CH]
print("Afz / lower-case spelling ok:", torch.allclose(a, load(32, alt).predict(X32), atol=1e-5))
try:
    load(32, TRAIN_CH[:-1] + ["XX"]); print("missing channel did NOT raise  <-- BAD")
except ValueError:
    print("missing channel raises: OK")
