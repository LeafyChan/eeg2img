"""Compare Charvi's v3 single solver (v3/submission.py + v3/weights.pt) with a 1-member ensemble
built from ./submission.py and weights_a.pt. Run from ~/ens2 on a GPU node.
Synthetic data unless --x real.npy (N, 32, 120) is given."""
import argparse, importlib.util, pathlib, sys
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--x", default=None)
a = ap.parse_args()
root = pathlib.Path(".").resolve()


def load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


v3 = load(root / "v3" / "submission.py", "v3sub")
ens = load(root / "submission.py", "enssub")
dev = "cuda" if torch.cuda.is_available() else "cpu"
w = torch.load(root / "weights_a.pt", map_location="cpu", weights_only=True)
ch = list(w["ch_names"])
meta_v3 = dict(submission_dir=root / "v3", device=dev, n_chans=32, n_times=120,
               sfreq=120.0, ch_names=ch, n_outputs=1536)
meta_en = dict(meta_v3, submission_dir=root)

if a.x:
    X = torch.as_tensor(np.load(a.x), dtype=torch.float32)[:256]
else:
    g = torch.Generator().manual_seed(0)
    X = (torch.randn(256, 32, 120, generator=g) * 0.8).clamp(-6.7, 6.7)
    print("SYNTHETIC data")
X = X.to(dev)

single = v3.Solver().load_model(meta_v3)
member = ens.HarnessReve(w, meta_en).to(dev).eval()
one = ens.EnsembleHarness([member]).to(dev).eval()
with torch.inference_mode():
    p_single = single.predict(X)
    p_one = one.predict(X)
print("v3 single vs 1-member ensemble, max|diff|: %.3e" % (p_single - p_one).abs().max().item())
print("shapes", tuple(p_single.shape), tuple(p_one.shape))
