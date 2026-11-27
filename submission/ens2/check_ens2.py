"""Checks for the 2-member ensemble before upload.

Usage (from anywhere, in an env with torch + braindecode):
    python check_ens2.py <folder> [--x real.npy] [--batches 50]

<folder> holds submission.py, weights_a.pt, weights_b.pt, reve_positions.json.
--x: real harness input windows, array (N, 32, 120) float, channels in the training order
     (weights_a["ch_names"]). Without it, synthetic robust-scaled-like data is used
     (fine for the equivalence/statelessness maths, NOT a real-data smoke test).

Reports:
  1. ensemble == manual mean of the two single models (max |diff|)
  2. a 1-member ensemble == the single model (max |diff|)
  3. batch independence (shuffled rows, other batch sizes) and statelessness
  4. runtime per batch of 64: one member vs the ensemble (GPU if available)
"""
import argparse
import importlib.util
import pathlib
import sys
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--x", default=None)
ap.add_argument("--batches", type=int, default=50)
a = ap.parse_args()
folder = pathlib.Path(a.folder).resolve()

spec = importlib.util.spec_from_file_location("submission", folder / "submission.py")
sub = importlib.util.module_from_spec(spec)
sys.modules["submission"] = sub
spec.loader.exec_module(sub)

dev = "cuda" if torch.cuda.is_available() else "cpu"
wa = torch.load(folder / "weights_a.pt", map_location="cpu", weights_only=True)
wb = torch.load(folder / "weights_b.pt", map_location="cpu", weights_only=True)
ch = list(wa["ch_names"])
meta = dict(submission_dir=folder, device=dev, n_chans=32, n_times=120, sfreq=120.0,
            ch_names=ch, n_outputs=1536)
print("device:", dev, "| channels:", len(ch))

ens = sub.Solver().load_model(meta)
ma = sub.HarnessReve(wa, meta).to(dev).eval()
mb = sub.HarnessReve(wb, meta).to(dev).eval()

if a.x:
    X = torch.as_tensor(np.load(a.x), dtype=torch.float32)
    print("real data:", tuple(X.shape), "std %.3f max|x| %.2f nan %s" %
          (X.std().item(), X.abs().max().item(), bool(torch.isnan(X).any())))
else:
    g = torch.Generator().manual_seed(0)
    X = (torch.randn(256, 32, 120, generator=g) * 0.8).clamp(-6.7, 6.7)
    print("SYNTHETIC data (no --x given):", tuple(X.shape))
X = X[:256].to(dev)
assert X.shape[1:] == (32, 120), X.shape

with torch.inference_mode():
    e = ens.predict(X)
    pa, pb = ma.predict(X), mb.predict(X)
    man = (pa + pb) / 2
    print("1. ensemble vs manual mean, max|diff|: %.3e" % (e - man).abs().max().item())
    one = sub.EnsembleHarness([ma]).predict(X)
    print("2. 1-member ensemble vs single, max|diff|: %.3e" % (one - pa).abs().max().item())
    print("   members differ (should be > 0): %.3e" % (pa - pb).abs().max().item())
    perm = torch.randperm(X.shape[0], device=dev)
    d_perm = (ens.predict(X[perm]) - e[perm]).abs().max().item()
    d_bs = max((torch.cat([ens.predict(X[i:i + bs]) for i in range(0, X.shape[0], bs)]) - e).abs().max().item()
               for bs in (1, 7, 64))
    d_again = (ens.predict(X) - e).abs().max().item()
    print("3. shuffled rows %.3e | batch sizes 1/7/64 %.3e | repeat call %.3e" % (d_perm, d_bs, d_again))
    print("   output shape", tuple(e.shape), "finite", bool(torch.isfinite(e).all()))
    cos = torch.nn.functional.cosine_similarity(pa, pb).mean().item()
    print("   mean cosine between members: %.3f" % cos)


def bench(model, n):
    xb = X[:64]
    with torch.inference_mode():
        for _ in range(3):
            model.predict(xb)
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            model.predict(xb)
        if dev == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


t1 = bench(ma, a.batches)
t2 = bench(ens, a.batches)
print("4. ms per batch of 64: one member %.1f | ensemble %.1f | ratio %.2f" % (t1 * 1e3, t2 * 1e3, t2 / t1))
