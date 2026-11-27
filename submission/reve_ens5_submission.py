"""REVE (fine-tuned on THINGS-EEG2) -> DINOv2-giant embedding, Track 1.

Ships: submission.py + weights.pt (made by make_weights.py from best.ckpt).
Input X is (B, C, T) at meta["sfreq"] (120 Hz in the warm-up loader); the
model upsamples it to the 200 Hz REVE was trained at, then predicts (B, 1536).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from benchmark_utils.base_solver import CompetSolver
except ImportError:           # lets make_weights.py import this off-platform
    CompetSolver = object

REVE_SFREQ = 200.0
REVE_N_TIMES = 200            # 1.0 s at 200 Hz, as in training
EMBED_DIM = 512


class ReveClip(nn.Module):
    def __init__(self, n_chans, n_outputs, sfreq):
        super().__init__()
        from braindecode.models import REVE
        kw = dict(n_chans=n_chans, n_times=REVE_N_TIMES, n_outputs=2)
        try:   # pretrained cache is staged on the worker; weights get
            # overwritten by weights.pt anyway, so fall back to random init
            self.reve = REVE.from_pretrained("brain-bzh/reve-base", **kw)
        except Exception:
            self.reve = REVE(**kw)
        # electrode positions used in training (saved in the checkpoint)
        self.register_buffer("bank_positions", torch.zeros(n_chans, 3))
        self.head = nn.Linear(EMBED_DIM, n_outputs)
        self.sfreq = float(sfreq)

    # ---- A/B switches: change ONE per upload -------------------------
    FFT_RESAMPLE = True        # False = original linear interpolation
    ZSCORE = False             # per-window, per-channel standardise + clamp 15
    TTA_SHIFTS = (0,)          # e.g. (-2, 0, 2) averages shifted copies

    def _resample(self, X):
        n_out = int(round(X.shape[-1] * REVE_SFREQ / self.sfreq))
        if n_out == X.shape[-1]:
            return X
        if not self.FFT_RESAMPLE:
            return F.interpolate(X, size=n_out, mode="linear",
                                 align_corners=False)
        spec = torch.fft.rfft(X, dim=-1)
        out = torch.zeros(*X.shape[:-1], n_out // 2 + 1,
                          dtype=spec.dtype, device=X.device)
        m = min(spec.shape[-1], out.shape[-1])
        out[..., :m] = spec[..., :m]
        return torch.fft.irfft(out, n=n_out, dim=-1) * (n_out / X.shape[-1])

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        X = X.float()
        if self.ZSCORE:
            X = (X - X.mean(-1, keepdim=True)) / (X.std(-1, keepdim=True) + 1e-6)
            X = X.clamp(-15, 15)
        outs = []
        for s in self.TTA_SHIFTS:
            Xs = torch.roll(X, shifts=s, dims=-1) if s else X
            Xr = self._resample(Xs)
            pos = self.bank_positions.to(Xr).expand(Xr.shape[0], -1, -1)
            h = self.reve(Xr, pos=pos, return_output=True)[-1]
            h = h.flatten(1, -2).mean(1) if h.ndim > 3 else h.mean(1)
            outs.append(F.normalize(self.head(h), dim=-1))
        return torch.stack(outs).mean(0)


class EnsembleReve(nn.Module):
    """Average of L2-normalised embeddings from several ReveClip models."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        outs = [F.normalize(m.predict(X), dim=-1) for m in self.members]
        return torch.stack(outs).mean(0)


class Solver(CompetSolver):
    name = "REVE-C120-ENS"
    requirements = []
    WEIGHT_FILES = ("weights_s33.pt", "weights_s34.pt", "weights_ct33.pt", "weights_ct34.pt", "weights_ct35.pt")

    def load_model(self, meta):
        members = []
        for fname in self.WEIGHT_FILES:
            m = ReveClip(meta["n_chans"], meta["n_outputs"], meta["sfreq"])
            sd = torch.load(meta["submission_dir"] / fname,
                            map_location="cpu", weights_only=True)
            m.load_state_dict(sd)
            members.append(m.to(meta["device"]).eval())
        return EnsembleReve(members).to(meta["device"]).eval()
