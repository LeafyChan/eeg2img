import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from benchmark_utils.base_solver import CompetSolver
except ImportError:
    CompetSolver = object

REVE_SFREQ = 200.0
REVE_N_TIMES = 200
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

    def _resample(self, X):
        if abs(self.sfreq - REVE_SFREQ) < 1e-6:
            return X
        n_out = int(round(X.shape[-1] * REVE_SFREQ / self.sfreq))
        return F.interpolate(X, size=n_out, mode="linear", align_corners=False)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        X = self._resample(X.float())
        pos = self.bank_positions.to(X).expand(X.shape[0], -1, -1)
        h = self.reve(X, pos=pos, return_output=True)[-1]
        h = h.flatten(1, -2).mean(1) if h.ndim > 3 else h.mean(1)
        return self.head(h)


class Solver(CompetSolver):
    name = "REVE-THINGS"
    requirements = []

    def load_model(self, meta):
        model = ReveClip(meta["n_chans"], meta["n_outputs"], meta["sfreq"])
        sd = torch.load(meta["submission_dir"] / "weights.pt",
                        map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        return model.to(meta["device"]).eval()