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
N_MODEL = 32

# Channel order of the 32-channel model (checked against the checkpoint's bank_positions).
TRAIN_CH = ['Cz', 'FCz', 'AFz', 'Fp1', 'F5', 'F1', 'CP5', 'CP3', 'CP1', 'P1',
            'P3', 'P5', 'P7', 'PO7', 'PO3', 'O1', 'Pz', 'POz', 'Oz', 'O2',
            'PO4', 'PO8', 'P8', 'P6', 'P4', 'P2', 'CP2', 'CP4', 'CP6', 'F2',
            'F6', 'Fp2']
# Model channel i is channel THINGS_IDX[i] of the 63-channel THINGS-EEG2 tensor
# (electrode positions matched, max distance 4e-5).
THINGS_IDX = [22, 38, 33, 0, 35, 34, 9, 42, 10, 43, 12, 44, 13, 45, 46, 14, 11, 47, 15, 16,
              48, 49, 18, 50, 17, 51, 21, 53, 20, 60, 59, 30]


def _chan_index(ch_names):
    pos = {str(n).strip().lower(): i for i, n in enumerate(ch_names)}
    missing = [n for n in TRAIN_CH if n.lower() not in pos]
    if missing:
        raise ValueError(f"channels missing from meta['ch_names']: {missing}")
    return [pos[n.lower()] for n in TRAIN_CH]


class ReveClip(nn.Module):
    def __init__(self, n_chans, n_outputs, sfreq, chan_idx=None):
        super().__init__()
        from braindecode.models import REVE
        kw = dict(n_chans=n_chans, n_times=REVE_N_TIMES, n_outputs=2)
        try:
            self.reve = REVE.from_pretrained("brain-bzh/reve-base", **kw)
        except Exception:
            self.reve = REVE(**kw)
        self.register_buffer("bank_positions", torch.zeros(n_chans, 3))
        self.head = nn.Linear(EMBED_DIM, n_outputs)
        self.sfreq = float(sfreq)
        self.chan_idx = chan_idx

    def _resample(self, X):
        if abs(self.sfreq - REVE_SFREQ) < 1e-6:
            return X
        n_out = int(round(X.shape[-1] * REVE_SFREQ / self.sfreq))
        return F.interpolate(X, size=n_out, mode="linear", align_corners=False)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        X = X.float()
        if self.chan_idx is not None:
            X = X[:, torch.as_tensor(self.chan_idx, device=X.device)]
        if X.shape[1] != N_MODEL:
            raise ValueError(f"expected {N_MODEL} channels after selection, got {X.shape[1]}")
        X = self._resample(X)
        pos = self.bank_positions.to(X).expand(X.shape[0], -1, -1)
        h = self.reve(X, pos=pos, return_output=True)[-1]
        h = h.flatten(1, -2).mean(1) if h.ndim > 3 else h.mean(1)
        return self.head(h)


class EnsembleReve(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        outs = [F.normalize(m.predict(X), dim=-1) for m in self.members]
        return torch.stack(outs).mean(0)


class Solver(CompetSolver):
    name = "REVE-32-ENS3"
    requirements = []
    WEIGHT_FILES = ("weights_a.pt", "weights_b.pt", "weights_c.pt")

    def load_model(self, meta):
        n_chans = int(meta["n_chans"])
        ch = meta.get("ch_names")
        if ch is not None and len(ch) == n_chans:
            chan_idx = _chan_index(ch)
        elif n_chans == 63:
            chan_idx = list(THINGS_IDX)
        elif n_chans == N_MODEL:
            print("WARNING: no ch_names, assuming training order")
            chan_idx = None
        else:
            raise ValueError(f"n_chans={n_chans} and no usable ch_names")
        members = []
        for fname in self.WEIGHT_FILES:
            m = ReveClip(N_MODEL, meta["n_outputs"], meta["sfreq"], chan_idx)
            sd = torch.load(meta["submission_dir"] / fname,
                            map_location="cpu", weights_only=True)
            m.load_state_dict(sd)
            members.append(m.to(meta["device"]).eval())
        return EnsembleReve(members).to(meta["device"]).eval()
