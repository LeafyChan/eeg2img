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

# Channel order the 32-channel model is trained with (NeuralBench prepare log).
# ASSUMPTION: verify against a trained checkpoint's bank_positions.
TRAIN_CH = ['Cz', 'FCz', 'AFz', 'Fp1', 'F5', 'F1', 'CP5', 'CP3', 'CP1', 'P1',
            'P3', 'P5', 'P7', 'PO7', 'PO3', 'O1', 'Pz', 'POz', 'Oz', 'O2',
            'PO4', 'PO8', 'P8', 'P6', 'P4', 'P2', 'CP2', 'CP4', 'CP6', 'F2',
            'F6', 'Fp2']


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
        X = self._resample(X)
        pos = self.bank_positions.to(X).expand(X.shape[0], -1, -1)
        h = self.reve(X, pos=pos, return_output=True)[-1]
        h = h.flatten(1, -2).mean(1) if h.ndim > 3 else h.mean(1)
        return self.head(h)


class Solver(CompetSolver):
    name = "REVE-32"
    requirements = []

    def load_model(self, meta):
        n_model = len(TRAIN_CH)
        ch = meta.get("ch_names")
        if ch is not None:
            chan_idx = _chan_index(ch)
        elif meta["n_chans"] == n_model:
            chan_idx = None
            print("WARNING: no ch_names, assuming training order")
        else:
            raise ValueError(f"n_chans={meta['n_chans']}, no ch_names; keys={sorted(meta)}")
        model = ReveClip(n_model, meta["n_outputs"], meta["sfreq"], chan_idx)
        sd = torch.load(meta["submission_dir"] / "weights.pt",
                        map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        return model.to(meta["device"]).eval()
