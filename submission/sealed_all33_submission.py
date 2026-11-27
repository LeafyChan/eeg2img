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
MIN_FALLBACK_CH = 16

TRAIN_CH = ['Cz', 'FCz', 'AFz', 'Fp1', 'F5', 'F1', 'CP5', 'CP3', 'CP1', 'P1',
            'P3', 'P5', 'P7', 'PO7', 'PO3', 'O1', 'Pz', 'POz', 'Oz', 'O2',
            'PO4', 'PO8', 'P8', 'P6', 'P4', 'P2', 'CP2', 'CP4', 'CP6', 'F2',
            'F6', 'Fp2']
THINGS_IDX = [22, 38, 33, 0, 35, 34, 9, 42, 10, 43, 12, 44, 13, 45, 46, 14, 11, 47, 15, 16,
              48, 49, 18, 50, 17, 51, 21, 53, 20, 60, 59, 30]


def _canon(name):
    n = str(name).strip().upper()
    if n.endswith("Z"):
        n = n[:-1] + "z"
    if n.startswith("FP"):
        n = "Fp" + n[2:]
    return n


def _chan_index(ch_names):
    pos = {str(n).strip().lower(): i for i, n in enumerate(ch_names)}
    missing = [n for n in TRAIN_CH if n.lower() not in pos]
    if missing:
        raise ValueError(f"channels missing from meta['ch_names']: {missing}")
    return [pos[n.lower()] for n in TRAIN_CH]


def _positions(names):
    from braindecode.models import REVE
    return REVE.get_positions(names)


def _position_plan(ch_names, getter):
    idx, rows, dropped = [], [], []
    for i, n in enumerate(ch_names):
        c = _canon(n)
        try:
            p = torch.as_tensor(getter([c]), dtype=torch.float32)
            ok = tuple(p.shape) == (1, 3) and bool(torch.isfinite(p).all())
        except Exception:
            ok = False
        if ok:
            idx.append(i)
            rows.append(p[0])
        else:
            dropped.append(str(n))
    if len(idx) < MIN_FALLBACK_CH:
        raise ValueError(f"only {len(idx)} channels have known positions; dropped: {dropped}")
    print(f"WARNING: position-based channels, using {len(idx)}, dropped: {dropped}")
    return idx, torch.stack(rows)


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
        self.n_expected = N_MODEL

    def use_custom_channels(self, chan_idx, positions):
        self.chan_idx = list(chan_idx)
        self.n_expected = len(self.chan_idx)
        self.bank_positions = positions.to(torch.float32)

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
        if X.shape[1] != self.n_expected:
            raise ValueError(f"expected {self.n_expected} channels after selection, got {X.shape[1]}")
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
    name = "REVE-32-ALL33"
    requirements = []
    WEIGHT_FILES = ("weights_a.pt",)

    def load_model(self, meta):
        n_chans = int(meta["n_chans"])
        ch = meta.get("ch_names")
        chan_idx, fallback = None, False
        if ch is not None and len(ch) == n_chans:
            try:
                chan_idx = _chan_index(ch)
            except ValueError as err:
                print("WARNING:", err, "-> falling back to position-based channels")
                fallback = True
        elif n_chans == 63:
            chan_idx = list(THINGS_IDX)
        elif n_chans == N_MODEL:
            print("WARNING: no ch_names, assuming training order")
        else:
            raise ValueError(f"n_chans={n_chans} and no usable ch_names")
        members = []
        for fname in self.WEIGHT_FILES:
            m = ReveClip(N_MODEL, meta["n_outputs"], meta["sfreq"], chan_idx)
            sd = torch.load(meta["submission_dir"] / fname, map_location="cpu", weights_only=True)
            m.load_state_dict(sd)
            members.append(m)
        if fallback:
            idx, pos = _position_plan(ch, members[0].reve.get_positions)
            for m in members:
                m.use_custom_channels(idx, pos)
        members = [m.to(meta["device"]).eval() for m in members]
        return EnsembleReve(members).to(meta["device"]).eval()
