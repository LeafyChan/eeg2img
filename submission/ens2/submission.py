"""Track 1 (EEG -> image) Solver: 2-member ensemble of fine-tuned REVE + linear head (HarnessReve).

Members: weights_a.pt, weights_b.pt (same format as the single-model weights.pt made by
tools/export_submission.py: reve, head.weight, head.bias, bank_positions, ch_names, preprocess).
reve_positions.json ships alongside, used only if the worker has no cached REVE position bank.

Each member is the unchanged HarnessReve from the single-model solver (v3: name matching with
casefold + up to 4 zero-filled channels; with all 32 present the computation is unchanged).
Ensemble output = plain mean of the members' raw outputs (no per-member L2 normalisation), so a
1-member ensemble reproduces the single-model submission exactly. The scorer averages raw
predictions per (subject, image) and normalises afterwards.
Stateless: no state is carried across predict() calls; every row is computed independently.
"""

import os
import warnings

import torch
import torch.nn.functional as F

try:
    from benchmark_utils.base_solver import CompetSolver
except ImportError:  # local testing outside benchopt
    CompetSolver = object

# At most this many of the 32 training channels may be missing (12.5%); each is fed as a constant 0.
MAX_MISSING_CHANNELS = 4


def _normalise_name(name):
    """Strip decorations the way the organisers' REVE solver does (reve_probe.py:65):
    "EEG Cz-REF" -> "Cz". Matching is exact (case-sensitive) after this."""
    return str(name).removeprefix("EEG ").split("-", 1)[0]


def _match_channels(meta_names, train_names):
    """Index into the input channels for each training channel, by normalised name.

    Exact match first; a training channel without an exact match is then matched
    case-insensitively (str.casefold, e.g. raw Emotiv "Afz" -> "AFz") and reported, not counted as
    missing. Two inputs that collide after normalisation or after casefold raise ValueError.

    Returns (index, missing, casefold_pairs): index[i] is the input column of training channel i,
    or -1 when it is missing; at most MAX_MISSING_CHANNELS may be missing, otherwise ValueError;
    casefold_pairs lists (input name, training name) matched only case-insensitively."""
    if not meta_names:
        raise ValueError(
            "meta['ch_names'] is empty: this model selects its 32 training channels by name"
        )
    raw = [str(n) for n in meta_names]
    names = [_normalise_name(n) for n in raw]
    index, folded = {}, {}
    for i, n in enumerate(names):
        if n in index:
            raise ValueError(
                f"input channels {raw[index[n]]!r} and {raw[i]!r} both normalise to {n!r}; "
                "cannot select channels unambiguously"
            )
        index[n] = i
        f = n.casefold()
        if f in folded:
            raise ValueError(
                f"input channels {raw[folded[f]]!r} and {raw[i]!r} are the same name ignoring case "
                f"({f!r}); cannot select channels unambiguously"
            )
        folded[f] = i
    idx, missing, casefold_pairs = [], [], []
    for c in train_names:
        if c in index:
            idx.append(index[c])
        elif c.casefold() in folded:
            j = folded[c.casefold()]
            idx.append(j)
            casefold_pairs.append((raw[j], c))
        else:
            idx.append(-1)
            missing.append(c)
    if len(missing) > MAX_MISSING_CHANNELS:
        raise ValueError(
            f"input lacks {len(missing)} training channel(s) {missing} "
            f"(at most {MAX_MISSING_CHANNELS} can be zero-filled); "
            f"input has {raw} (normalised {names})"
        )
    return idx, missing, casefold_pairs


class HarnessReve(torch.nn.Module):
    def __init__(self, weights, meta):
        super().__init__()
        from braindecode.models import REVE

        pre = weights["preprocess"]
        self.train_names = list(weights["ch_names"])
        self.n_times = int(pre["n_times"])
        self.clamp = float(pre["clamp"])
        self.eps = float(pre["eps"])

        kw = dict(n_chans=len(self.train_names), n_times=self.n_times, n_outputs=2)
        try:
            self.reve = REVE(**kw)
        except RuntimeError:
            # REVE() loads its position bank (cached by the worker image). If absent and
            # offline, fall back to the copy shipped in the submission folder.
            os.environ["REVE_POSITIONS_PATH"] = str(meta["submission_dir"])
            self.reve = REVE(**kw)
        self.reve.load_state_dict(weights["reve"], strict=True)
        self.register_buffer("bank_positions", weights["bank_positions"].float())
        self.head = torch.nn.Linear(weights["head.weight"].shape[1], weights["head.weight"].shape[0])
        self.head.load_state_dict({"weight": weights["head.weight"], "bias": weights["head.bias"]})

        idx, missing, casefold_pairs = _match_channels(meta.get("ch_names"), self.train_names)
        self.missing_channels = missing
        self.casefold_pairs = casefold_pairs
        if casefold_pairs:
            msg = (f"[REVE-SHO-H] WARNING: {len(casefold_pairs)} channel(s) matched ignoring case "
                   f"(input -> training): {casefold_pairs}")
            print(msg, flush=True)
            warnings.warn(msg)
        if missing:
            msg = (f"[REVE-SHO-H] WARNING: {len(missing)} training channel(s) missing from the input, "
                   f"fed as constant 0: {missing}")
            print(msg, flush=True)
            warnings.warn(msg)
            present = [i for i, j in enumerate(idx) if j >= 0]
            self.register_buffer("present_pos", torch.tensor(present, dtype=torch.long))
            idx = [j for j in idx if j >= 0]
        self.register_buffer("channel_index", torch.tensor(idx, dtype=torch.long))

        sfreq, n_times = float(meta["sfreq"]), int(meta["n_times"])
        if abs(n_times / sfreq - 1.0) > 0.05:
            msg = (f"[REVE-SHO-H] WARNING: input window is {n_times / sfreq:.3f} s "
                   f"({n_times} samples @ {sfreq} Hz); trained on 1.0 s windows")
            print(msg, flush=True)
            warnings.warn(msg)

    def preprocess(self, X):
        X = X.to(torch.float32)[:, self.channel_index]
        if self.missing_channels:
            # A zero row stays exactly 0 through resampling, z-score (0 / eps) and clamp.
            full = X.new_zeros(X.shape[0], len(self.train_names), X.shape[-1])
            full[:, self.present_pos] = X
            X = full
        if X.shape[-1] != self.n_times:
            X = F.interpolate(X, size=self.n_times, mode="linear", align_corners=False)
        mean = X.mean(-1, keepdim=True)
        std = X.std(-1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return ((X - mean) / std).clamp(-self.clamp, self.clamp)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        X = self.preprocess(X)
        pos = self.bank_positions.to(X).expand(X.shape[0], -1, -1)
        tokens = self.reve(X, pos=pos, return_output=True)[-1]  # (B, 32 tokens, 512)
        return self.head(tokens.mean(dim=1))                    # (B, 1536)


class EnsembleHarness(torch.nn.Module):
    """Plain mean of the members' raw outputs."""

    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        return torch.stack([m.predict(X) for m in self.members]).mean(0)


class Solver(CompetSolver):
    name = "REVE-SHO-ENS2"
    requirements = []
    WEIGHT_FILES = ("weights_a.pt", "weights_b.pt")

    def load_model(self, meta):
        members = []
        for fname in self.WEIGHT_FILES:
            weights = torch.load(
                meta["submission_dir"] / fname, map_location="cpu", weights_only=True
            )
            n_out = weights["head.weight"].shape[0]
            if "n_outputs" in meta and int(meta["n_outputs"]) != n_out:
                raise ValueError(f"{fname}: meta n_outputs {meta['n_outputs']} != head width {n_out}")
            members.append(HarnessReve(weights, meta))
        names = [tuple(m.train_names) for m in members]
        if len(set(names)) != 1:
            raise ValueError(f"members were trained on different channel orders: {names}")
        print(f"[ENS2] {len(members)} members: {self.WEIGHT_FILES}", flush=True)
        return EnsembleHarness(members).to(meta["device"]).eval()
