# Appended to a COPY of submission.py by 10_harness_replay.sh.
# Logs what the real harness hands the model (first 3 batches + total window
# count), so it can be compared with what our solver assumes. Does not change
# predictions. TAP_ABORT=1 stops the process after the 3rd batch (no score).
import atexit as _atexit
import json as _json
import os as _os

_TAP_LOG = _os.environ.get("TAP_LOG", "/tmp/tap.jsonl")
_TAP_N = {"windows": 0}
_orig_load_model = Solver.load_model


def _tap_write(obj):
    with open(_TAP_LOG, "a") as f:
        f.write(_json.dumps(obj) + "\n")


def _tapped_load_model(self, meta):
    m = _orig_load_model(self, meta)
    _tap_write({
        "event": "meta",
        "sfreq": float(meta.get("sfreq", -1)),
        "n_chans": int(meta.get("n_chans", -1)),
        "n_times": int(meta.get("n_times", -1)),
        "n_outputs": int(meta.get("n_outputs", -1)),
        "ch_names": list(meta.get("ch_names", [])),
    })
    _predict = m.predict
    seen = {"n": 0}

    def predict(X, *a, **k):
        if seen["n"] < 3:
            x = X.detach().float()
            _tap_write({
                "event": "batch", "shape": list(x.shape),
                "mean": float(x.mean()), "std": float(x.std()),
                "min": float(x.min()), "max": float(x.max()),
                "abs_max_per_chan_mean": float(x.abs().amax(-1).mean()),
                "nan": int(x.isnan().sum()),
            })
        seen["n"] += 1
        _TAP_N["windows"] += int(X.shape[0])
        out = _predict(X, *a, **k)
        if seen["n"] == 3 and _os.environ.get("TAP_ABORT") == "1":
            _tap_write({"event": "abort_after_3_batches",
                        "windows_seen": _TAP_N["windows"]})
            _os._exit(0)
        return out

    m.predict = predict
    return m


def _tap_total():
    _tap_write({"event": "total_windows", "n": _TAP_N["windows"]})


_atexit.register(_tap_total)
Solver.load_model = _tapped_load_model
