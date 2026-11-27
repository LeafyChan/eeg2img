import sys, os, yaml, neuralbench
base, new, flags = sys.argv[1], sys.argv[2], set(sys.argv[3:])
D = os.path.join(os.path.dirname(neuralbench.__file__), "tasks/eeg/image/datasets")
cfg = yaml.safe_load(open(f"{D}/{base}.yaml")) or {}
data = cfg.setdefault("data", {})
if "c120" in flags:
    data["neuro"] = {"frequency": 200.0, "filter": [0.1, 55.0], "notch_filter": [50.0, 60.0],
                     "baseline": [0.0, 0.2], "scaler": "RobustScaler", "clamp": 20.0}
if "bs4" in flags:
    data.setdefault("target", {})["batch_size"] = 4
if "ct6" in flags:
    cfg["loss.temperature"] = True
    cfg["loss.symmetric"] = True
    cfg["trainer_config.n_epochs"] = 6
out = f"{D}/{new}.yaml"
if os.path.exists(out): sys.exit(f"already exists: {out}")
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
print(out); print(open(out).read())
