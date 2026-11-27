import os, yaml, neuralbench
D = os.path.join(os.path.dirname(neuralbench.__file__), "tasks/eeg/image/datasets")
out = f"{D}/xu2025alljoined_reve_ho3_pred.yaml"
if os.path.exists(out): raise SystemExit("exists: " + out)
cfg = yaml.safe_load(open(f"{D}/xu2025alljoined_reve_ho3.yaml"))
cfg["save_test_predictions"] = True
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False); print(open(out).read()[-200:])
