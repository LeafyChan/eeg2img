import os, yaml, neuralbench
D = os.path.join(os.path.dirname(neuralbench.__file__), "tasks/eeg/image/datasets")
out = f"{D}/xu2025alljoined_reve_ho_eval.yaml"
if os.path.exists(out): raise SystemExit("exists: " + out)
cfg = yaml.safe_load(open(f"{D}/xu2025alljoined_reve_ho.yaml"))
cfg["eval_only"] = True
cfg["save_test_predictions"] = True
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False); print(open(out).read()[-300:])
