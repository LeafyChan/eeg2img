import os, yaml, neuralbench
D = os.path.join(os.path.dirname(neuralbench.__file__), "tasks/eeg/image/datasets")
out = f"{D}/xu2025alljoined_reve_ho_b256.yaml"
if os.path.exists(out): raise SystemExit("exists: " + out)
cfg = yaml.safe_load(open(f"{D}/xu2025alljoined_reve_ho.yaml"))
cfg["data"]["batch_size"] = 256
cfg["lightning_optimizer_config.optimizer.lr"] = 3e-4
cfg["lightning_optimizer_config.scheduler.kwargs.max_lr"] = 3e-4
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
print(open(out).read())
