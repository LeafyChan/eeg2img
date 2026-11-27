import os, yaml, neuralbench
D = os.path.join(os.path.dirname(neuralbench.__file__), "tasks/eeg/image/datasets")
out = f"{D}/xu2025alljoined_reve_ho2.yaml"
if os.path.exists(out): raise SystemExit("exists: " + out)
H = [f"Xu2025Alljoined/{i}" for i in (9, 12, 15, 18)]
cfg = yaml.safe_load(open(f"{D}/xu2025alljoined_reve.yaml"))
cfg["data"]["study"]["filter_stimuli"]["query"] = (
    "type != 'Image' or (label == 'stim_train' and subject not in %s) "
    "or (label == 'stim_test' and subject in %s)" % (H, H))
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False); print(open(out).read())
