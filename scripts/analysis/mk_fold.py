import os, sys, yaml, neuralbench
name, subs = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
D = os.path.join(os.path.dirname(neuralbench.__file__), "tasks/eeg/image/datasets")
out = f"{D}/{name}.yaml"
if os.path.exists(out): raise SystemExit("exists: " + out)
H = [f"Xu2025Alljoined/{i}" for i in subs]
cfg = yaml.safe_load(open(f"{D}/xu2025alljoined_reve.yaml"))
cfg["data"]["study"]["filter_stimuli"]["query"] = (
    "type != 'Image' or (label == 'stim_train' and subject not in %s) "
    "or (label == 'stim_test' and subject in %s)" % (H, H))
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False); print(open(out).read())
