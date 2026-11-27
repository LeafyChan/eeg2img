import os, yaml, neuralbench
D = os.path.join(os.path.dirname(neuralbench.__file__), "tasks/eeg/image/datasets")
H = [f"Xu2025Alljoined/{i}" for i in (3, 8, 13, 18)]
Q = ("type != 'Image' or (label == 'stim_train' and subject not in %s) "
     "or (label == 'stim_test' and subject in %s)" % (H, H))
out = f"{D}/xu2025alljoined_reve_ho.yaml"
if os.path.exists(out):
    print("already exists:", out)
else:
    cfg = yaml.safe_load(open(f"{D}/xu2025alljoined_reve.yaml"))
    cfg["data"]["study"]["filter_stimuli"]["query"] = Q
    yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
    print("created:", out)
print(open(out).read())
