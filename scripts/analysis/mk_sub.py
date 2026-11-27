import os, random, yaml, neuralbench
D = os.path.join(os.path.dirname(neuralbench.__file__), "tasks/eeg/image/datasets")
HELD = [3, 8, 13, 18]
POOL = [s for s in range(1, 21) if s not in HELD]          # 16 training subjects
order = POOL[:]; random.Random(0).shuffle(order)            # fixed shuffle -> nested subsets
sub = lambda ids: [f"Xu2025Alljoined/{i}" for i in ids]
test = sub(HELD)
for n, epochs in ((4, 10), (8, 6), (12, 6)):
    out = f"{D}/xu2025alljoined_reve_f1s{n}.yaml"
    if os.path.exists(out): print("exists:", out); continue
    keep = sub(sorted(order[:n]))
    cfg = yaml.safe_load(open(f"{D}/xu2025alljoined_reve.yaml"))
    cfg["data"]["study"]["filter_stimuli"]["query"] = (
        "type != 'Image' or (label == 'stim_train' and subject in %s) "
        "or (label == 'stim_test' and subject in %s)" % (keep, test))
    cfg["trainer_config.n_epochs"] = epochs
    yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
    print(n, "subjects:", sorted(order[:n]), "epochs:", epochs)
