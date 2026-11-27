import os, glob, yaml, pandas as pd, neuralbench
D = os.path.join(os.path.dirname(neuralbench.__file__), "tasks/eeg/image/datasets")
H = [f"Xu2025Alljoined/{i}" for i in (2, 6, 11, 17)]
HI = [f"{i:05d}.jpg" for i in range(16540) if (i // 10) % 10 == 0]
Q = ("type != 'Image' or (label == 'stim_train' and ((subject not in %s and stim_file not in %s) "
     "or (subject in %s and stim_file in %s)))" % (H, HI, H, HI))
TQ = "label == 'stim_train' and subject in %s" % (H,)
p = glob.glob(os.path.expanduser("~/eeg2img_new/cache/name=Xu2025Alljoined-b7782d64/cache/data/*.parquet"))[0]
d = pd.read_parquet(p, columns=["type", "label", "subject", "stim_file"])
im = d.query(Q); im = im[im.type == "Image"]; te = im.query(TQ)
print("held-out images:", len(HI), "| kept image rows:", len(im), "| test rows:", len(te), "| train+val rows:", len(im) - len(te))
print("test subjects:", sorted(te.subject.unique()), "| test images:", te.stim_file.nunique(),
      "| windows per (subject,image):", round(te.groupby(["subject", "stim_file"]).size().mean(), 2))
tr = im.drop(te.index)
print("train subjects:", tr.subject.nunique(), "| held-out images in train rows:", int(tr.stim_file.isin(HI).sum()),
      "| held-out subjects in train rows:", int(tr.subject.isin(H).sum()))
out = f"{D}/xu2025alljoined_reve_hi.yaml"
if os.path.exists(out): raise SystemExit("exists: " + out)
cfg = yaml.safe_load(open(f"{D}/xu2025alljoined_reve.yaml"))
cfg["data"]["study"]["filter_stimuli"]["query"] = Q
cfg["data"]["study"]["split"]["test_split_query"] = TQ
cfg["save_test_predictions"] = True
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False); print("wrote", out)
