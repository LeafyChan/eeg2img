import glob, os, torch
w = torch.load(os.path.expanduser("~/eeg2img_new/submissions/reve_aj/weights.pt"), map_location="cpu", weights_only=True)
hk = [k for k in w if k.endswith("weight") and tuple(w[k].shape) == (1536, 512)]
print("n keys:", len(w), "head key(s):", hk)
h = w[hk[0]]
for p in sorted(glob.glob(os.path.expanduser("~/eeg2img_new/checkpoints/auto_*.ckpt")), key=os.path.getmtime):
    if os.path.getsize(p) != 280246924: continue
    sd = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
    print(("MATCH  " if torch.equal(sd["model.probe.weight"], h) else "       ") + os.path.basename(p))
