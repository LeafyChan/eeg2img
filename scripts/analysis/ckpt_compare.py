import os, torch
from collections import defaultdict
def load(p): return torch.load(os.path.expanduser(p), map_location="cpu", weights_only=False)["state_dict"]
a1 = load("~/eeg2img_new/checkpoints/auto_632062be_1013_0218.ckpt")   # first copy, end of epoch 1
a6 = load("~/eeg2img_new/checkpoints/auto_632062be_1013_0358.ckpt")   # final best
t  = load("~/eeg2img/checkpoints/reve_ct6_s33_1010_1709.ckpt")        # THINGS, 0.3.1
def grp(k):
    for g in ("final_layer", "probe", "bank_positions", "temperature"):
        if g in k: return g
    return "encoder"
print("== seed-34 run, copy 0218 vs 0358: did each group of weights change?")
st = defaultdict(lambda: [0, 0, 0.0])
for k in a6:
    d = (a6[k].float() - a1[k].float()).abs().max().item()
    s = st[grp(k)]; s[0] += 1; s[1] += int(d > 0); s[2] = max(s[2], d)
for g, (n, c, m) in st.items(): print(f"{g:15s} tensors={n:4d} changed={c:4d} max_abs_diff={m:.3e}")
print("== key sets: Alljoined 0.4.0 vs THINGS 0.3.1")
print("only in alljoined:", sorted(set(a6) - set(t)))
print("only in things:   ", sorted(set(t) - set(a6)))
for k in a6:
    if k in t and a6[k].shape != t[k].shape: print("shape differs:", k, tuple(a6[k].shape), tuple(t[k].shape))
