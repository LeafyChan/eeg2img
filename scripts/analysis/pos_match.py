import os, torch
def L(p): return torch.load(os.path.expanduser(p), map_location="cpu", weights_only=False)["state_dict"]["model.wrapped_model.bank_positions"].float()
a = L("~/eeg2img_new/checkpoints/auto_632062be_1013_0358.ckpt")   # 32 x 3 (Alljoined)
t = L("~/eeg2img/checkpoints/reve_ct6_s33_1010_1709.ckpt")        # 63 x 3 (THINGS)
m, j = torch.cdist(a, t).min(1)
print("max nearest distance (0 = exact match):", m.max().item())
print("index in the THINGS 63-channel order for each of the 32 rows:", j.tolist())
print("distinct matches:", len(set(j.tolist())), "of 32")
