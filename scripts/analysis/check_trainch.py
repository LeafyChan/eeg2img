import os, torch
from braindecode.models import REVE
TRAIN_CH = ['Cz','FCz','AFz','Fp1','F5','F1','CP5','CP3','CP1','P1','P3','P5','P7','PO7','PO3','O1',
            'Pz','POz','Oz','O2','PO4','PO8','P8','P6','P4','P2','CP2','CP4','CP6','F2','F6','Fp2']
ck = torch.load(os.path.expanduser("~/eeg2img_new/checkpoints/auto_632062be_1013_0358.ckpt"),
                map_location="cpu", weights_only=False)["state_dict"]["model.wrapped_model.bank_positions"].float()
m = REVE(n_chans=32, n_times=200, n_outputs=2)
p = m.get_positions(TRAIN_CH).float()
print("positions for TRAIN_CH:", tuple(p.shape), " checkpoint bank:", tuple(ck.shape))
if p.shape != ck.shape:
    raise SystemExit("SHAPE MISMATCH: a channel name was dropped or unknown")
dist, idx = torch.cdist(ck, p).min(1)
print("max nearest distance:", dist.max().item(), " distinct:", len(set(idx.tolist())))
print("identity order:", idx.tolist() == list(range(32)))
print("ckpt row -> TRAIN_CH index:", idx.tolist())
print("suggested TRAIN_CH:", [TRAIN_CH[k] for k in idx.tolist()])
