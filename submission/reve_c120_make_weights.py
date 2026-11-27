import sys, torch
from submission import ReveClip

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = ck["state_dict"]
print("example ckpt keys:", list(sd)[:4], "...", len(sd), "total")

N_OUT, N_CHANS = 1536, 63
model = ReveClip(N_CHANS, N_OUT, sfreq=120.0)
new = {}
nonreve = {k: v for k, v in sd.items() if "wrapped_model" not in k}
for k, v in model.state_dict().items():
    if k.startswith("reve."):
        key = "model.wrapped_model.model." + k[5:]
        c = [key] if key in sd else []
    elif k == "bank_positions":
        c = [x for x in sd if x.endswith("wrapped_model.bank_positions")]
    else:  # head: only among keys outside the REVE encoder
        want = (N_OUT, 512) if k.endswith("weight") else (N_OUT,)
        c = [x for x, t in nonreve.items() if tuple(t.shape) == want]
    if not c and k.startswith("reve.final_layer"):
        print("unused (not in ckpt), keeping init:", k)
        new[k] = v.clone(); continue
    assert len(c) == 1, f"{k}: {len(c)} candidates {c[:5]}"
    assert sd[c[0]].shape == v.shape, f"{k}: {sd[c[0]].shape} vs {v.shape}"
    new[k] = sd[c[0]].clone()
model.load_state_dict(new, strict=True)
torch.save(new, sys.argv[2])

model.eval()
out = model.predict(torch.randn(4, N_CHANS, 120))
print("OK, output", tuple(out.shape), "finite:", bool(torch.isfinite(out).all()))