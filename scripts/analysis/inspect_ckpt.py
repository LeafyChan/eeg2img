import sys, torch
for p in sys.argv[1:]:
    print("==", p)
    try:
        c = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print("LOAD FAILED:", repr(e)[:300]); continue
    print("top keys:", list(c.keys()))
    sd = c.get("state_dict", c)
    ks = list(sd.keys())
    print("n tensors:", len(ks), "total params:", sum(v.numel() for v in sd.values() if hasattr(v, "numel")))
    for k in ks[:4] + ks[-6:]:
        print(" ", k, tuple(sd[k].shape))
    for k in ks:
        if any(s in k for s in ("bank", "temperature", "probe")):
            print(" *", k, tuple(sd[k].shape))
