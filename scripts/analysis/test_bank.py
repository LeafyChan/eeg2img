import os, types, torch
os.environ["NB_BANK"] = "1"
import neuralbench.pl_module as m
from neuraltrain.losses.losses import ClipLoss
cls = [c for c in vars(m).values() if isinstance(c, type) and hasattr(c, "_bank_loss")][0]
torch.manual_seed(0)
for sym in (True, False):
    obj = types.SimpleNamespace(loss=ClipLoss(norm_kind="y", temperature=True, symmetric=sym))
    true = torch.randn(8, 1536); pred = torch.randn(8, 1536, requires_grad=True)
    a = cls._bank_loss(obj, pred, true, "train"); b = obj.loss(pred, true)
    print("sym", sym, "first call equals baseline:", torch.allclose(a, b, atol=1e-5))
    true2 = torch.cat([true[:3], torch.randn(5, 1536)])
    c = cls._bank_loss(obj, pred, true2, "train"); c.backward()
    print("  bank size", len(obj._nb_bank), "(expect 13) loss", float(c), "grad finite", bool(torch.isfinite(pred.grad).all()))
    print("  val step unchanged:", torch.allclose(cls._bank_loss(obj, pred, true, "val"), b))
