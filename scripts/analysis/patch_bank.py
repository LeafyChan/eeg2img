import os, sys, shutil
P = os.path.expanduser("~/miniconda3/envs/eeg2img_new/lib/python3.12/site-packages/neuralbench/pl_module.py")
src = open(P).read()
if "_bank_loss" in src: sys.exit("already patched")
A1 = "            loss = self.loss(loss_pred, loss_true)\n"
A2 = "    def training_step(self, batch: Batch, batch_idx: int):\n"
assert src.count(A1) == 1 and src.count(A2) == 1, "anchors not unique"
METHOD = '''    def _bank_loss(self, pred, true, step_name):
        """Train-only: rank against batch targets + a bank of seen training-image embeddings."""
        import os
        import torch
        import torch.nn.functional as F
        loss = self.loss
        if (step_name != "train" or os.environ.get("NB_BANK") != "1"
                or not hasattr(loss, "get_scores") or true.ndim != 2 or pred.shape != true.shape):
            return loss(pred, true)
        thr = 0.9999
        with torch.no_grad():
            t = true.detach()
            tn = F.normalize(t.float(), dim=1)
            if getattr(self, "_nb_bank", None) is None:
                self._nb_bank, self._nb_bank_n, self._nb_step = t[:0].clone(), tn[:0].clone(), 0
                print("[NB_BANK] bank negatives ON", flush=True)
            keep = ~(torch.tril(tn @ tn.T, -1) > thr).any(1)
            if len(self._nb_bank_n):
                keep &= ~((tn @ self._nb_bank_n.T).max(1).values > thr)
            if keep.any():
                self._nb_bank = torch.cat([self._nb_bank, t[keep]], 0)
                self._nb_bank_n = torch.cat([self._nb_bank_n, tn[keep]], 0)
            dup = (self._nb_bank_n @ tn.T).max(1).values > thr
            self._nb_step += 1
            if self._nb_step % 2000 == 1:
                print(f"[NB_BANK] step {self._nb_step} bank size {len(self._nb_bank)}", flush=True)
        B = len(true)
        cand = torch.cat([true, self._nb_bank.to(true.dtype)], 0)
        scores = loss.get_scores(pred, cand)
        mask = torch.cat([torch.zeros(B, dtype=torch.bool, device=dup.device), dup])
        scores = scores.masked_fill(mask[None, :], float("-inf"))
        tgt = torch.arange(B, device=pred.device)
        out = F.cross_entropy(scores, tgt, reduction=loss.reduction)
        if getattr(loss, "symmetric", False):
            out_c = F.cross_entropy(scores[:, :B].transpose(0, 1), tgt, reduction=loss.reduction)
            out = (out + out_c) / 2
        return out

'''
if not os.path.exists(P + ".bak_bank"): shutil.copy(P, P + ".bak_bank")
src = src.replace(A1, "            loss = self._bank_loss(loss_pred, loss_true, step_name)\n")
src = src.replace(A2, METHOD + A2)
open(P, "w").write(src); print("patched")
