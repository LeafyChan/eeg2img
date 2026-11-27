import os, shutil, sys
S = os.path.expanduser("~/miniconda3/envs/eeg2img_new/lib/python3.12/site-packages/neuralbench")
HEAD_OLD = """        self._build_probe(
            probe_config, n_inputs, wrapper_n_outputs, brain_model_output_size
        )
"""
HEAD_NEW = HEAD_OLD + """        import os as _os
        if _os.environ.get("NB_AUX") == "1":
            _k = len(_os.environ.get("NB_AUX_LAYERS", "L20,L24,L28,L32,L36").split(","))
            self.aux_head = nn.Linear(n_inputs, 1536 * _k)
            print(f"[NB_AUX] aux head {n_inputs} -> {_k} x 1536", flush=True)
"""
EMB_OLD = """        out = self.aggregation(out)
        if return_embedding:
            return out
"""
EMB_NEW = """        out = self.aggregation(out)
        self._last_emb = out
        if return_embedding:
            return out
"""
CALL_OLD = "            loss = self._p2_or_bank(loss_pred, loss_true, step_name)\n"
CALL_NEW = CALL_OLD + "            loss = self._aux_add(loss, loss_true, step_name)\n"
DEF_OLD = "    def training_step(self, batch: Batch, batch_idx: int):\n"
AUX = '''    def _aux_add(self, loss, true, step_name):
        import os
        if (os.environ.get("NB_AUX") != "1" or step_name != "train"
                or not hasattr(self.model, "aux_head") or isinstance(loss, dict)):
            return loss
        import numpy as np
        import torch
        import torch.nn.functional as F
        dev = true.device
        if getattr(self, "_aux_tab", None) is None:
            z = np.load(os.environ["NB_AUX_NPZ"])
            names = os.environ.get("NB_AUX_LAYERS", "L20,L24,L28,L32,L36").split(",")
            self._aux_cands = {k: F.normalize(torch.tensor(z[k].astype("float32")), dim=1).to(dev)
                               for k in ("L26", "L27")}
            tabs = []
            for k in names:
                a = torch.tensor(z[k].astype("float32"))
                tabs.append((a - a.mean(0)) / (a.std(0) + 1e-6))
            self._aux_tab = torch.stack(tabs, 1).to(dev)
            self._aux_key = os.environ.get("NB_AUX_MATCH", "L26")
            self._aux_n = 0
        tn = F.normalize(true.detach().float(), dim=1)
        mx, idx = (tn @ self._aux_cands[self._aux_key].T).max(1)
        if self._aux_n == 0:
            st = {k: (round((tn @ v.T).max(1).values.mean().item(), 4),
                      round((tn @ v.T).max(1).values.min().item(), 4)) for k, v in self._aux_cands.items()}
            print("[NB_AUX] match cos (mean, min):", st, flush=True)
            assert mx.min().item() > 0.98, f"NB_AUX: targets do not match {self._aux_key} (min cos {mx.min().item():.3f})"
        emb = self.model._last_emb
        K = self._aux_tab.shape[1]
        pred = self.model.aux_head(emb).view(len(emb), K, -1)
        la = (1 - F.cosine_similarity(pred, self._aux_tab[idx], dim=-1)).mean()
        self._aux_n += 1
        if self._aux_n % 500 == 1:
            print(f"[NB_AUX] step {self._aux_n} aux {la.item():.4f} main {loss.item():.4f}", flush=True)
        return loss + float(os.environ.get("NB_AUX_LAMBDA", "1.0")) * la

'''
new = {}
for fn, pairs in (("modules.py", [(HEAD_OLD, HEAD_NEW), (EMB_OLD, EMB_NEW)]),
                  ("pl_module.py", [(CALL_OLD, CALL_NEW), (DEF_OLD, AUX + DEF_OLD)])):
    s = open(f"{S}/{fn}").read()
    if "NB_AUX" in s: sys.exit(f"{fn} already patched")
    for o, n in pairs:
        assert s.count(o) == 1, (fn, s.count(o), o[:50])
        s = s.replace(o, n)
    new[fn] = s
for fn, s in new.items():
    shutil.copy(f"{S}/{fn}", f"{S}/{fn}.bak_aux"); open(f"{S}/{fn}", "w").write(s); print("patched", fn)
