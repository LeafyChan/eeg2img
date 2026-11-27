import sys
p = sys.argv[1]; s = open(p).read()
A_old = '''            self._aux_cands = {k: F.normalize(torch.tensor(z[k].astype("float32")), dim=1).to(dev)
                               for k in ("L26", "L27")}'''
A_new = '''            self._aux_cands = {k: F.normalize(torch.tensor(z[k].astype("float32")), dim=1)
                               for k in z.files if k.startswith("L")}'''
B_old = '''        tn = F.normalize(true.detach().float(), dim=1)
        mx, idx = (tn @ self._aux_cands[self._aux_key].T).max(1)
        if self._aux_n == 0:
            st = {k: (round((tn @ v.T).max(1).values.mean().item(), 4),
                      round((tn @ v.T).max(1).values.min().item(), 4)) for k, v in self._aux_cands.items()}
            print("[NB_AUX] match cos (mean, min):", st, flush=True)
            assert mx.min().item() > 0.98, f"NB_AUX: targets do not match {self._aux_key} (min cos {mx.min().item():.3f})"'''
B_new = '''        tn = F.normalize(true.detach().float(), dim=1)
        if self._aux_n == 0:
            tc = tn.cpu(); st = {}
            for k, v in self._aux_cands.items():
                m = (tc @ v.T).max(1).values
                st[k] = (round(m.mean().item(), 4), round(m.min().item(), 4), round((m > 0.98).float().mean().item(), 3))
            print("[NB_AUX] match cos (mean, min, frac>0.98):", st, flush=True)
            best = max(st, key=lambda k: (st[k][2], st[k][0]))
            self._aux_key = os.environ.get("NB_AUX_MATCH") or best
            assert st[self._aux_key][2] >= 0.9, f"NB_AUX: no layer matches: {st}"
            self._aux_cands = {self._aux_key: self._aux_cands[self._aux_key].to(dev)}
            print("[NB_AUX] using", self._aux_key, flush=True)
        mx, idx = (tn @ self._aux_cands[self._aux_key].T).max(1)
        ok = (mx > 0.98).float()'''
C_old = '''        la = (1 - F.cosine_similarity(pred, self._aux_tab[idx], dim=-1)).mean()'''
C_new = '''        cs = F.cosine_similarity(pred, self._aux_tab[idx], dim=-1).mean(1)
        la = ((1 - cs) * ok).sum() / ok.sum().clamp(min=1.0)'''
D_old = '''print(f"[NB_AUX] step {self._aux_n} aux {la.item():.4f} main {loss.item():.4f}", flush=True)'''
D_new = '''print(f"[NB_AUX] step {self._aux_n} aux {la.item():.4f} main {loss.item():.4f} ok {ok.mean().item():.3f}", flush=True)'''
for o, n in ((A_old, A_new), (B_old, B_new), (C_old, C_new), (D_old, D_new)):
    assert s.count(o) == 1, "pattern not found exactly once:\n" + o[:80]
    s = s.replace(o, n)
open(p, "w").write(s); print("patched", p)
