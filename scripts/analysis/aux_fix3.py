import sys
p = sys.argv[1]; s = open(p).read()
R = [
('''            tc = tn.cpu(); st = {}''',
 '''            MG = float(os.environ.get("NB_AUX_MARGIN", "0.02")); tc = tn.cpu(); st = {}'''),
('''                m = (tc @ v.T).max(1).values
                st[k] = (round(m.mean().item(), 4), round(m.min().item(), 4), round((m > 0.98).float().mean().item(), 3))''',
 '''                t2 = (tc @ v.T).topk(2, dim=1).values
                m = t2[:, 0]; g = t2[:, 0] - t2[:, 1]
                st[k] = (round(m.mean().item(), 4), round(m.min().item(), 4), round((g > MG).float().mean().item(), 3), round(g.median().item(), 4))'''),
('''print("[NB_AUX] match cos (mean, min, frac>0.98):", st, flush=True)''',
 '''print("[NB_AUX] match (mean cos, min cos, frac margin>MG, median margin):", st, flush=True)'''),
('''        mx, idx = (tn @ self._aux_cands[self._aux_key].T).max(1)
        ok = (mx > 0.98).float()''',
 '''        t2v, t2i = (tn @ self._aux_cands[self._aux_key].T).topk(2, dim=1)
        idx = t2i[:, 0]
        ok = ((t2v[:, 0] - t2v[:, 1]) > float(os.environ.get("NB_AUX_MARGIN", "0.02"))).float()'''),
]
for o, n in R:
    assert s.count(o) == 1, "pattern not found exactly once:\n" + o[:90]
    s = s.replace(o, n)
open(p, "w").write(s); print("patched", p)
