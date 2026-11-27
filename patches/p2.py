import numpy as np, pandas as pd, torch
import torch.nn.functional as F

def _ce(loss, scores, tgt):
    out = F.cross_entropy(scores, tgt, reduction=loss.reduction)
    if getattr(loss, "symmetric", False) and scores.shape[0] == scores.shape[1]:
        out = (out + F.cross_entropy(scores.transpose(0, 1), tgt, reduction=loss.reduction)) / 2
    return out

def dedup_ce(loss, pred, true, thr=0.9999):
    """Retrieval loss where other rows with the same target are NOT negatives."""
    tn = F.normalize(true.float(), dim=1)
    dup = tn @ tn.T > thr
    dup.fill_diagonal_(False)
    scores = loss.get_scores(pred, true).masked_fill(dup, float("-inf"))
    return _ce(loss, scores, torch.arange(len(true), device=pred.device))

def p2_loss(loss, pred, true, G, lam=1.0, thr=0.9999, group_on=True):
    ind = dedup_ce(loss, pred, true, thr)
    B = len(true)
    if not group_on or G < 1 or B % G != 0:
        return ind
    if G == 1:
        return ind + lam * ind
    D = pred.shape[1]
    p, t = pred.reshape(B // G, G, D), true.reshape(B // G, G, D)
    assert torch.allclose(t, t[:, :1].expand_as(t), atol=1e-6), "P2: targets differ inside a group (sampler broken)"
    return ind + lam * dedup_ce(loss, p.mean(1), t[:, 0], thr)

class GroupSampler(torch.utils.data.Sampler):
    def __init__(self, keys, G, generator=None):
        codes, _ = pd.factorize(np.asarray(keys))
        order = np.argsort(codes, kind="stable")
        bounds = np.flatnonzero(np.diff(codes[order])) + 1
        self.groups = np.split(order, bounds)
        self.G, self.gen = G, generator
        self._n = sum((len(g) // G) * G for g in self.groups)
    def __len__(self):
        return self._n
    def __iter__(self):
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=self.gen)))
        blocks = [rng.permutation(g)[: (len(g) // self.G) * self.G].reshape(-1, self.G)
                  for g in self.groups if len(g) >= self.G]
        blocks = np.concatenate(blocks)
        return iter(blocks[rng.permutation(len(blocks))].reshape(-1).tolist())

def build_group_sampler(ds, G, gen):
    trig = ds.triggers
    miss = [c for c in ("subject", "filepath") if c not in trig.columns]
    if miss:
        raise RuntimeError(f"NB_GROUP needs columns {miss}; triggers has {list(trig.columns)}")
    assert len(trig) == len(ds), (len(trig), len(ds))
    keys = (trig["subject"].astype(str) + "|" + trig["filepath"].astype(str)).to_numpy()
    s = GroupSampler(keys, G, gen)
    print(f"[P2] GroupSampler G={G}: {len(s)} windows in {len(s.groups)} groups", flush=True)
    return s
