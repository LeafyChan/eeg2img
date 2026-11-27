import json, glob, os, collections
import numpy as np
root = glob.glob(os.path.expanduser(
    "~/eeg2img_new/cache/neuralset.extractors.image.HuggingFaceImage._get_data,v6/imsize=518*"))[0]
emb = {}
pref = collections.Counter()
for info in sorted(glob.glob(f"{root}/*-info.jsonl")):
    for line in open(info):
        r = json.loads(line)
        k = r["#key"]
        pref[k.split("/")[0]] += 1
        mm = np.memmap(f"{root}/{r['filename']}", dtype=r["dtype"], mode="r",
                       offset=r["offset"], shape=tuple(r["shape"]))
        emb[k] = np.array(mm, dtype=np.float32)
print("keys per study:", dict(pref))
aj = {k: v for k, v in emb.items() if k.startswith("Xu2025Alljoined")}
print("Alljoined keys:", len(aj))
print("example keys:", list(aj)[:3])
X = np.stack(list(aj.values()))
print("shape", X.shape, "finite", np.isfinite(X).all(),
      "unique rows", len(np.unique(X, axis=0)), "mean norm", np.linalg.norm(X, axis=1).mean())
np.savez(os.path.expanduser("~/aj_emb.npz"), keys=np.array(list(aj)), X=X)
print("saved ~/aj_emb.npz")
