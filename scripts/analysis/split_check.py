import glob, os, time, pandas as pd
t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.1f}s] {m}", flush=True)
root = os.path.expanduser("~/eeg2img/cache")
fs = []
for d in glob.glob(root + "/name=Gifford2022Large*"):
    for dp, _, names in os.walk(d):
        fs += [os.path.join(dp, n) for n in names if n.endswith(".parquet") and "timeline_index" in dp]
fs.sort(key=os.path.getsize)
df = pd.read_parquet(fs[0])
tl = sorted(df.timeline.unique())
log("importing PredefinedSplit")
from neuralbench.transforms import PredefinedSplit
log("imported")
pick = [t for t in tl if "split=test" in t][-1]
def run(q, ratio):
    t = PredefinedSplit(test_split_query=q, col_name="split", valid_split_by="timeline",
                        valid_split_ratio=ratio, valid_random_state=33)
    out = t._run(df)
    print("query:", q, "ratio:", ratio)
    print(out.groupby("timeline").split.first().to_string(), flush=True)
run(None, 0.2)
run(f"timeline == '{pick}'", 0.2)
log("done")
