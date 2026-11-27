import pandas as pd, glob, os
p = glob.glob(os.path.expanduser('~/eeg2img_new/cache/name=Xu2025Alljoined-b7782d64/cache/data/*.parquet'))[0]
d = pd.read_parquet(p, columns=['type','label','subject'])
H = [f'Xu2025Alljoined/{i}' for i in (9,12,15,18)]
t = d[(d.type=='Image') & (d.label=='stim_test') & d.subject.isin(H)]
print(t.groupby('subject').size())
print('total', len(t))
