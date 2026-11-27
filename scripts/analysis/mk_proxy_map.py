import hashlib, json, glob, os, numpy as np
home = os.path.expanduser
al = home('~/eeg2img_new/data/Xu2025Alljoined/download/nm000134/stimuli')
th = home('~/eeg2img/data/Gifford2022Large/download')
m = np.load(th + '/image_metadata.npy', allow_pickle=True).item()
cons, files = [str(x) for x in m['train_img_concepts']], [str(x) for x in m['train_img_files']]
hit = glob.glob(th + '/**/' + files[0], recursive=True)
print('example THINGS train image paths:', hit[:2])
root = None
for h in hit:
    if cons[0] in h: root = h[: h.index(cons[0])]; break
print('train image root:', root)
if root is None: raise SystemExit('could not locate THINGS training images: paste the line above')
def md5(p):
    with open(p, 'rb') as f: return hashlib.md5(f.read()).hexdigest()
tmap = {md5(f'{root}{c}/{f}'): i for i, (c, f) in enumerate(zip(cons, files))}
print('THINGS train images hashed:', len(tmap))
out = {}
for i in range(16540):
    h = md5(f'{al}/{i:05d}.jpg')
    if h in tmap: out[i] = tmap[h]
print('alljoined ids mapped:', len(out), 'of 16540')
print('ids where alljoined id == THINGS index:', sum(1 for i, j in out.items() if i == j))
json.dump({'map': out, 'concepts': cons}, open(home('~/proxy_map.json'), 'w'))
print('wrote ~/proxy_map.json')
