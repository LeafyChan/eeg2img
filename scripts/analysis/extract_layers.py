import os, time, numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
D = os.path.expanduser("~/eeg2img_new/data/Xu2025Alljoined/download/nm000134/stimuli")
LAYERS = [20, 24, 26, 27, 28, 32, 36, 40]
files = sorted(f for f in os.listdir(D) if f.endswith(".jpg") and int(f[:-4]) < 16740)
print("images:", len(files), flush=True)
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-giant",
    size={"shortest_edge": 518}, crop_size={"height": 518, "width": 518})
model = AutoModel.from_pretrained("facebook/dinov2-giant", torch_dtype=torch.float16).cuda().eval()
out = {l: np.zeros((len(files), 1536), np.float16) for l in LAYERS}
B, t0 = 4, time.time()
with torch.no_grad():
    for i in range(0, len(files), B):
        ims = [Image.open(os.path.join(D, f)).convert("RGB") for f in files[i:i+B]]
        x = proc(images=ims, return_tensors="pt")["pixel_values"].cuda().half()
        if i == 0: print("pixel shape", tuple(x.shape), flush=True)
        hs = model(pixel_values=x, output_hidden_states=True).hidden_states
        for l in LAYERS:
            out[l][i:i+B] = hs[l].mean(1).float().cpu().numpy()   # all tokens incl. CLS, like the target
        if (i // B) % 200 == 0: print(i, round(time.time() - t0), "s", flush=True)
np.savez("dino518_allmean.npz", files=np.array(files), **{f"L{l}": out[l] for l in LAYERS})
print("done", flush=True)
