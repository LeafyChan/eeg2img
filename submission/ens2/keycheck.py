import torch
a=torch.load("weights_a.pt",map_location="cpu",weights_only=True)
b=torch.load("weights_b.pt",map_location="cpu",weights_only=True)
print("same keys:", sorted(a.keys())==sorted(b.keys()))
print("same ch_names:", a["ch_names"]==b["ch_names"])
print("preprocess a:", a["preprocess"]); print("preprocess b:", b["preprocess"])
print("head shapes:", a["head.weight"].shape, b["head.weight"].shape)
