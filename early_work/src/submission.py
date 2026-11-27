import torch
import torch.nn as nn
from benchmark_utils.base_solver import CompetSolver


class EEGNet(nn.Module):
    def __init__(self, n_chans=32, n_times=120, n_outputs=1536,
                 F1=8, D=2, F2=16, kernel_size=64, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_size+1), padding=(0, kernel_size//2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1*D, (n_chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(F1*D, F1*D, (1, 17), padding=(0, 8), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        dummy = torch.zeros(1, 1, n_chans, n_times)
        flat = self.net(dummy).flatten(1).shape[1]
        self.proj = nn.Linear(flat, n_outputs)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net(x).flatten(1)
        return self.proj(x)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        out = self(X)
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
        return out


class Solver(CompetSolver):
    name = "EEGNet"

    def load_model(self, meta):
        model = EEGNet(
            n_chans=meta["n_chans"],
            n_times=meta["n_times"],
            n_outputs=meta["n_outputs"],
        )
        weights_path = meta["submission_dir"] / "weights.pt"
        state_dict = torch.load(
            weights_path,
            map_location=meta["device"],
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        return model.to(meta["device"]).eval()
