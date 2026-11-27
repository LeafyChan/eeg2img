"""
SleepNet encoder — outputs (batch, time, hidden_dim) token sequence.
NeuralBench's aggregation='mean' pools over time to get (batch, hidden_dim),
then adds its own regression head.
"""

import torch
import torch.nn as nn


def _make_proj(n_channels: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(n_channels, n_channels * 3, kernel_size=31, padding=15, groups=n_channels),
        nn.ELU(),
        nn.Conv1d(n_channels * 3, hidden_dim, kernel_size=1),
        nn.BatchNorm1d(hidden_dim),
        nn.ELU(),
        nn.Dropout(dropout),
    )


class SleepNetEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout
        self.projections = nn.ModuleDict({
            str(n): _make_proj(n, hidden_dim, dropout)
            for n in [3, 4, 6, 19, 64, 128]
        })
        # Temporal encoder — preserves time dimension
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
        )

    def _get_proj(self, n_channels: int, device: torch.device) -> nn.Module:
        key = str(n_channels)
        if key not in self.projections:
            self.projections[key] = _make_proj(n_channels, self.hidden_dim, self.dropout_p).to(device)
        return self.projections[key]

    def forward(self, x: torch.Tensor, channel_positions: torch.Tensor = None) -> torch.Tensor:
        """
        Returns (batch, time, hidden_dim) — token sequence.
        NeuralBench's mean aggregation pools over time dimension.
        """
        n_channels = x.shape[1]
        proj = self._get_proj(n_channels, x.device)

        # (batch, channels, time) -> (batch, hidden_dim, time)
        x = proj(x)
        x = self.temporal(x)

        # (batch, hidden_dim, time) -> (batch, time, hidden_dim)
        return x.transpose(1, 2)


if __name__ == "__main__":
    model = SleepNetEncoder()
    for n_ch in [3, 4, 6, 19, 64, 128]:
        out = model(torch.randn(8, n_ch, 600))
        print(f"channels={n_ch:3d} -> {out.shape} ✓")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
