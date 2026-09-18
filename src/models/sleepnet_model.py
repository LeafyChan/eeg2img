"""
SleepNet — custom temporal regression encoder for Track 3.
Drop-in replacement for EEGNet/REVE via neuralbench's evaluate_model API.

Interface contract (from NeuralBench docs):
- forward(x, channel_positions) -> (batch, 1) regression output
- x: (batch, channels, time) float32
- channel_positions: (batch, channels, 3) float32 — 3D head positions
- Must accept ANY channel count — never hardcode n_channels
"""

import torch
import torch.nn as nn


class SleepNetEncoder(nn.Module):
    """
    Temporal CNN encoder for sleep onset regression.
    Handles variable channel counts via dynamic depthwise convolutions.
    """

    def __init__(
        self,
        n_timepoints: int = 600,
        hidden_dim: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout

        # Temporal encoder — channel-agnostic (operates after projection)
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=15, stride=5, padding=7),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=9, stride=3, padding=4),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _get_spatial_proj(self, n_channels: int, device: torch.device) -> nn.Module:
        """
        Build a per-forward spatial projection from n_channels -> hidden_dim.
        Uses a simple learned linear mix across channels.
        Cached per channel count to avoid rebuilding every forward pass.
        """
        if not hasattr(self, '_proj_cache'):
            self._proj_cache = {}
        if n_channels not in self._proj_cache:
            proj = nn.Sequential(
                # Multi-scale depthwise temporal convolutions
                nn.Conv1d(n_channels, n_channels * 3, kernel_size=31,
                          padding=15, groups=n_channels),
                nn.ELU(),
                # Pointwise: mix channels -> hidden_dim
                nn.Conv1d(n_channels * 3, self.hidden_dim, kernel_size=1),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ELU(),
                nn.Dropout(self.dropout_p),
            ).to(device)
            self._proj_cache[n_channels] = proj
        return self._proj_cache[n_channels]

    def forward(self, x: torch.Tensor, channel_positions: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time) — any channel count
            channel_positions: (batch, channels, 3) — 3D head positions
        Returns:
            (batch, 1) — predicted seconds to N2 onset
        """
        n_channels = x.shape[1]

        # Dynamic spatial projection
        proj = self._get_spatial_proj(n_channels, x.device)
        x = proj(x)

        # Temporal encoding
        x = self.temporal(x)

        # Regression
        return self.head(x)


if __name__ == "__main__":
    model = SleepNetEncoder()
    for n_ch in [3, 4, 19, 64, 128]:
        x = torch.randn(8, n_ch, 600)
        out = model(x)
        print(f"channels={n_ch:3d} -> output={out.shape} ✓")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
