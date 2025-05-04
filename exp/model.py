import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Simple LSTM --------------------------------------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_size=4*3, hidden_size=8*3, num_layers=2):
        super(SimpleNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

    def forward(self, x):
        x = x.flatten(2) # (N, T, 12)
        out, _ = self.lstm(x) # (N, T, 24)
        return out
    

# Transformer -----------------------------------------------------------
class ConvTransformer(nn.Module):
    """
    Conv + Transformer encoder for IMU→MoCap regression.
    Input:  x of shape (B, window_size, input_dim)
    Output: Tensor of shape (B, window_size * 8 * 3)
    """
    def __init__(
        self,
        input_dim: int,              # e.g. 12
        transformer_dim: int,        # d_model, e.g. 64
        window_size: int,            # sequence length
        nhead: int,                  # attention heads, e.g. 8
        dim_feedforward: int,        # transformer FF size, e.g. 256
        transformer_dropout: float,  # dropout rate, e.g. 0.1
        transformer_activation: str, # "gelu" or "relu"
        num_encoder_layers: int,     # number of Transformer layers, e.g. 6
        encode_position: bool = False
    ):
        super().__init__()
        d_model = transformer_dim
        W = window_size

        # 1) Conv1d stack to embed raw IMU → (B, d_model, W)
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim,   d_model, 1), nn.GELU(),
            nn.Conv1d(d_model,     d_model, 1), nn.GELU(),
            nn.Conv1d(d_model,     d_model, 1), nn.GELU(),
            nn.Conv1d(d_model,     d_model, 1), nn.GELU(),
        )
        
        # 2) Positional embedding (optional)
        self.encode_position = encode_position
        if encode_position:
            # one position vector per timestep
            self.position_embed = nn.Parameter(torch.randn(W, 1, d_model))

        # 3) Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation=transformer_activation
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )

        # 4) Regression head (token‑wise)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(transformer_dropout),
            nn.Linear(d_model // 4, 8 * 3)
        )

        # 5) Xavier init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: FloatTensor of shape (B, window_size, 4, 3) OR already flattened (B, W, C)
        returns: FloatTensor of shape (B, W*8*3)
        """
        # if it's still (B, W, 4, 3), flatten sensor×axis → channels
        if x.dim() == 4:
            B, W, S, A = x.shape   # S=4 sensors, A=3 axes
            x = x.reshape(B, W, S * A)

        # ── embed + reshape ───────────────────────────────
        x = x.transpose(1, 2)         # (B, C, W)
        x = self.input_proj(x)       # (B, d_model, W)
        x = x.permute(2, 0, 1)       # (W, B, d_model)

        if self.encode_position:
            x = x + self.position_embed

        # ── transformer ─────────────────────────────────────
        x = self.transformer_encoder(x)  # (W, B, d_model)
        x = x.permute(1, 0, 2)           # (B, W, d_model)

        # ── regression head ──────────────────────────────────
        B, W, D = x.shape
        x = x.reshape(B * W, D)          # (B*W, d_model)
        out = self.regression_head(x)    # (B*W, 8*3)
        out = out.view(B, W * 8 * 3)     # (B, W*8*3)
        return out
