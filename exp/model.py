import torch
import torch.nn as nn
import math

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
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T]

class IMU2PoseTransformer(nn.Module):
    def __init__(self,
                 input_size=4*3,        # 12 IMU channels
                 d_model=96,
                 nhead=4,
                 num_layers=3,
                 dim_feedforward=256,
                 dropout=0.1,
                 out_dim=8*3):           # Output dimension is (J * 3)
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.posenc = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.head = nn.Linear(d_model, out_dim)
        
        self.out_dim = out_dim
        self.num_joints = out_dim // 3

    def forward(self, x, target_shape=None):
        """
        Args:
            x: Tensor of shape (B, T, 4, 3)
        Returns:
            output: Tensor of shape (B, T, J, 3)
        """
        B, T, S, C = x.shape  # Batch, Time, Sensors (4), Channels (3)
        x = x.view(B, T, S * C)  # Flatten 4*3=12 -> (B, T, 12)
        
        h = self.input_proj(x)     # (B, T, d_model)
        h = self.posenc(h)         # (B, T, d_model)
        h = self.encoder(h)        # (B, T, d_model)
        out = self.head(h)         # (B, T, out_dim)
        
        if target_shape is not None:
            B, T, J, C = target_shape
            return out.view(B, T, J, C)  # (B, T, J, 3)
        else:
            return out
