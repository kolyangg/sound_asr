import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConformerModel(nn.Module):
    def __init__(self, n_feats, n_tokens, d_model=144, num_layers=16, d_decoder=320):
        super(ConformerModel, self).__init__()

        self.n_feats = n_feats
        self.n_tokens = n_tokens

        # Compute subsampled frequency dimension
        self.subsampled_dim = self.compute_subsampled_dim()

        # Convolutional subsampling
        self.conv_subsample = Conv2dSubsampling(d_model=d_model)
        self.linear_proj = nn.Linear(d_model * self.subsampled_dim, d_model)

        # Encoder
        self.encoder = ConformerEncoder(d_input=d_model, d_model=d_model, num_layers=num_layers)

        # Decoder
        self.decoder = LSTMDecoder(d_encoder=d_model, d_decoder=d_decoder, num_classes=n_tokens)

        # Fully connected layer for classification
        self.fc = nn.Linear(d_decoder, n_tokens)

    def compute_subsampled_dim(self):
        freq = (self.n_feats - 1) // 2 + 1  # After first Conv2D
        freq = (freq - 1) // 2 + 1          # After second Conv2D
        return freq

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Forward pass of the Conformer-based ASR model.

        Args:
            spectrogram (Tensor): Input spectrogram of shape (B, F, T).
            spectrogram_length (Tensor): Original spectrogram lengths.

        Returns:
            dict: Contains "log_probs" and "log_probs_length".
        """
        print(f"[DEBUG] Input spectrogram shape: {spectrogram.shape}")
        if torch.isnan(spectrogram).any() or torch.isinf(spectrogram).any():
            print("[ERROR] NaN or Inf in input spectrogram!")

        # Subsampling and projection
        x = self.conv_subsample(spectrogram)
        print(f"[DEBUG] After Conv2dSubsampling: shape={x.shape}, min={x.min().item()}, max={x.max().item()}")
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[ERROR] NaN or Inf after Conv2dSubsampling!")
        x = torch.clamp(x, min=-1e3, max=1e3)

        x = self.linear_proj(x)
        print(f"[DEBUG] After Linear Projection: shape={x.shape}, min={x.min().item()}, max={x.max().item()}")
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[ERROR] NaN or Inf after Linear Projection!")
        x = torch.clamp(x, min=-1e3, max=1e3)

        # Encoder
        x = self.encoder(x)
        print(f"[DEBUG] After Encoder: shape={x.shape}, min={x.min().item()}, max={x.max().item()}")
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[ERROR] NaN or Inf after Encoder!")
        x = torch.clamp(x, min=-1e3, max=1e3)

        # Decoder
        x = self.decoder(x)
        print(f"[DEBUG] After Decoder: shape={x.shape}, min={x.min().item()}, max={x.max().item()}")
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[ERROR] NaN or Inf after Decoder!")
        x = torch.clamp(x, min=-1e3, max=1e3)

        # Apply log_softmax to decoder output
        log_probs = F.log_softmax(x, dim=-1)
        print(f"[DEBUG] After log_softmax: shape={log_probs.shape}, min={log_probs.min().item()}, max={log_probs.max().item()}")
        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            print("[ERROR] NaN or Inf after log_softmax!")

        # Compute output lengths
        log_probs_length = spectrogram_length // 4

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}


class Conv2dSubsampling(nn.Module):
    def __init__(self, d_model):
        super(Conv2dSubsampling, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.module[0](x.unsqueeze(1))  # First Conv2D
        print(f"[DEBUG] After first Conv2D: min={x.min().item()}, max={x.max().item()}")
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[ERROR] NaN or Inf after first Conv2D!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
        x = torch.clamp(x, min=-1e3, max=1e3)  # Clamp to prevent extreme values
        x = x + torch.randn_like(x) * 1e-6  # Add small noise for numerical stability

        x = self.module[1](x)  # First ReLU
        x = self.module[2](x)  # Second Conv2D
        print(f"[DEBUG] After second Conv2D: min={x.min().item()}, max={x.max().item()}")
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("[ERROR] NaN or Inf after second Conv2D!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
        x = torch.clamp(x, min=-1e3, max=1e3)
        x = x + torch.randn_like(x) * 1e-6  # Add small noise for numerical stability

        x = self.module[3](x)  # Second ReLU
        batch_size, d_model, subsampled_freq, subsampled_time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, subsampled_time, -1)
        return x


class ConformerEncoder(nn.Module):
    def __init__(self, d_input, d_model, num_layers):
        super(ConformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(d_model=d_model) for _ in range(num_layers)
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            print(f"[DEBUG] After Encoder Layer {i}: shape={x.shape}, min={x.min().item()}, max={x.max().item()}")
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"[ERROR] NaN or Inf after Encoder Layer {i}!")
            x = torch.clamp(x, min=-1e3, max=1e3)
        return x


class ConformerBlock(nn.Module):
    def __init__(self, d_model, conv_kernel_size=31):
        super(ConformerBlock, self).__init__()
        self.ff1 = FeedForwardBlock(d_model)
        self.attention = RelativeMultiHeadAttention(d_model=d_model)
        self.conv_block = ConvBlock(d_model, conv_kernel_size=conv_kernel_size)
        self.ff2 = FeedForwardBlock(d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=6.1e-5)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        x = x + self.attention(x)
        x = x + self.conv_block(x)
        x = x + 0.5 * self.ff2(x)
        return self.layer_norm(x)


class LSTMDecoder(nn.Module):
    def __init__(self, d_encoder, d_decoder, num_classes):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size=d_encoder, hidden_size=d_decoder, num_layers=1, batch_first=True)
        self.linear = nn.Linear(d_decoder, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.linear(x)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, expansion=4):
        super(FeedForwardBlock, self).__init__()
        self.module = nn.Sequential(
            nn.LayerNorm(d_model, eps=6.1e-5),
            nn.Linear(d_model, d_model * expansion),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.module(x)


class ConvBlock(nn.Module):
    def __init__(self, d_model, conv_kernel_size):
        super(ConvBlock, self).__init__()
        self.module = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=conv_kernel_size,
                stride=1,
                padding=(conv_kernel_size // 2),
                groups=d_model
            ),
            nn.BatchNorm1d(d_model, eps=6.1e-5),
            nn.SiLU(),
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.module(x)
        return x.transpose(1, 2)


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super(RelativeMultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output
