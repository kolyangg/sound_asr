import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class BNReluRNN(nn.Module):
    """
    Recurrent neural network with batch normalization layer & ReLU activation function.

    Args:
        input_size (int): size of input
        hidden_state_dim (int): the number of features in the hidden state h
        rnn_type (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (default: True)
        dropout_p (float, optional): dropout probability (default: 0.1)
    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            input_size: int,
            hidden_state_dim: int = 512,
            rnn_type: str = 'gru',
            bidirectional: bool = True,
            dropout_p: float = 0.1,
    ):
        super().__init__()
        self.hidden_state_dim = hidden_state_dim
        self.batch_norm = nn.BatchNorm1d(input_size)
        rnn_cell = self.supported_rnns.get(rnn_type.lower())
        if rnn_cell is None:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")
        self.rnn = rnn_cell(
            input_size=input_size,
            hidden_size=hidden_state_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional,
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        """
        inputs: (batch, time, dim)
        input_lengths: (batch,)
        """
        # Apply BN + ReLU
        inputs = inputs.transpose(1, 2)  # (B, dim, T)
        inputs = self.batch_norm(inputs)
        inputs = F.relu(inputs)
        inputs = inputs.transpose(1, 2)  # back to (B, T, dim)

        # Forward through RNN
        outputs, _ = self.rnn(inputs)
        return outputs

class DeepSpeech2Model(nn.Module):
    """
    Adapted DS2 that expects:
      - spectrogram of shape (B, F, T), i.e. (batch, freq_bins, time_frames)
      - spectrogram_length as (B,) with original number of frames
    Returns:
      - dict with {"log_probs": (B, T', n_tokens), "log_probs_length": (B,)}
    """
    def __init__(
        self,
        n_feats: int,         # Frequency dimension
        n_tokens: int,        # Number of output tokens (vocab size)
        rnn_hidden: int = 512,
        rnn_layers: int = 5,
        bidirectional: bool = True,
        dropout_p: float = 0.1,
        rnn_type: str = 'gru',
        activation: str = 'hardtanh',
    ):
        super().__init__()
        self.rnn_hidden = rnn_hidden
        self.rnn_layers_count = rnn_layers
        self.bidirectional = bidirectional
        self.n_tokens = n_tokens
        self.n_feats = n_feats
        self.dropout_p = dropout_p
        self.rnn_type = rnn_type

        self.debug = False

        # Define the 2D convolution stack with updated strides
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(41, 11),
                stride=(2, 2),  # First Conv: stride=(2,2)
                padding=(20, 5),
                bias=False
            ),
            nn.BatchNorm2d(32),
            self._get_activation(activation),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(21, 11),
                stride=(2, 2),  # Second Conv: stride=(2,2)
                padding=(10, 5),
                bias=False
            ),
            nn.BatchNorm2d(32),
            self._get_activation(activation),
            nn.Conv2d(
                in_channels=32,
                out_channels=96,
                kernel_size=(21, 11),
                stride=(2, 1),  # Third Conv: stride=(2,1)
                padding=(10, 5),
                bias=False
            ),
            nn.BatchNorm2d(96),
            self._get_activation(activation),
        )

        # # RNN layers and FC will be lazily initialized
        # self.rnn_layers = None
        # self.batch_norms = None
        # self.fc = None

        self.conv_out_channels = 96
        self.conv_out_freq = 16     # Based on your conv setup
        self.first_rnn_input_size = self.conv_out_channels * self.conv_out_freq  # 96 * 16 = 1536

        # Build RNN layers
        input_sizes = self._make_rnn_input_sizes(
            first_input_size=self.first_rnn_input_size,
            layers=rnn_layers,
            hidden=rnn_hidden,
            bidirectional=bidirectional
        )

        self.rnn_layers = nn.ModuleList([
            BNReluRNN(
                input_size=input_sizes[i],
                hidden_state_dim=rnn_hidden,
                rnn_type=rnn_type,
                bidirectional=bidirectional,
                dropout_p=dropout_p,
            )
            for i in range(rnn_layers)
        ])

        # Build BN for each RNN output (i.e. hidden * directions)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(rnn_hidden * (2 if bidirectional else 1))
            for _ in range(rnn_layers)
        ])

        # Final linear layer
        self.fc = nn.Linear(rnn_hidden * (2 if bidirectional else 1), n_tokens)

    def _make_rnn_input_sizes(self, first_input_size: int, layers: int, hidden: int, bidirectional: bool):
        sizes = [first_input_size]
        out_dim = hidden * (2 if bidirectional else 1)
        for _ in range(layers - 1):
            sizes.append(out_dim)
        return sizes
        

    def _get_activation(self, activation):
        if activation.lower() == 'hardtanh':
            return nn.Hardtanh(0, 20, inplace=True)
        elif activation.lower() == 'relu':
            return nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, spectrogram: Tensor, spectrogram_length: Tensor, **batch):
        if self.debug:
            print(f"[DEBUG] Input shape: {spectrogram.shape}, lengths: {spectrogram_length}")

        x = spectrogram.unsqueeze(1)
        if self.debug:
            print(f"[DEBUG] After unsqueeze: shape => {x.shape}")

        # 1) Pass through conv_layers step by step
        for idx, layer in enumerate(self.conv_layers):
            x = layer(x)
            if self.debug:
                print(f"[DEBUG] After conv layer {idx}, shape => {x.shape}")

        B, C, Freq, Time = x.size()
        if Time == 0:
            if self.debug:
                print("[ERROR] Time dimension = 0 after conv stack! This sample can't proceed.")
            # Possibly handle or skip
            # For debugging, do something like:
            # return {"log_probs": torch.zeros(B, 0, self.n_tokens), "log_probs_length": torch.zeros(B)}

        # 2) Flatten & RNN
        x = x.permute(0, 3, 1, 2).contiguous().view(B, Time, C * Freq)
        if self.debug:
            print(f"[DEBUG] After flatten for RNN => {x.shape}")

        # RNN Layers
        for i, (rnn, bn) in enumerate(zip(self.rnn_layers, self.batch_norms)):
            x = rnn(x, spectrogram_length)
            if self.debug:
                print(f"[DEBUG] After RNN {i}, shape => {x.shape}")
            # If x.shape is (B, 0, hidden_dim), confirm Time=0
            if x.size(1) == 0:
                if self.debug:
                    print(f"[ERROR] Time dimension collapsed to 0 after RNN {i}.")
            x = bn(x.transpose(1, 2)).transpose(1, 2)

        # FC & log_probs
        x = self.fc(x)
        if self.debug:
            print(f"[DEBUG] After FC => {x.shape}")
        log_probs = F.log_softmax(x, dim=-1)

        log_probs_length = self.transform_input_lengths(spectrogram_length)
        if self.debug:
            print(f"[DEBUG] Final output => {log_probs.shape}, lengths => {log_probs_length}")

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}


    def transform_input_lengths(self, input_lengths: Tensor) -> Tensor:
        """
        Calculate proper output lengths based on convolution operations:
        - First Conv: stride=2 in time
        - Second Conv: stride=2 in time
        - Third Conv: stride=1 in time
        Total time reduction factor: 4
        """
        # First Conv: stride=2
        lengths = ((input_lengths - 1) // 2) + 1
        if self.debug:
            print(f"[DEBUG] After First Stride Reduction: {lengths}")
        # Second Conv: stride=2
        lengths = ((lengths - 1) // 2) + 1
        if self.debug:
            print(f"[DEBUG] After Second Stride Reduction: {lengths}")
        # Third Conv: stride=1 (no reduction)
        # No change needed
        return lengths.long()
