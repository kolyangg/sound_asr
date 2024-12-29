import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# class BNReluRNN(nn.Module):
#     """
#     Recurrent neural network with batch normalization layer & ReLU activation function.

#     Args:
#         input_size (int): size of input
#         hidden_state_dim (int): the number of features in the hidden state h
#         rnn_type (str, optional): type of RNN cell (default: gru)
#         bidirectional (bool, optional): if True, becomes a bidirectional encoder (default: True)
#         dropout_p (float, optional): dropout probability (default: 0.1)

#     Inputs: inputs, input_lengths
#         - inputs (batch, time, dim): Tensor containing input vectors
#         - input_lengths: Tensor containing containing sequence lengths

#     Returns: outputs
#         - outputs: Tensor produced by the BNReluRNN module
#     """
#     supported_rnns = {
#         'lstm': nn.LSTM,
#         'gru': nn.GRU,
#         'rnn': nn.RNN,
#     }

#     def __init__(
#             self,
#             input_size: int,
#             hidden_state_dim: int = 512,
#             rnn_type: str = 'gru',
#             bidirectional: bool = True,
#             dropout_p: float = 0.1,
#     ):
#         super().__init__()
#         self.hidden_state_dim = hidden_state_dim
#         # BatchNorm over "features" dimension -> BN1d expects shape (B, C, T)
#         # We'll apply BN to the last dimension (features), so we rearrange input first.
#         self.batch_norm = nn.BatchNorm1d(input_size)

#         rnn_cell = self.supported_rnns[rnn_type]
#         self.rnn = rnn_cell(
#             input_size=input_size,
#             hidden_size=hidden_state_dim,
#             num_layers=1,
#             bias=True,
#             batch_first=True,
#             dropout=dropout_p,
#             bidirectional=bidirectional,
#         )

#     def forward(self, inputs: Tensor, input_lengths: Tensor):
#         """
#         inputs: (batch, time, dim)
#         input_lengths: (batch,) - not used directly by the GRU itself here,
#                        but important if we want to do pack_padded_sequence.
#         """
#         # Apply BN + ReLU
#         # BN expects (B, C, T) so we do (B, T, C) -> (B, C, T), BN, then back.
#         inputs = inputs.transpose(1, 2)  # (B, dim, T)
#         inputs = self.batch_norm(inputs)
#         inputs = F.relu(inputs)
#         inputs = inputs.transpose(1, 2)  # back to (B, T, dim)

#         # If you want to use pack/pad, do something like:
#         # packed = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
#         # outputs, _ = self.rnn(packed)
#         # outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
#         #
#         # But if you don't do dynamic padding:
#         outputs, _ = self.rnn(inputs)

#         return outputs


# class DeepSpeech2Model(nn.Module):
#     """
#     Adapted DS2 that expects:
#       - spectrogram of shape (B, F, T), i.e. (batch, freq_bins, time_frames)
#       - spectrogram_length as (B,) with original number of frames
#     Returns:
#       - dict with {"log_probs": (B, T', n_tokens), "log_probs_length": (B,)}
#     """
#     def __init__(
#         self,
#         n_feats: int,         # Frequency dimension
#         n_tokens: int,        # Number of output tokens (vocab size)
#         rnn_hidden: int = 512,
#         rnn_layers: int = 5,
#         bidirectional: bool = True,
#         dropout_p: float = 0.1,
#         rnn_type: str = 'gru',
#         activation: str = 'hardtanh',
#     ):
#         super().__init__()
#         self.rnn_hidden = rnn_hidden
#         self.rnn_layers_count = rnn_layers
#         self.bidirectional = bidirectional
#         self.n_tokens = n_tokens
#         self.n_feats = n_feats

#         # Define the 2D convolution stack
#         # By default from the original DS2 paper:
#         #   conv1: kernel=(41,11), stride=(2,2), pad=(20,5)
#         #   conv2: kernel=(21,11), stride=(2,1), pad=(10,5)
#         # Some people add a 3rd conv or modify strides, but let's keep the typical pattern.
#         self.conv = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=1,
#                 out_channels=32,
#                 kernel_size=(41, 11),
#                 stride=(2, 2),
#                 padding=(20, 5),
#                 bias=False
#             ),
#             nn.BatchNorm2d(32),
#             self._get_activation(activation),
#             nn.Conv2d(
#                 in_channels=32,
#                 out_channels=32,
#                 kernel_size=(21, 11),
#                 stride=(2, 1),
#                 padding=(10, 5),
#                 bias=False
#             ),
#             nn.BatchNorm2d(32),
#             self._get_activation(activation),
#         )

#         # We'll lazily init the RNN + FC layers the first time we see data
#         # so that we know the correct input dimension after the conv layers
#         self.rnn_layers = None
#         self.batch_norms = None
#         self.fc = None
#         self.dropout_p = dropout_p
#         self.rnn_type = rnn_type

#     def forward(self, spectrogram: Tensor, spectrogram_length: Tensor, **batch):
#         """
#         spectrogram: (B, F, T)
#         spectrogram_length: (B,)
#         """
#         # Expand to NCDHW-like shape => (B, 1, F, T)
#         x = spectrogram.unsqueeze(1)   # (B, 1, F, T)

#         # Convolution
#         x = self.conv(x)  # => (B, C, F', T')
#         B, C, Freq, Time = x.size()

#         # Convert for RNN => (B, Time, C*Freq)
#         x = x.permute(0, 3, 1, 2).contiguous().view(B, Time, C * Freq)

#         # Lazy init the RNN, BN, FC
#         if self.rnn_layers is None:
#             input_size = C * Freq
#             self.rnn_layers = nn.ModuleList([
#                 BNReluRNN(
#                     input_size=input_size if i == 0 else self.rnn_hidden * (2 if self.bidirectional else 1),
#                     hidden_state_dim=self.rnn_hidden,
#                     rnn_type=self.rnn_type,
#                     bidirectional=self.bidirectional,
#                     dropout_p=self.dropout_p,
#                 ) for i in range(self.rnn_layers_count)
#             ])
#             # Each BNReluRNN has its own internal BN, so we don't need separate BN modules here
#             self.fc = nn.Linear(self.rnn_hidden * (2 if self.bidirectional else 1), self.n_tokens)
#             self.to(x.device)

#         # Forward through RNN layers
#         for layer in self.rnn_layers:
#             x = layer(x, spectrogram_length)  # shape stays (B, T', hidden*direction)

#         # Final projection => (B, T', n_tokens)
#         x = self.fc(x)

#         # Log softmax
#         log_probs = F.log_softmax(x, dim=-1)

#         # Adjust output sequence lengths after conv
#         log_probs_length = self.transform_input_lengths(spectrogram_length)

#         return {"log_probs": log_probs, "log_probs_length": log_probs_length}

#     def transform_input_lengths(self, input_lengths: Tensor) -> Tensor:
#         """
#         For the given convolution pattern:
#           conv1 => stride=(2,2): the time dimension is halved
#           conv2 => stride=(2,1): the time dimension is halved again
#         So total time-downsampling = factor of 2 * 2 = 4
#         """
#         # Each stride of 2 in the time dimension roughly halves the time length.
#         out_lengths = (input_lengths.float() / 2.0).floor()
#         out_lengths = (out_lengths / 2.0).floor()
#         return out_lengths.long()

#     def _get_activation(self, activation):
#         if activation.lower() == 'hardtanh':
#             return nn.Hardtanh(0, 20, inplace=True)
#         elif activation.lower() == 'relu':
#             return nn.ReLU(inplace=True)
#         else:
#             raise ValueError(f"Unsupported activation: {activation}")


import torch
import torch.nn.functional as F

class DeepSpeech2Model(nn.Module):
    def __init__(self, n_feats, n_tokens, rnn_hidden=512, rnn_layers=5, bidirectional=True):
        super().__init__()
        self.rnn_hidden = rnn_hidden
        self.rnn_layers_count = rnn_layers
        self.bidirectional = bidirectional
        self.n_tokens = n_tokens
        self.n_feats = n_feats

        # Modify convolution layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 1), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )

        # Lazy initialization for RNN and FC layers
        self.rnn_layers = None
        self.fc = None

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Forward pass of DeepSpeech2.
        Args:
            spectrogram (Tensor): Input spectrogram of shape (B, F, T).
            spectrogram_length (Tensor): Original spectrogram lengths.
        Returns:
            dict: Contains "log_probs" and "log_probs_length".
        """
        MIN_TIME = 16  # Define minimum time dimension after convolution

        # Debug: Input shapes
        # print(f"[DEBUG] Input shape: {spectrogram.shape}, lengths: {spectrogram_length}")

        # Pad the input if necessary
        if spectrogram.size(2) < MIN_TIME:
            pad_amount = MIN_TIME - spectrogram.size(2)
            spectrogram = F.pad(spectrogram, (0, pad_amount))  # Pad only the time dimension
            spectrogram_length = spectrogram_length + pad_amount
            print(f"[DEBUG] After padding: {spectrogram.shape}, lengths: {spectrogram_length}")

        # Input shape: (B, F, T) -> (B, 1, F, T)
        x = spectrogram.unsqueeze(1)

        # Pass through convolutional layers
        x = self.conv_layers(x)  # Shape: (B, C, F', T')
        B, C, Freq, Time = x.size()

        # Flatten for RNN input: (B, T', C*F')
        x = x.permute(0, 3, 1, 2).contiguous().view(B, Time, C * Freq)

        # Dynamically initialize RNN layers if needed
        if self.rnn_layers is None:
            input_size = C * Freq
            self.rnn_layers = nn.ModuleList([
                nn.GRU(input_size=input_size if i == 0 else self.rnn_hidden * (2 if self.bidirectional else 1),
                       hidden_size=self.rnn_hidden, num_layers=1, batch_first=True, bidirectional=self.bidirectional)
                for i in range(self.rnn_layers_count)
            ])
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(self.rnn_hidden * (2 if self.bidirectional else 1))
                for _ in range(self.rnn_layers_count)
            ])
            self.fc = nn.Linear(self.rnn_hidden * (2 if self.bidirectional else 1), self.n_tokens)
            self.to(x.device)

        # Pass through RNN layers with BatchNorm
        for rnn, bn in zip(self.rnn_layers, self.batch_norms):
            x, _ = rnn(x)
            x = bn(x.transpose(1, 2)).transpose(1, 2)

        # Fully connected layer for character probabilities
        x = self.fc(x)  # Shape: (B, T', n_tokens)

        # Apply log_softmax - keep in (B, T, C) format
        log_probs = F.log_softmax(x, dim=-1)

        # Calculate output lengths based on the actual sequence reduction
        log_probs_length = self.transform_input_lengths(spectrogram_length)

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        Calculate proper output lengths based on convolution operations.
        """
        lengths = ((input_lengths - 1) // 2) + 1  # Adjust this as needed
        return lengths.long()
