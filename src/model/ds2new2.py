import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils

class DeepSpeech2Model(nn.Module):
    def __init__(self, n_feats, n_tokens, rnn_hidden=512, rnn_layers=5, bidirectional=True):
        super(DeepSpeech2Model, self).__init__()
        self.rnn_hidden = rnn_hidden
        self.rnn_layers_count = rnn_layers
        self.bidirectional = bidirectional
        self.n_tokens = n_tokens
        self.n_feats = n_feats

        # Define convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(41, 11),
                      stride=(2, 1), padding=(20, 5)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(21, 11),
                      stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(21, 11),
                      stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
        )

        # Compute RNN input size after convolutional layers
        self.rnn_input_size = self.compute_rnn_input_size()
        # print(f"[DEBUG] Computed RNN input size: {self.rnn_input_size}")

        # Initialize RNN layers and BatchNorm layers
        self.rnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(self.rnn_layers_count):
            input_size = self.rnn_input_size if i == 0 else self.rnn_hidden * (2 if self.bidirectional else 1)
            # print(f"[DEBUG] Initializing RNN layer {i+1} with input_size={input_size}")
            rnn = nn.GRU(
                input_size=input_size,
                hidden_size=self.rnn_hidden,
                num_layers=1,
                bidirectional=self.bidirectional,
                batch_first=True
            )
            self.rnn_layers.append(rnn)
            bn = nn.BatchNorm1d(self.rnn_hidden * (2 if self.bidirectional else 1))
            self.batch_norms.append(bn)

        # Fully connected layer
        self.fc = nn.Linear(self.rnn_hidden * (2 if self.bidirectional else 1), self.n_tokens, bias=False)

        # Initialize weights
        self.apply(self.initialize_weights)

        self.__str__()

    def compute_rnn_input_size(self):
        initial_freq_bins = self.n_feats  # Corrected

        # After first Conv2D layer
        freq = math.floor((initial_freq_bins + 2 * 20 - 41) / 2 + 1)  # 128 -> 64
        # After second Conv2D layer
        freq = math.floor((freq + 2 * 10 - 21) / 2 + 1)  # 64 -> 32
        # After third Conv2D layer
        freq = math.floor((freq + 2 * 10 - 21) / 2 + 1)  # 32 -> 16

        rnn_input_size = freq * 96  # 16 * 96 = 1536

        # print(f"[DEBUG] Frequency after convolutions: {freq}")
        # print(f"[DEBUG] rnn_input_size: {rnn_input_size}")
        return rnn_input_size

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Forward pass of DeepSpeech2.

        Args:
            spectrogram (Tensor): Input spectrogram of shape (B, F, T).
            spectrogram_length (Tensor): Original spectrogram lengths.
            **batch: Additional batch information (unused).

        Returns:
            dict: Contains "log_probs" and "log_probs_length".
        """
        MIN_TIME = 16  # Define minimum time dimension after convolution

        # Pad the input if necessary
        if spectrogram.size(2) < MIN_TIME:
            pad_amount = MIN_TIME - spectrogram.size(2)
            spectrogram = F.pad(spectrogram, (0, pad_amount))  # Pad only the time dimension
            spectrogram_length = spectrogram_length + pad_amount
            # print(f"[DEBUG] After padding: {spectrogram.shape}, lengths: {spectrogram_length}")

        # Input shape: (B, F, T) -> (B, 1, F, T)
        x = spectrogram.unsqueeze(1)
        # print(f"[DEBUG] After unsqueeze: {x.shape}")

        # Pass through convolutional layers
        x = self.conv_layers(x)  # Shape: (B, C, F', T')
        B, C, Freq, Time = x.size()
        # print(f"[DEBUG] After conv_layers: {x.shape}")

        # Reshape for RNN input: (B, T', C * F')
        x = x.permute(0, 3, 1, 2).contiguous().view(B, Time, C * Freq)
        # print(f"[DEBUG] After reshaping for RNN: {x.shape}")  # Should be (B, T', C*F')

        # Pass through RNN layers with BatchNorm
        for idx, (rnn, bn) in enumerate(zip(self.rnn_layers, self.batch_norms)):
            x, _ = rnn(x)  # Shape: (B, T', hidden_size * num_directions)
            # print(f"[DEBUG] After RNN layer {idx+1}: {x.shape}")
            # Apply BatchNorm1d
            x = x.permute(0, 2, 1)  # (B, C, T')
            x = bn(x)
            x = x.permute(0, 2, 1)  # (B, T', C)
            # print(f"[DEBUG] After BatchNorm layer {idx+1}: {x.shape}")

            # # Check for NaNs or Infs
            # if torch.isnan(x).any():
            #     print(f"[DEBUG] NaNs detected after RNN and BatchNorm layer {idx+1}.")
            # if torch.isinf(x).any():
            #     print(f"[DEBUG] Infs detected after RNN and BatchNorm layer {idx+1}.")

        # Fully connected layer to map RNN outputs to token classes
        x = self.fc(x)  # Shape: (B, T', n_tokens)
        # print(f"[DEBUG] After fc layer: {x.shape}")

        # Check for NaNs or Infs before log_softmax
        # if torch.isnan(x).any():
        #     print("[DEBUG] NaNs detected in FC layer output.")
        # if torch.isinf(x).any():
        #     print("[DEBUG] Infs detected in FC layer output.")

        # Apply log_softmax
        log_probs = F.log_softmax(x, dim=-1)
        # print(f"[DEBUG] After log_softmax: {log_probs.shape}")

        # Check for NaNs or Infs after log_softmax
        # if torch.isnan(log_probs).any():
        #     print("[DEBUG] NaNs detected in log_probs after log_softmax.")
        # if torch.isinf(log_probs).any():
        #     print("[DEBUG] Infs detected in log_probs after log_softmax.")

        # Calculate output lengths based on the actual sequence reduction
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        # print(f"[DEBUG] log_probs_length: {log_probs_length}")

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        Calculate proper output lengths based on convolution operations.

        Args:
            input_lengths (Tensor): Original input lengths.

        Returns:
            Tensor: Adjusted output lengths after convolution.
        """
        # Convolution parameters from the three Conv2D layers
        # Conv1: kernel_size=(41,11), stride=(2,1), padding=(20,5)
        # Conv2: kernel_size=(21,11), stride=(2,1), padding=(10,5)
        # Conv3: kernel_size=(21,11), stride=(2,1), padding=(10,5)
        # Only the time dimension is affected by stride and padding
        p, s = self.get_conv_num()

        for i in range(len(s)):
            input_lengths = ((input_lengths + p[i]) // s[i]) + 1

        return input_lengths

    def get_conv_num(self):
        """
        Retrieve padding and stride values from convolutional layers.

        Returns:
            Tuple[List[int], List[int]]: Lists of padding and stride values.
        """
        p, s = [], []
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                # Extract padding and stride for the time dimension
                kernel_size = layer.kernel_size
                padding_time = layer.padding[1]
                stride_time = layer.stride[1]
                # Calculate temp based on original model's formula
                temp = 2 * padding_time - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1
                p.append(temp)
                s.append(stride_time)
        return p, s
    



    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info

