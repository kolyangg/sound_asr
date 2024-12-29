import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils


class DeepSpeech2Model(nn.Module):
    def __init__(self, n_feats, n_tokens, rnn_hidden=512, rnn_layers=5, bidirectional=True):
        super(DeepSpeech2Model, self).__init__()

        print("[DEBUG] Initializing DeepSpeech2Model...")
        print(f"  -> n_feats={n_feats}, n_tokens={n_tokens}, rnn_hidden={rnn_hidden}, "
              f"rnn_layers={rnn_layers}, bidirectional={bidirectional}")

        self.rnn_hidden = rnn_hidden
        self.rnn_layers_count = rnn_layers
        self.bidirectional = bidirectional
        self.n_tokens = n_tokens
        self.n_feats = n_feats

        # Define convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(41, 11),
                stride=(2, 1),
                padding=(20, 5)
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(21, 11),
                stride=(2, 1),
                padding=(10, 5)
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=96,
                kernel_size=(21, 11),
                stride=(2, 1),
                padding=(10, 5)
            ),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
        )

        # Compute RNN input size after convolutional layers
        self.rnn_input_size = self.compute_rnn_input_size()

        # print(f"[DEBUG] Computed RNN input size (conv freq bins * channels): {self.rnn_input_size}")

        # Initialize RNN layers and BatchNorm layers
        self.rnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(self.rnn_layers_count):
            input_size = (self.rnn_input_size if i == 0
                          else self.rnn_hidden * (2 if self.bidirectional else 1))
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
        self.fc = nn.Linear(
            self.rnn_hidden * (2 if self.bidirectional else 1),
            self.n_tokens,
            bias=False
        )

        # Initialize weights
        self.apply(self.initialize_weights)
        # print("[DEBUG] DeepSpeech2Model initialization done.\n")

    def compute_rnn_input_size(self):
        """
        Compute the frequency dimension after the 3 convolution layers.
        """
        initial_freq_bins = self.n_feats
        freq = initial_freq_bins

        # 1st conv: kernel_size=41, stride=2, padding=20
        freq = math.floor((freq + 2 * 20 - 41) / 2 + 1)
        # 2nd conv: kernel_size=21, stride=2, padding=10
        freq = math.floor((freq + 2 * 10 - 21) / 2 + 1)
        # 3rd conv: kernel_size=21, stride=2, padding=10
        freq = math.floor((freq + 2 * 10 - 21) / 2 + 1)

        rnn_input_size = freq * 96
        # print(f"[DEBUG] Frequency after all convs => {freq}, so rnn_input_size => {rnn_input_size}")
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
        MIN_TIME = 16  # Define a minimum time dimension after convolution

        B, F1, T = spectrogram.size()
        # print(f"[DEBUG] DS2 forward called, spectrogram shape => (B={B}, F={F1}, T={T})")
        # print(f"[DEBUG] spectrogram_length => {spectrogram_length}")

        # Pad if needed
        if T < MIN_TIME:
            pad_amount = MIN_TIME - T
            spectrogram = F.pad(spectrogram, (0, pad_amount))  # pad the time dimension only
            spectrogram_length = spectrogram_length + pad_amount
            # print(f"[DEBUG] Padded spectrogram => {spectrogram.shape}, new lengths => {spectrogram_length}")

        # Reshape => (B, 1, F, T)
        x = spectrogram.unsqueeze(1)
        # print(f"[DEBUG] After unsqueeze => {x.shape}")

        # Pass through conv layers
        x = self.conv_layers(x)
        B2, C2, Freq, Time = x.size()
        # print(f"[DEBUG] After conv_layers => shape (B={B2}, C={C2}, Freq={Freq}, Time={Time})")

        # Reshape for RNN => (B, Time, C*Freq)
        x = x.permute(0, 3, 1, 2).contiguous().view(B2, Time, C2 * Freq)
        # print(f"[DEBUG] Reshaped for RNN => {x.shape}")

        # Pass through RNN + BatchNorm
        for i, (rnn, bn) in enumerate(zip(self.rnn_layers, self.batch_norms)):
            x, _ = rnn(x)  # (B, Time, hidden * (2 if bi else 1))
            # print(f"[DEBUG] After RNN layer {i+1} => {x.shape}")

            x = x.permute(0, 2, 1)  # => (B, C, Time)
            x = bn(x)
            x = x.permute(0, 2, 1)  # => (B, Time, C)
            # print(f"[DEBUG] After batchnorm layer {i+1} => {x.shape}")

            # # Optional: check for NaNs or infs
            # if torch.isnan(x).any():
            #     print(f"[DEBUG] NaNs after RNN+BN layer {i+1}")
            # if torch.isinf(x).any():
            #     print(f"[DEBUG] Infs after RNN+BN layer {i+1}")

        # FC => (B, Time, n_tokens)
        x = self.fc(x)
        # print(f"[DEBUG] After FC => {x.shape}")

        # # Optional: check for NaNs
        # if torch.isnan(x).any():
        #     print("[DEBUG] NaNs in FC output")

        # log_softmax => final shape (B, Time, n_tokens)
        log_probs = F.log_softmax(x, dim=-1)
        # print(f"[DEBUG] After log_softmax => {log_probs.shape}")

        # Compute output lengths
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        # print(f"[DEBUG] log_probs_length => {log_probs_length}")

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        Calculate proper output lengths based on convolution operations.
        Each conv layer has stride=2 on the time dimension.
        """
        p, s = self.get_conv_num()
        # print(f"[DEBUG] transform_input_lengths => p={p}, s={s}, input_lengths={input_lengths}")

        for i in range(len(s)):
            # Each conv: new_time = floor((old_time + p[i]) / s[i]) + 1
            input_lengths = ((input_lengths + p[i]) // s[i]) + 1

        # print(f"[DEBUG] Final transformed lengths => {input_lengths}")
        return input_lengths

    def get_conv_num(self):
        """
        Retrieve padding and stride values from conv layers for the time dimension.
        Returns: (p, s) => (List of "temp" for each layer, List of strides)
        """
        p, s = [], []
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                # kernel_size, stride, and padding are (freq, time)
                # we only care about the time dimension => index=1
                kernel_size = layer.kernel_size[1]
                padding_time = layer.padding[1]
                stride_time = layer.stride[1]
                dilation_time = layer.dilation[1]

                # matching the formula used in forward:
                # new_time = floor((old_time + 2 * padding_time - dilation*(kernel_size-1) -1)/stride_time +1)
                temp = 2 * padding_time - dilation_time * (kernel_size - 1) - 1
                p.append(temp)
                s.append(stride_time)
        return p, s
