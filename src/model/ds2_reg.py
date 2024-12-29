import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSpeech2ModelReg(nn.Module):
    def __init__(self, n_feats, n_tokens, rnn_hidden=512, rnn_layers=5, bidirectional=True, dropout_prob=0.2):
        super(DeepSpeech2ModelReg, self).__init__()
        self.rnn_hidden = rnn_hidden
        self.rnn_layers_count = rnn_layers
        self.bidirectional = bidirectional
        self.n_tokens = n_tokens
        self.n_feats = n_feats
        self.dropout_prob = dropout_prob

        # Define convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(41, 11),
                      stride=(2, 1), padding=(20, 5)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_prob),  # Add dropout after ReLU
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(21, 11),
                      stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_prob),  # Add dropout after ReLU
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(21, 11),
                      stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_prob)  # Add dropout after ReLU
        )

        # Compute RNN input size after convolutional layers
        self.rnn_input_size = self.compute_rnn_input_size()

        # Initialize RNN layers and BatchNorm layers
        self.rnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()  # Add dropout for RNN outputs
        for i in range(self.rnn_layers_count):
            input_size = self.rnn_input_size if i == 0 else self.rnn_hidden * (2 if self.bidirectional else 1)
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
            self.dropouts.append(nn.Dropout(p=self.dropout_prob))  # Add dropout

        # Fully connected layer
        self.fc = nn.Linear(self.rnn_hidden * (2 if self.bidirectional else 1), self.n_tokens, bias=False)

        # Initialize weights
        self.apply(self.initialize_weights)

    def forward(self, spectrogram, spectrogram_length, **batch):
        MIN_TIME = 16

        # Pad the input if necessary
        if spectrogram.size(2) < MIN_TIME:
            pad_amount = MIN_TIME - spectrogram.size(2)
            spectrogram = F.pad(spectrogram, (0, pad_amount))
            spectrogram_length = spectrogram_length + pad_amount

        x = spectrogram.unsqueeze(1)

        # Pass through convolutional layers
        x = self.conv_layers(x)
        B, C, Freq, Time = x.size()

        # Reshape for RNN input
        x = x.permute(0, 3, 1, 2).contiguous().view(B, Time, C * Freq)

        # Pass through RNN layers with BatchNorm and Dropout
        for idx, (rnn, bn, dropout) in enumerate(zip(self.rnn_layers, self.batch_norms, self.dropouts)):
            x, _ = rnn(x)
            x = x.permute(0, 2, 1)
            x = bn(x)
            x = x.permute(0, 2, 1)
            x = dropout(x)  # Apply dropout to RNN output

        # Fully connected layer
        x = self.fc(x)

        # Apply log_softmax
        log_probs = F.log_softmax(x, dim=-1)

        # Calculate output lengths
        log_probs_length = self.transform_input_lengths(spectrogram_length)

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

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
