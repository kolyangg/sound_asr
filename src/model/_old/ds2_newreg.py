import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSpeech2ModelReg(nn.Module):
    def __init__(
        self, 
        n_feats, 
        n_tokens, 
        rnn_hidden=512, 
        rnn_layers=5, 
        bidirectional=True,
        dropout=0.5  # <--- Add dropout
    ):
        super().__init__()
        self.rnn_hidden = rnn_hidden
        self.rnn_layers_count = rnn_layers
        self.bidirectional = bidirectional
        self.n_tokens = n_tokens
        self.n_feats = n_feats
        self.dropout_prob = dropout

        # Convolutional layers (unchanged)
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

        self.rnn_input_size = self.compute_rnn_input_size()

        # Create RNN + BN + Dropout for each layer
        self.rnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()  # One dropout layer per RNN layer

        for i in range(self.rnn_layers_count):
            if i == 0:
                input_size = self.rnn_input_size
            else:
                input_size = rnn_hidden * (2 if bidirectional else 1)

            rnn = nn.GRU(
                input_size=input_size,
                hidden_size=self.rnn_hidden,
                num_layers=1,
                bidirectional=self.bidirectional,
                batch_first=True
            )
            self.rnn_layers.append(rnn)

            bn = nn.BatchNorm1d(self.rnn_hidden * (2 if bidirectional else 1))
            self.batch_norms.append(bn)

            do = nn.Dropout(p=self.dropout_prob)
            self.dropouts.append(do)

        # Fully connected output layer
        self.fc = nn.Linear(self.rnn_hidden * (2 if bidirectional else 1), self.n_tokens, bias=False)

        # Initialize weights
        self.apply(self.initialize_weights)

    def compute_rnn_input_size(self):
        initial_freq_bins = self.n_feats
        freq = math.floor((initial_freq_bins + 2 * 20 - 41) / 2 + 1)
        freq = math.floor((freq + 2 * 10 - 21) / 2 + 1)
        freq = math.floor((freq + 2 * 10 - 21) / 2 + 1)
        return freq * 96

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
        MIN_TIME = 16
        if spectrogram.size(2) < MIN_TIME:
            pad_amount = MIN_TIME - spectrogram.size(2)
            spectrogram = F.pad(spectrogram, (0, pad_amount))
            spectrogram_length = spectrogram_length + pad_amount

        x = spectrogram.unsqueeze(1)
        x = self.conv_layers(x)  # (B, C, Freq, Time)
        B, C, Freq, Time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(B, Time, C * Freq)

        # Pass through RNN + BN + Dropout for each layer
        for (rnn, bn, do) in zip(self.rnn_layers, self.batch_norms, self.dropouts):
            x, _ = rnn(x)
            # Apply Dropout
            x = do(x)
            # Apply BatchNorm -> (B, T, C) => (B, C, T)
            x = x.permute(0, 2, 1)
            x = bn(x)
            x = x.permute(0, 2, 1)

        x = self.fc(x)  # (B, T, n_tokens)
        log_probs = F.log_softmax(x, dim=-1)

        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        p, s = self.get_conv_num()
        for i in range(len(s)):
            input_lengths = ((input_lengths + p[i]) // s[i]) + 1
        return input_lengths

    def get_conv_num(self):
        p, s = [], []
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                padding_time = layer.padding[1]
                stride_time = layer.stride[1]
                temp = 2 * padding_time - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1
                p.append(temp)
                s.append(stride_time)
        return p, s
