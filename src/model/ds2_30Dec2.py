import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSpeech2ModelSimple2(nn.Module):
    def __init__(self, n_feats, n_tokens):
        """
        Args:
            n_feats (int): Number of input frequency features.
            n_tokens (int): Number of output tokens (vocabulary size).
        """
        super(DeepSpeech2ModelSimple2, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 64, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(64),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(64, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(96),
            nn.Hardtanh(0, 20, inplace=True)
        )

        self.rnn = nn.GRU(
            input_size=self.compute_rnn_input_size(n_feats),
            hidden_size=768,
            num_layers=4,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(1536, n_tokens)

        self.__str__()

    def compute_rnn_input_size(self, n_feats):
        """
        Compute the input size to the RNN layers after the convolutional layers.
        Args:
            n_feats (int): Number of frequency features in the input spectrogram.
        Returns:
            int: Flattened feature size after convolutions.
        """
        freq = (n_feats + 2 * 20 - 41) // 2 + 1  # After first Conv2D
        freq = (freq + 2 * 10 - 21) // 2 + 1     # After second Conv2D
        freq = (freq + 2 * 10 - 21) // 2 + 1     # After third Conv2D
        return freq * 96

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Forward pass of the model.

        Args:
            spectrogram (Tensor): Input spectrogram of shape (B, F, T).
            spectrogram_length (Tensor): Original spectrogram lengths.

        Returns:
            dict: Contains "log_probs" and "log_probs_length".
        """
        MIN_TIME = 16  # Minimum required time dimension after convolution

        # Pad the input if necessary
        if spectrogram.size(2) < MIN_TIME:
            pad_amount = MIN_TIME - spectrogram.size(2)
            spectrogram = F.pad(spectrogram, (0, pad_amount))  # Pad the time dimension
            spectrogram_length = spectrogram_length + pad_amount

        # Input shape: (B, F, T) -> (B, 1, F, T)
        x = spectrogram.unsqueeze(1)

        # Pass through convolutional layers
        x = self.conv(x)
        B, C, Freq, Time = x.size()

        # Reshape for RNN input: (B, T, C * Freq)
        x = x.permute(0, 3, 1, 2).contiguous().view(B, Time, C * Freq)

        # Pass through RNN
        x, _ = self.rnn(x)  # Shape: (B, T, 1536)

        # Fully connected layer
        x = self.fc(x)  # Shape: (B, T, n_tokens)

        # Apply log_softmax
        log_probs = F.log_softmax(x, dim=-1)

        # Calculate output lengths
        log_probs_length = self.transform_input_lengths(spectrogram_length)

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        Calculate proper output lengths based on convolution operations.

        Args:
            input_lengths (Tensor): Original input lengths.

        Returns:
            Tensor: Adjusted output lengths after convolution.
        """
        input_lengths = (input_lengths + 2 * 20 - 41) // 2 + 1  # After first Conv2D
        input_lengths = (input_lengths + 2 * 10 - 21) // 2 + 1  # After second Conv2D
        input_lengths = (input_lengths + 2 * 10 - 21) // 2 + 1  # After third Conv2D
        return input_lengths

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
