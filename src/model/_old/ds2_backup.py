import torch
from torch import nn
from torch.nn import Sequential

# class DeepSpeech2Model(nn.Module):
#     """
#     Deep Speech 2 Model Architecture (simplified version).
    
#     This implementation follows the general structure described in the 
#     DeepSpeech2 paper, using:
#     - 2D Convolutional front-end layers with batch normalization and ReLU,
#     - Multiple bidirectional RNN (GRU) layers with batch normalization,
#     - (Optional) Lookahead convolution for unidirectional models (not implemented here),
#     - Fully connected layer to output character probabilities,
#     - Log-Softmax output for CTC loss.

#     Note:
#     - This is a simplified version focused on structure.
#     - The exact hyperparameters (number of RNN layers, hidden units, etc.)
#       should be tuned as required.
#     - The transform_input_lengths function is implemented based on 
#       the strides in the convolutional layers.
#     """

#     def __init__(
#         self,
#         n_feats: int,
#         n_tokens: int,
#         rnn_hidden: int = 512,
#         rnn_layers: int = 5,
#         bidirectional: bool = True,
#     ):
#         """
#         Args:
#             n_feats (int): Number of input features (frequency dimension).
#             n_tokens (int): Number of output tokens (vocabulary size).
#             rnn_hidden (int): Number of hidden units in each RNN layer.
#             rnn_layers (int): Number of RNN layers.
#             bidirectional (bool): Whether to use a bidirectional RNN.
#         """
#         super().__init__()

#         # According to DS2 paper (example configuration):
#         # Convolutional layers (2D):
#         # Layer 1: Conv2d with filter_size=(41x11), stride=(2x2), out_channels=32
#         # Layer 2: Conv2d with filter_size=(21x11), stride=(2x1), out_channels=32
#         # Layer 3: Conv2d with filter_size=(21x11), stride=(2x1), out_channels=96
        
#         # For simplicity, we may choose smaller kernel sizes due to memory constraints,
#         # but here we keep them as-is. We must ensure (Frequency, Time) mapping to (H, W).
#         # We'll treat frequency as height (H) and time as width (W).
        
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=1,
#                 out_channels=32,
#                 kernel_size=(41, 11),
#                 stride=(2, 2),  # reduces frequency and time
#                 padding=(20, 5)  # same-padding style for large kernels
#             ),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(
#                 in_channels=32,
#                 out_channels=32,
#                 kernel_size=(21, 11),
#                 stride=(2, 1),  # reduces frequency only
#                 padding=(10, 5)
#             ),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(
#                 in_channels=32,
#                 out_channels=96,
#                 kernel_size=(21, 11),
#                 stride=(2, 1),  # reduces frequency only
#                 padding=(10, 5)
#             ),
#             nn.BatchNorm2d(96),
#             nn.ReLU(inplace=True),
#         )

#         # After convolutions, we have output shape roughly:
#         # B x 96 x (Freq_out) x (Time_out)
#         #
#         # Time dimension is reduced by a factor of 2 from the first conv layer.
#         # The second and third layers have stride (2,1), meaning they reduce only frequency.
#         # So final time dimension is approximately T/2.
        
#         # Next, we flatten frequency dimension and treat it as feature dimension for RNN:
#         # RNN input_size = 96 * (reduced frequency dimension)
#         # However, frequency dimension is also reduced by stride:
#         # Initial freq dimension = n_feats
#         # After 1st conv: freq_out ~ n_feats / 2
#         # After 2nd conv: freq_out ~ (n_feats/2) / 2 = n_feats/4
#         # After 3rd conv: freq_out ~ (n_feats/4) / 2 = n_feats/8
#         #
#         # So the RNN input_size = 96 * (n_feats/8) approximately.
#         # Because of padding, exact dimension depends on input size. For training,
#         # we usually fix input sizes. Here we assume large padding ensures stable dimension.
#         # A more flexible approach: infer RNN input size after a dummy forward pass.
        
#         # We will initialize a dummy forward pass to infer the RNN input size:
#         dummy_input = torch.zeros(1, 1, n_feats, 100)  # (B, C, F, T)
#         with torch.no_grad():
#             dummy_out = self.conv_layers(dummy_input)
#         conv_out_channels = dummy_out.size(1)
#         conv_out_freq = dummy_out.size(2)
#         # Time dimension after conv: dummy_out.size(3) ~ 100/2 = 50
#         rnn_input_size = conv_out_channels * conv_out_freq

#         # RNN layers (Bidirectional GRU)
#         # Apply batch normalization between RNN layers: DS2 does sequence-wise normalization.
#         # We'll implement a wrapper RNN + BatchNorm block for each layer.
        
#         self.rnn_layers = nn.ModuleList()
#         for i in range(rnn_layers):
#             input_size = rnn_input_size if i == 0 else rnn_hidden * (2 if bidirectional else 1)
#             rnn = nn.GRU(
#                 input_size=input_size,
#                 hidden_size=rnn_hidden,
#                 num_layers=1,
#                 batch_first=True,
#                 bidirectional=bidirectional
#             )
#             # Sequence-wise batch normalization (applied on output features of RNN)
#             bn = nn.BatchNorm1d(rnn_hidden * (2 if bidirectional else 1))
#             self.rnn_layers.append(nn.ModuleDict({
#                 'rnn': rnn,
#                 'bn': bn
#             }))

#         # Lookahead Convolution Layer (only needed for unidirectional):
#         # Not implemented here. If unidirectional, we could add a conv1D layer on the time dimension.
#         # For simplicity, skip this step or comment it out.
#         # self.lookahead_conv = ...

#         # Fully connected layer to map to character classes
#         self.fc = nn.Linear(rnn_hidden * (2 if bidirectional else 1), n_tokens)


#     def forward(self, spectrogram, spectrogram_length, **batch):
#         """
#         Forward pass of the Deep Speech 2 model.

#         Args:
#             spectrogram (Tensor): input spectrogram of shape (B, T, n_feats).
#             spectrogram_length (Tensor): lengths of each input sequence before padding.

#         Returns:
#             output (dict): A dictionary with:
#                 "log_probs": Log probabilities (B, T', n_tokens)
#                 "log_probs_length": Lengths of the output time steps after transforms
#         """

#         # DS2 expects input as (B, C=1, Freq, Time)
#         # The input given: (B, T, n_feats)
#         # Transpose to (B, n_feats, T) then unsqueeze channel dim: (B, 1, n_feats, T)
#         x = spectrogram.transpose(1, 2).unsqueeze(1)  # (B, 1, F, T)

#         # Pass through convolutional layers
#         x = self.conv_layers(x)  # (B, C_out=96, F_out, T_out)

#         # Now we transpose to (B, T_out, Feature_dim) for RNN:
#         # Feature_dim = C_out * F_out
#         B, C, Freq, Time = x.size()
#         x = x.permute(0, 3, 1, 2)  # (B, T_out, C_out, F_out)
#         x = x.contiguous().view(B, Time, C * Freq)  # (B, T_out, C_out*F_out)

#         # Pass through RNN layers
#         # DS2 applies batch norm sequence-wise (on the features dimension)
#         for layer in self.rnn_layers:
#             # rnn step
#             x, _ = layer['rnn'](x)  # (B, T_out, hidden*dirs)
#             # sequence-wise batch norm: apply on (B, Feature, T_out) for BN
#             x = x.transpose(1, 2)  # (B, Features, T)
#             x = layer['bn'](x)
#             x = x.transpose(1, 2)  # (B, T, Features)

#         # Lookahead convolution (if unidirectional) - skipped here

#         # Final fully connected layer
#         x = self.fc(x)  # (B, T, n_tokens)

#         log_probs = nn.functional.log_softmax(x, dim=-1)

#         # Transform lengths
#         log_probs_length = self.transform_input_lengths(spectrogram_length)

#         return {"log_probs": log_probs, "log_probs_length": log_probs_length}

#     def transform_input_lengths(self, input_lengths):
#         """
#         Given the input lengths (time steps), transform them after the convolutional layers.
        
#         Due to the first conv layer's stride in the time dimension (which is 2), 
#         the time dimension is halved. The subsequent conv layers do not reduce 
#         the time dimension further (they have stride=2 only in frequency dimension).

#         So, final time_length = ceil(input_length / 2).

#         Note: If padding is used in convolution, exact formula might differ slightly, 
#         but typically we assume a halving. For safety, we can use integer division.
#         """
#         # time dimension reduced by factor of 2 at first conv layer:
#         output_lengths = (input_lengths + 1) // 2
#         return output_lengths

#     def __str__(self):
#         """
#         Print the model with parameters information.
#         """
#         all_parameters = sum([p.numel() for p in self.parameters()])
#         trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])

#         result_info = super().__str__()
#         result_info += f"\nAll parameters: {all_parameters}"
#         result_info += f"\nTrainable parameters: {trainable_parameters}"

#         return result_info



# class DeepSpeech2Model(nn.Module):
#     def __init__(self, n_feats, n_tokens, rnn_hidden=512, rnn_layers=5, bidirectional=True):
#         super().__init__()
#         # Define conv_layers as before
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 32, (41, 11), stride=(2, 2), padding=(20, 5)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, (21, 11), stride=(2, 1), padding=(10, 5)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 96, (21, 11), stride=(2, 1), padding=(10, 5)),
#             nn.BatchNorm2d(96),
#             nn.ReLU(inplace=True),
#         )

#         self.rnn_layers = nn.ModuleList()
#         self.bidirectional = bidirectional
#         self.rnn_hidden = rnn_hidden
#         self.rnn_layers_count = rnn_layers
#         self.n_tokens = n_tokens

#         # We'll create RNN layers after we know the final feature dimension
#         self.rnn_built = False
#         self.fc = None

#     def _build_rnn_layers(self, input_size):
#         for i in range(self.rnn_layers_count):
#             rnn_input_size = input_size if i == 0 else self.rnn_hidden * (2 if self.bidirectional else 1)
#             rnn = nn.GRU(
#                 input_size=rnn_input_size,
#                 hidden_size=self.rnn_hidden,
#                 num_layers=1,
#                 batch_first=True,
#                 bidirectional=self.bidirectional
#             )
#             bn = nn.BatchNorm1d(self.rnn_hidden * (2 if self.bidirectional else 1))
#             self.rnn_layers.append(nn.ModuleDict({'rnn': rnn, 'bn': bn}))
#         self.fc = nn.Linear(self.rnn_hidden * (2 if self.bidirectional else 1), self.n_tokens)
#         self.rnn_built = True

#     def forward(self, spectrogram, spectrogram_length, **batch):
#         x = spectrogram.transpose(1, 2).unsqueeze(1)  # (B, 1, F, T)
#         x = self.conv_layers(x)  # (B, C, F_out, T_out)

#         B, C, Freq, Time = x.size()
#         x = x.permute(0, 3, 1, 2).contiguous()  # (B, T_out, C, Freq)
#         x = x.view(B, Time, C * Freq)           # (B, T_out, C*Freq)

#         if not self.rnn_built:
#             rnn_input_size = C * Freq
#             self._build_rnn_layers(rnn_input_size)
#             # Move to the device of x
#             device = x.device
#             for layer in self.rnn_layers:
#                 for submodule in layer.values():
#                     submodule.to(device)
#             self.fc.to(device)
#             self.rnn_built = True

#         # Now layers are on the correct device
#         for layer in self.rnn_layers:
#             x, _ = layer['rnn'](x)
#             x = x.transpose(1, 2)
#             x = layer['bn'](x)
#             x = x.transpose(1, 2)

#         x = self.fc(x)
#         log_probs = nn.functional.log_softmax(x, dim=-1)
#         log_probs_length = self.transform_input_lengths(spectrogram_length)
#         return {"log_probs": log_probs, "log_probs_length": log_probs_length}


#     def transform_input_lengths(self, input_lengths):
#         """
#         Adjust input lengths based on convolutional downsampling.

#         Args:
#             input_lengths (Tensor): Original spectrogram lengths.

#         Returns:
#             Tensor: Adjusted lengths after convolutions.
#         """
#         # Conv1: stride (2, 2), reduces time by 2
#         # Conv2, Conv3: stride (2, 1), reduces freq by 2 each
#         return ((input_lengths - 1) // 2) + 1


# import torch
# from torch import nn


# class DeepSpeech2Model(nn.Module):
#     """
#     Deep Speech 2 Model Architecture.
#     """

#     def __init__(
#         self, n_feats: int, n_tokens: int, rnn_hidden: int = 512, rnn_layers: int = 5, bidirectional: bool = True
#     ):
#         """
#         Args:
#             n_feats (int): Number of input features (frequency dimension).
#             n_tokens (int): Number of output tokens (vocabulary size).
#             rnn_hidden (int): Number of hidden units in each RNN layer.
#             rnn_layers (int): Number of RNN layers.
#             bidirectional (bool): Use bidirectional RNNs.
#         """
#         super().__init__()

#         # 2D Convolutional Layers
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),  # Reduces freq and time
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),  # Reduces freq only
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),  # Reduces freq only
#             nn.BatchNorm2d(96),
#             nn.ReLU(inplace=True),
#         )

#         # Bi-directional GRU layers
#         self.rnn_layers = nn.ModuleList([
#             nn.GRU(
#                 input_size=96 * (n_feats // 8),  # Reduced frequency dimension after conv layers
#                 hidden_size=rnn_hidden,
#                 num_layers=1,
#                 batch_first=True,
#                 bidirectional=bidirectional,
#             )
#             for _ in range(rnn_layers)
#         ])

#         # BatchNorm after RNN
#         self.batch_norms = nn.ModuleList([
#             nn.BatchNorm1d(rnn_hidden * (2 if bidirectional else 1))
#             for _ in range(rnn_layers)
#         ])

#         # Fully connected layer
#         self.fc = nn.Linear(rnn_hidden * (2 if bidirectional else 1), n_tokens)

#     def forward(self, spectrogram, spectrogram_length, **batch):
#         """
#         Forward pass of DeepSpeech2.

#         Args:
#             spectrogram (Tensor): Input spectrogram of shape (B, T, n_feats).
#             spectrogram_length (Tensor): Original spectrogram lengths.

#         Returns:
#             dict: Contains "log_probs" and "log_probs_length".
#         """
#         # Reshape spectrogram: (B, 1, n_feats, T)
#         x = spectrogram.transpose(1, 2).unsqueeze(1)

#         # Pass through Conv layers
#         x = self.conv_layers(x)  # (B, C, F', T')
#         batch_size, channels, freq, time = x.size()

#         # Reduce input lengths
#         log_probs_length = self.transform_input_lengths(spectrogram_length)

#         # Reshape to (B, T', C*F') for RNN input
#         x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, time, channels * freq)

#         # Pass through RNN layers with BatchNorm
#         for rnn, bn in zip(self.rnn_layers, self.batch_norms):
#             x, _ = rnn(x)
#             x = bn(x.transpose(1, 2)).transpose(1, 2)  # BatchNorm on features

#         # Fully connected layer to predict character probabilities
#         x = self.fc(x)  # (B, T', n_tokens)

#         # Transpose to (T', B, n_tokens) for CTC Loss
#         log_probs = nn.functional.log_softmax(x, dim=-1).permute(1, 0, 2)

#         return {"log_probs": log_probs, "log_probs_length": log_probs_length}

#     def transform_input_lengths(self, input_lengths):
#         """
#         Adjust input lengths based on convolutional downsampling.

#         Args:
#             input_lengths (Tensor): Original spectrogram lengths.

#         Returns:
#             Tensor: Adjusted lengths after convolutions.
#         """
#         # Conv1: stride (2, 2), reduces time by 2
#         # Conv2, Conv3: stride (2, 1), reduces freq by 2 each
#         return ((input_lengths - 1) // 2) + 1

#     def __str__(self):
#         """
#         Model prints with the number of parameters.
#         """
#         all_parameters = sum([p.numel() for p in self.parameters()])
#         trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])

#         result_info = super().__str__()
#         result_info += f"\nAll parameters: {all_parameters}"
#         result_info += f"\nTrainable parameters: {trainable_parameters}"

#         return result_info




import torch
from torch import nn


# class DeepSpeech2Model(nn.Module):
#     def __init__(self, n_feats, n_tokens, rnn_hidden=512, rnn_layers=5, bidirectional=True):
#         super().__init__()
#         self.rnn_hidden = rnn_hidden
#         self.rnn_layers_count = rnn_layers
#         self.bidirectional = bidirectional
#         self.n_tokens = n_tokens
#         self.n_feats = n_feats

#         # Convolutional layers
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
#             nn.BatchNorm2d(96),
#             nn.ReLU(inplace=True),
#         )

#         # RNN layers and FC will be lazily initialized
#         self.rnn_layers = None
#         self.fc = None

    
# #     def forward(self, spectrogram, spectrogram_length, **batch):
# #         """
# #         Forward pass of DeepSpeech2.

# #         Args:
# #             spectrogram (Tensor): Input spectrogram of shape (B, T, n_feats).
# #             spectrogram_length (Tensor): Original spectrogram lengths.

# #         Returns:
# #             dict: Contains "log_probs" and "log_probs_length".
# #         """
# #         # Ensure input shape: (B, T, n_feats) -> (B, 1, n_feats, T)
# #         x = spectrogram.unsqueeze(1).transpose(2, 3)

# #         # Pass through convolutional layers
# #         x = self.conv_layers(x)  # Shape: (B, C, F', T')

# #         B, C, Freq, Time = x.size()

# #         # Flatten for RNN input: (B, T', C*F')
# #         x = x.permute(0, 3, 1, 2).contiguous().view(B, Time, C * Freq)

# #         # Dynamically initialize RNN layers and FC on first forward pass
# #         if self.rnn_layers is None:
# #             input_size = C * Freq
# #             self.rnn_layers = nn.ModuleList([
# #                 nn.GRU(input_size=input_size if i == 0 else self.rnn_hidden * (2 if self.bidirectional else 1),
# #                     hidden_size=self.rnn_hidden, num_layers=1, batch_first=True, bidirectional=self.bidirectional)
# #                 for i in range(self.rnn_layers_count)
# #             ])
# #             self.batch_norms = nn.ModuleList([
# #                 nn.BatchNorm1d(self.rnn_hidden * (2 if self.bidirectional else 1))
# #                 for _ in range(self.rnn_layers_count)
# #             ])
# #             self.fc = nn.Linear(self.rnn_hidden * (2 if self.bidirectional else 1), self.n_tokens)
# #             self.to(x.device)  # Ensure newly initialized layers are on correct device

# #         # Pass through RNN layers with BatchNorm
# #         for rnn, bn in zip(self.rnn_layers, self.batch_norms):
# #             x, _ = rnn(x)
# #             x = bn(x.transpose(1, 2)).transpose(1, 2)

# #         # Fully connected layer for character probabilities
# #         x = self.fc(x)  # Shape: (B, T', n_tokens)

# #         # Transpose for CTC Loss: (T', B, C)
# #         log_probs = nn.functional.log_softmax(x, dim=-1).permute(1, 0, 2)

# #         # Adjust input lengths
# #         log_probs_length = self.transform_input_lengths(spectrogram_length)

# #         return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    

# #     def transform_input_lengths(self, input_lengths):
# #         """
# #         Adjust input lengths based on convolutional downsampling.

# #         Args:
# #             input_lengths (Tensor): Original spectrogram lengths.

# #         Returns:
# #             Tensor: Adjusted lengths after convolutions.
# #         """
# #         # Total downsampling factor = 8 (Conv1: stride=2, Conv2: stride=2, Conv3: stride=2)
# #         reduced_lengths = ((input_lengths - 1) // 8) + 1

# #         # Ensure correct type and device
# #         return reduced_lengths.to(input_lengths.device).long()



#     def transform_input_lengths(self, input_lengths):
#         """
#         Adjust input lengths based on convolutional downsampling.
#         Also ensures lengths don't exceed the actual sequence length.
#         """
#         lengths = input_lengths.clone()
        
#         # Follow convolution arithmetic
#         # Conv1: stride (2, 2)
#         lengths = ((lengths + 2*20 - 41) // 2) + 1
#         # Conv2: stride (2, 1)
#         lengths = ((lengths + 2*10 - 21) // 2) + 1
#         # Conv3: stride (2, 1)
#         lengths = ((lengths + 2*10 - 21) // 2) + 1
        
#         # Convert to proper type
#         lengths = lengths.to(torch.int64)
        
#         # Ensure lengths are not too large
#         T = 64  # This is the actual sequence length we see in log_probs
#         lengths = torch.clamp(lengths, max=T)
        
#         return lengths

#     def forward(self, spectrogram, spectrogram_length, **batch):
#         """
#         Forward pass of DeepSpeech2.
#         """
#         batch_size = spectrogram.size(0)
        
#         # Ensure input shape: (B, T, n_feats) -> (B, 1, n_feats, T)
#         x = spectrogram.unsqueeze(1).transpose(2, 3)

#         # Pass through convolutional layers
#         x = self.conv_layers(x)
#         B, C, Freq, Time = x.size()

#         # Flatten for RNN input: (B, T', C*F')
#         x = x.permute(0, 3, 1, 2).contiguous().view(B, Time, C * Freq)

#         # Initialize RNN layers if needed
#         if self.rnn_layers is None:
#             input_size = C * Freq
#             self.rnn_layers = nn.ModuleList([
#                 nn.GRU(input_size=input_size if i == 0 else self.rnn_hidden * (2 if self.bidirectional else 1),
#                     hidden_size=self.rnn_hidden, num_layers=1, batch_first=True, bidirectional=self.bidirectional)
#                 for i in range(self.rnn_layers_count)
#             ])
#             self.batch_norms = nn.ModuleList([
#                 nn.BatchNorm1d(self.rnn_hidden * (2 if self.bidirectional else 1))
#                 for _ in range(self.rnn_layers_count)
#             ])
#             self.fc = nn.Linear(self.rnn_hidden * (2 if self.bidirectional else 1), self.n_tokens)
#             self.to(x.device)

#         # Pass through RNN layers with BatchNorm
#         for rnn, bn in zip(self.rnn_layers, self.batch_norms):
#             x, _ = rnn(x)
#             x = bn(x.transpose(1, 2)).transpose(1, 2)

#         # Fully connected layer for character probabilities
#         x = self.fc(x)  # Shape: (B, T', n_tokens)

#         # Apply log_softmax and prepare for CTC
# #         log_probs = nn.functional.log_softmax(x, dim=-1)
# #         log_probs = log_probs.transpose(0, 1)  # (T, B, C)
        
# #         # Get output lengths
# #         log_probs_length = self.transform_input_lengths(spectrogram_length)

# #         print(f"Final log_probs shape: {log_probs.shape}")
# #         print(f"Output lengths: {log_probs_length}")

# #         return {
# #             "log_probs": log_probs,
# #             "log_probs_length": log_probs_length
# #         }



# class DeepSpeech2Model(nn.Module):
#     def __init__(self, n_feats, n_tokens, rnn_hidden=512, rnn_layers=5, bidirectional=True):
#         super().__init__()
#         self.rnn_hidden = rnn_hidden
#         self.rnn_layers_count = rnn_layers
#         self.bidirectional = bidirectional
#         self.n_tokens = n_tokens  # including blank
#         self.n_feats = n_feats

#         # Convolutional layers
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
#             nn.BatchNorm2d(96),
#             nn.ReLU(inplace=True),
#         )

#         # RNN layers and FC will be lazily initialized
#         self.rnn_layers = None
#         self.fc = None

#     def forward(self, spectrogram, spectrogram_length, **batch):
#         """
#         Forward pass of DeepSpeech2.
        
#         Args:
#             spectrogram (Tensor): Input spectrogram of shape (B, F, T)
#                 where F is n_feats and T is sequence length
#             spectrogram_length (Tensor): Original lengths of shape (B,)
            
#         Returns:
#             dict: Contains:
#                 - log_probs: Tensor of shape (T, B, C)
#                 - log_probs_length: Tensor of shape (B,)
#         """
        
#         print("\nModel Forward Pass Debug:")
#         print(f"Input spectrogram shape: {spectrogram.shape}")
#         print(f"Input spectrogram_length: {spectrogram_length}")
#         print(f"Input spectrogram_length shape: {spectrogram_length.shape}")

            
        
#         # Shape check on input
#         batch_size, n_feats, max_input_length = spectrogram.shape
#         assert n_feats == self.n_feats, f"Expected {self.n_feats} features, got {n_feats}"
        
#         # Add channel dim: (B, F, T) -> (B, 1, F, T)
#         x = spectrogram.unsqueeze(1)
        
#         # Apply CNN layers
#         x = self.conv_layers(x)  # Shape: (B, C, F', T')
#         print(f"After CNN shape: {x.shape}")
        
        
#         B, C, Freq, Time = x.size()
        
        
#         # Prepare for RNN: (B, C, F', T') -> (B, T', C*F')
#         x = x.permute(0, 3, 1, 2).contiguous().view(B, Time, C * Freq)
#         print(f"After reshape for RNN shape: {x.shape}")

        
#         # Initialize RNN and FC if needed
#         if self.rnn_layers is None:
#             input_size = C * Freq
#             self.rnn_layers = nn.ModuleList([
#                 nn.GRU(
#                     input_size=input_size if i == 0 else self.rnn_hidden * (2 if self.bidirectional else 1),
#                     hidden_size=self.rnn_hidden,
#                     num_layers=1,
#                     batch_first=True,
#                     bidirectional=self.bidirectional
#                 )
#                 for i in range(self.rnn_layers_count)
#             ])
#             self.batch_norms = nn.ModuleList([
#                 nn.BatchNorm1d(self.rnn_hidden * (2 if self.bidirectional else 1))
#                 for _ in range(self.rnn_layers_count)
#             ])
#             self.fc = nn.Linear(self.rnn_hidden * (2 if self.bidirectional else 1), self.n_tokens)
#             self.to(x.device)
        
#         # RNN layers
#         for rnn, bn in zip(self.rnn_layers, self.batch_norms):
#             x, _ = rnn(x)
#             x = bn(x.transpose(1, 2)).transpose(1, 2)
        
#         # Final FC layer: (B, T', H) -> (B, T', n_tokens)
#         x = self.fc(x)
        
#         # Apply log_softmax and get (T, B, C) format for CTC
#         log_probs = nn.functional.log_softmax(x, dim=-1).transpose(0, 1)
        
#         # Calculate output lengths that are ≤ actual sequence length
#         output_lengths = self.transform_input_lengths(spectrogram_length)
#         actual_output_length = log_probs.size(0)
#         output_lengths = torch.clamp(output_lengths, max=actual_output_length)
        
#         # Debug info
#         print(f"Input shape: {spectrogram.shape}")  # (B, F, T)
#         print(f"Input lengths: {spectrogram_length}")  # (B,)
#         print(f"Final log_probs shape: {log_probs.shape}")
#         print(f"Output lengths: {output_lengths}")
        
#         return {
#             "log_probs": log_probs,         # Shape: (T, B, C)
#             "log_probs_length": output_lengths  # Shape: (B,)
#         }

#     def transform_input_lengths(self, input_lengths):
#         """Calculate output lengths after all transformations"""
#         lengths = input_lengths.clone()
        
#         # Each conv layer's effect on time dimension
#         lengths = ((lengths + 2*20 - 41) // 2) + 1  # Conv1
#         lengths = ((lengths + 2*10 - 21) // 2) + 1  # Conv2
#         lengths = ((lengths + 2*10 - 21) // 2) + 1  # Conv3
        
#         return lengths.to(torch.int64)
    
    
    
# class DeepSpeech2Model(nn.Module):
#     def __init__(self, n_feats, n_tokens, rnn_hidden=512, rnn_layers=5, bidirectional=True):
#         super().__init__()
#         self.rnn_hidden = rnn_hidden
#         self.rnn_layers_count = rnn_layers
#         self.bidirectional = bidirectional
#         self.n_tokens = n_tokens
#         self.n_feats = n_feats

#         # Convolutional layers - keeping track of stride for length calculation
#         self.conv_layers = nn.Sequential(
#             # Conv1: (41,11) kernel, (2,2) stride, (20,5) padding
#             nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             # Conv2: (21,11) kernel, (2,1) stride, (10,5) padding
#             nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             # Conv3: (21,11) kernel, (2,1) stride, (10,5) padding
#             nn.Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
#             nn.BatchNorm2d(96),
#             nn.ReLU(inplace=True),
#         )

#         self.rnn_layers = None
#         self.fc = None

#     def forward(self, spectrogram, spectrogram_length, **batch):
#         # Input shape: (B, F, T)
#         batch_size = spectrogram.shape[0]
        
#         # Add channel dimension for CNN
#         x = spectrogram.unsqueeze(1)  # (B, 1, F, T)
        
#         # Pass through CNN
#         x = self.conv_layers(x)  # (B, C, F', T')
#         B, C, Freq, Time = x.size()
        
#         # Prepare for RNN
#         x = x.permute(0, 3, 1, 2).contiguous().view(B, Time, C * Freq)
        
#         # Initialize RNN layers if needed
#         if self.rnn_layers is None:
#             input_size = C * Freq
#             self.rnn_layers = nn.ModuleList([
#                 nn.GRU(
#                     input_size=input_size if i == 0 else self.rnn_hidden * (2 if self.bidirectional else 1),
#                     hidden_size=self.rnn_hidden,
#                     num_layers=1,
#                     batch_first=True,
#                     bidirectional=self.bidirectional
#                 )
#                 for i in range(self.rnn_layers_count)
#             ])
#             self.batch_norms = nn.ModuleList([
#                 nn.BatchNorm1d(self.rnn_hidden * (2 if self.bidirectional else 1))
#                 for _ in range(self.rnn_layers_count)
#             ])
#             self.fc = nn.Linear(self.rnn_hidden * (2 if self.bidirectional else 1), self.n_tokens)
#             self.to(x.device)
        
#         # RNN layers
#         for rnn, bn in zip(self.rnn_layers, self.batch_norms):
#             x, _ = rnn(x)
#             x = bn(x.transpose(1, 2)).transpose(1, 2)
        
#         # Final FC layer and log softmax
#         x = self.fc(x)
#         log_probs = nn.functional.log_softmax(x, dim=-1).transpose(0, 1)  # (T, B, C)
        
#         # Calculate proper output lengths
#         output_lengths = self.transform_input_lengths(spectrogram_length)
#         output_lengths = torch.clamp(output_lengths, max=log_probs.size(0))
        
#         # Print debug info
#         print("\nSequence length debug:")
#         print(f"Original lengths: {spectrogram_length}")
#         print(f"Calculated lengths: {output_lengths}")
#         print(f"Actual output time dim: {log_probs.size(0)}")
        
#         return {
#             "log_probs": log_probs,
#             "log_probs_length": output_lengths
#         }

#     def transform_input_lengths(self, input_lengths):
#         """
#         More precise length calculation following CNN operations.
#         Formula for each conv layer: 
#         output_length = ((input_length + 2*padding - kernel_size) / stride) + 1
#         """
#         lengths = input_lengths.clone().to(torch.float32)

#         # Conv1: stride 2 in time dimension
#         lengths = ((lengths + 2*5 - 11) / 2) + 1
        
#         # Conv2: stride 1 in time dimension
#         lengths = ((lengths + 2*5 - 11) / 1) + 1
        
#         # Conv3: stride 1 in time dimension
#         lengths = ((lengths + 2*5 - 11) / 1) + 1
        
#         # Ensure lengths are valid
#         lengths = torch.floor(lengths)
#         lengths = torch.clamp(lengths, min=1)
        
#         return lengths.to(torch.int64)



# class DeepSpeech2Model(nn.Module):
#     def __init__(self, n_feats, n_tokens, rnn_hidden=512, rnn_layers=5, bidirectional=True):
#         super().__init__()
#         self.rnn_hidden = rnn_hidden
#         self.rnn_layers_count = rnn_layers
#         self.bidirectional = bidirectional
#         self.n_tokens = n_tokens
#         self.n_feats = n_feats  # This is the mel frequency bins (e.g., 128)

#         # Convolutional layers
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
#             nn.BatchNorm2d(96),
#             nn.ReLU(inplace=True),
#         )

#         # RNN layers and FC will be lazily initialized
#         self.rnn_layers = None
#         self.fc = None

#     def forward(self, spectrogram, spectrogram_length, **batch):
#         """
#         Forward pass of DeepSpeech2.
#         Args:
#             spectrogram (Tensor): Input spectrogram of shape (B, F, T).
#             spectrogram_length (Tensor): Original spectrogram lengths.
#         Returns:
#             dict: Contains "log_probs" and "log_probs_length".
#         """
#         # Input shape: (B, F, T) -> (B, 1, F, T)
#         x = spectrogram.unsqueeze(1)

#         # Pass through convolutional layers
#         x = self.conv_layers(x)  # Shape: (B, C, F', T')
#         B, C, Freq, Time = x.size()

#         # Flatten for RNN input: (B, T', C*F')
#         x = x.permute(0, 3, 1, 2).contiguous().view(B, Time, C * Freq)

#         # Dynamically initialize RNN layers if needed
#         if self.rnn_layers is None:
#             input_size = C * Freq
#             self.rnn_layers = nn.ModuleList([
#                 nn.GRU(input_size=input_size if i == 0 else self.rnn_hidden * (2 if self.bidirectional else 1),
#                     hidden_size=self.rnn_hidden, num_layers=1, batch_first=True, bidirectional=self.bidirectional)
#                 for i in range(self.rnn_layers_count)
#             ])
#             self.batch_norms = nn.ModuleList([
#                 nn.BatchNorm1d(self.rnn_hidden * (2 if self.bidirectional else 1))
#                 for _ in range(self.rnn_layers_count)
#             ])
#             self.fc = nn.Linear(self.rnn_hidden * (2 if self.bidirectional else 1), self.n_tokens)
#             self.to(x.device)

#         # Pass through RNN layers with BatchNorm
#         for rnn, bn in zip(self.rnn_layers, self.batch_norms):
#             x, _ = rnn(x)
#             x = bn(x.transpose(1, 2)).transpose(1, 2)

#         # Fully connected layer for character probabilities
#         x = self.fc(x)  # Shape: (B, T', n_tokens)

#         # Transpose for CTC Loss: (T', B, C)
#         log_probs = nn.functional.log_softmax(x, dim=-1).permute(1, 0, 2)

#         # Adjust input lengths
#         log_probs_length = self.transform_input_lengths(spectrogram_length)

#         return {"log_probs": log_probs, "log_probs_length": log_probs_length}

#     def transform_input_lengths(self, input_lengths):
#         """
#         Adjust input lengths based on convolutional downsampling.
#         """
#         lengths = ((input_lengths - 1) // 8) + 1
#         return lengths.long()


# class DeepSpeech2Model(nn.Module):
#     def __init__(self, n_feats, n_tokens, rnn_hidden=512, rnn_layers=5, bidirectional=True):
#         super().__init__()
#         self.rnn_hidden = rnn_hidden
#         self.rnn_layers_count = rnn_layers
#         self.bidirectional = bidirectional
#         self.n_tokens = n_tokens
#         self.n_feats = n_feats  # This is the mel frequency bins (e.g., 128)

#         # Convolutional layers
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
#             nn.BatchNorm2d(96),
#             nn.ReLU(inplace=True),
#         )

#         # RNN layers and FC will be lazily initialized
#         self.rnn_layers = None
#         self.fc = None

#     def forward(self, spectrogram, spectrogram_length, **batch):
#         """
#         Forward pass of DeepSpeech2.
#         Args:
#             spectrogram (Tensor): Input spectrogram of shape (B, F, T).
#             spectrogram_length (Tensor): Original spectrogram lengths.
#         Returns:
#             dict: Contains "log_probs" of shape (B, T, C) and "log_probs_length".
#         """
#         # Input shape: (B, F, T) -> (B, 1, F, T)
#         x = spectrogram.unsqueeze(1)

#         # Pass through convolutional layers
#         x = self.conv_layers(x)  # Shape: (B, C, F', T')
#         B, C, Freq, Time = x.size()

#         # Flatten for RNN input: (B, T', C*F')
#         x = x.permute(0, 3, 1, 2).contiguous().view(B, Time, C * Freq)

#         # Dynamically initialize RNN layers if needed
#         if self.rnn_layers is None:
#             input_size = C * Freq
#             self.rnn_layers = nn.ModuleList([
#                 nn.GRU(input_size=input_size if i == 0 else self.rnn_hidden * (2 if self.bidirectional else 1),
#                     hidden_size=self.rnn_hidden, num_layers=1, batch_first=True, bidirectional=self.bidirectional)
#                 for i in range(self.rnn_layers_count)
#             ])
#             self.batch_norms = nn.ModuleList([
#                 nn.BatchNorm1d(self.rnn_hidden * (2 if self.bidirectional else 1))
#                 for _ in range(self.rnn_layers_count)
#             ])
#             self.fc = nn.Linear(self.rnn_hidden * (2 if self.bidirectional else 1), self.n_tokens)
#             self.to(x.device)

#         # Pass through RNN layers with BatchNorm
#         for rnn, bn in zip(self.rnn_layers, self.batch_norms):
#             x, _ = rnn(x)
#             x = bn(x.transpose(1, 2)).transpose(1, 2)

#         # Fully connected layer for character probabilities
#         x = self.fc(x)  # Shape: (B, T', n_tokens)

#         # Apply log_softmax - keep in (B, T, C) format
#         log_probs = nn.functional.log_softmax(x, dim=-1)

#         # Calculate output lengths
#         log_probs_length = spectrogram_length  # Keep original lengths

#         return {"log_probs": log_probs, "log_probs_length": log_probs_length}

#     def transform_input_lengths(self, input_lengths):
#         """
#         For compatibility, but we're now using original lengths
#         """
#         return input_lengths.long()
    
    
    

# class DeepSpeech2Model(nn.Module):
#     def __init__(self, n_feats, n_tokens, rnn_hidden=512, rnn_layers=5, bidirectional=True):
#         super().__init__()
#         self.rnn_hidden = rnn_hidden
#         self.rnn_layers_count = rnn_layers
#         self.bidirectional = bidirectional
#         self.n_tokens = n_tokens
#         self.n_feats = n_feats

#         # Convolutional layers
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
#             nn.BatchNorm2d(96),
#             nn.ReLU(inplace=True),
#         )

#         # RNN layers and FC will be lazily initialized
#         self.rnn_layers = None
#         self.fc = None

#     def forward(self, spectrogram, spectrogram_length, **batch):
#         """
#         Forward pass of DeepSpeech2.
#         Args:
#             spectrogram (Tensor): Input spectrogram of shape (B, F, T).
#             spectrogram_length (Tensor): Original spectrogram lengths.
#         Returns:
#             dict: Contains "log_probs" and "log_probs_length".
#         """
#         # Input shape: (B, F, T) -> (B, 1, F, T)
#         x = spectrogram.unsqueeze(1)

#         # Pass through convolutional layers
#         x = self.conv_layers(x)  # Shape: (B, C, F', T')
#         B, C, Freq, Time = x.size()

#         # Flatten for RNN input: (B, T', C*F')
#         x = x.permute(0, 3, 1, 2).contiguous().view(B, Time, C * Freq)

#         # Dynamically initialize RNN layers if needed
#         if self.rnn_layers is None:
#             input_size = C * Freq
#             self.rnn_layers = nn.ModuleList([
#                 nn.GRU(input_size=input_size if i == 0 else self.rnn_hidden * (2 if self.bidirectional else 1),
#                     hidden_size=self.rnn_hidden, num_layers=1, batch_first=True, bidirectional=self.bidirectional)
#                 for i in range(self.rnn_layers_count)
#             ])
#             self.batch_norms = nn.ModuleList([
#                 nn.BatchNorm1d(self.rnn_hidden * (2 if self.bidirectional else 1))
#                 for _ in range(self.rnn_layers_count)
#             ])
#             self.fc = nn.Linear(self.rnn_hidden * (2 if self.bidirectional else 1), self.n_tokens)
#             self.to(x.device)

#         # Pass through RNN layers with BatchNorm
#         for rnn, bn in zip(self.rnn_layers, self.batch_norms):
#             x, _ = rnn(x)
#             x = bn(x.transpose(1, 2)).transpose(1, 2)

#         # Fully connected layer for character probabilities
#         x = self.fc(x)  # Shape: (B, T', n_tokens)

#         # Apply log_softmax - keep in (B, T, C) format
#         log_probs = nn.functional.log_softmax(x, dim=-1)

#         # Calculate output lengths based on the actual sequence reduction
#         log_probs_length = self.transform_input_lengths(spectrogram_length)

#         return {"log_probs": log_probs, "log_probs_length": log_probs_length}

#     def transform_input_lengths(self, input_lengths):
#         """
#         Calculate proper output lengths based on convolution operations:
#         - First conv: stride (2,2) in time dimension
#         - Second conv: stride (2,1) in time dimension
#         - Third conv: stride (2,1) in time dimension
#         Total time reduction: 8
#         """
#         lengths = ((input_lengths - 1) // 8) + 1
#         return lengths.long()
    
    


# ##### WORKING VERSION #####

class DeepSpeech2Model(nn.Module):
    def __init__(self, n_feats, n_tokens, rnn_hidden=512, rnn_layers=5, bidirectional=True):
        super().__init__()
        self.rnn_hidden = rnn_hidden
        self.rnn_layers_count = rnn_layers
        self.bidirectional = bidirectional
        self.n_tokens = n_tokens
        self.n_feats = n_feats

        # Modify convolution layers to reduce less in time dimension
        self.conv_layers = nn.Sequential(
            # Reduce stride in time dimension from (2,2) to (2,1)
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

        # RNN layers and FC will be lazily initialized
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
        log_probs = nn.functional.log_softmax(x, dim=-1)

        # Calculate output lengths based on the actual sequence reduction
        log_probs_length = self.transform_input_lengths(spectrogram_length)

        # Debug prints
        # print("\nLength check:")
        # print(f"Target lengths: {batch.get('text_encoded_length', None)}")
        # print(f"Output lengths: {log_probs_length}")
        # print(f"Actual output time dim: {log_probs.size(1)}")

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        Calculate proper output lengths based on convolution operations:
        - All convs now have stride (2,1) in time dimension
        Total time reduction: 2^3 = 8 in frequency only
        """
        lengths = ((input_lengths - 1) // 2) + 1  # Only one time reduction in first conv
        return lengths.long()
    
##### WORKING VERSION #####


##### IMPROVED STABILITY #####

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DeepSpeech2Model(nn.Module):
#     """
#     Deep Speech 2 Model: Convolutional + Recurrent Network with Clipped ReLU,
#     Batch Normalization, Dropout, and Fully Connected Layer.
#     """

#     def __init__(self, n_feats, n_tokens, rnn_hidden=512, rnn_layers=5, bidirectional=True, dropout_p=0.3):
#         """
#         Args:
#             n_feats (int): Number of input features (frequency bins).
#             n_tokens (int): Number of output tokens (vocabulary size).
#             rnn_hidden (int): Number of hidden units in RNN layers.
#             rnn_layers (int): Number of RNN layers.
#             bidirectional (bool): Use bidirectional RNNs.
#             dropout_p (float): Dropout probability.
#         """
#         super().__init__()
#         self.rnn_hidden = rnn_hidden
#         self.rnn_layers_count = rnn_layers
#         self.bidirectional = bidirectional
#         self.n_tokens = n_tokens
#         self.n_feats = n_feats
#         self.dropout_p = dropout_p

#         # Convolutional Layers
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 1), padding=(20, 5)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
#             nn.BatchNorm2d(96),
#             nn.ReLU(inplace=True),
#         )

#         # Placeholder for RNN Layers and Fully Connected
#         self.rnn_layers = None
#         self.batch_norms = None
#         self.fc = None
#         self.dropout = None

#     def forward(self, spectrogram, spectrogram_length, **batch):
#         """
#         Forward pass of DeepSpeech2.
#         Args:
#             spectrogram (Tensor): Input spectrogram of shape (B, F, T).
#             spectrogram_length (Tensor): Original spectrogram lengths.
#         Returns:
#             dict: Contains "log_probs" and "log_probs_length".
#         """
#         # Input shape: (B, F, T) -> (B, 1, F, T)
#         x = spectrogram.unsqueeze(1)

#         # Pass through Convolutional layers
#         x = self.conv_layers(x)  # Shape: (B, C, F', T')
#         B, C, Freq, Time = x.size()

#         # Flatten for RNN input: (B, T', C*F')
#         x = x.permute(0, 3, 1, 2).contiguous().view(B, Time, C * Freq)

#         # Initialize RNN and FC layers dynamically if needed
#         if self.rnn_layers is None:
#             input_size = C * Freq
#             self.rnn_layers = nn.ModuleList([
#                 nn.GRU(input_size=input_size if i == 0 else self.rnn_hidden * (2 if self.bidirectional else 1),
#                        hidden_size=self.rnn_hidden,
#                        num_layers=1,
#                        batch_first=True,
#                        bidirectional=self.bidirectional)
#                 for i in range(self.rnn_layers_count)
#             ])
#             self.batch_norms = nn.ModuleList([
#                 nn.BatchNorm1d(self.rnn_hidden * (2 if self.bidirectional else 1))
#                 for _ in range(self.rnn_layers_count)
#             ])
#             self.fc = nn.Linear(self.rnn_hidden * (2 if self.bidirectional else 1), self.n_tokens)
#             self.dropout = nn.Dropout(p=self.dropout_p)
#             self.to(x.device)  # Ensure layers are on correct device

#         # Pass through RNN layers with BatchNorm, Dropout, and Clipped ReLU
#         for rnn, bn in zip(self.rnn_layers, self.batch_norms):
#             x, _ = rnn(x)  # Shape: (B, T, Features)
#             x = bn(x.transpose(1, 2)).transpose(1, 2)  # Sequence-wise BN
#             x = torch.clamp(F.relu(x), min=0, max=20)  # Clipped ReLU
#             x = self.dropout(x)

#         # Fully connected layer for character probabilities
#         x = self.fc(x)  # Shape: (B, T, n_tokens)

#         # Apply log_softmax for CTC Loss
#         log_probs = F.log_softmax(x, dim=-1)

#         # Calculate output lengths
#         log_probs_length = self.transform_input_lengths(spectrogram_length)

#         return {"log_probs": log_probs, "log_probs_length": log_probs_length}

#     def transform_input_lengths(self, input_lengths):
#         """
#         Adjust lengths based on convolutional time downsampling.
#         Total time reduction factor = 2 from first Conv layer.
#         """
#         lengths = ((input_lengths - 1) // 2) + 1
#         return lengths.long()


# ##### IMPROVED STABILITY #####