import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict

def collate_fn(dataset_items: List[Dict]):
    if not dataset_items:
        raise ValueError("Received an empty list of dataset_items.")

    audios, audio_lengths = [], []
    spectrograms, spectrogram_lengths = [], []
    texts, text_encodeds, text_encoded_lengths = [], [], []
    audio_paths = []
    
    # Collect statistics for normalization
    spect_means, spect_stds = [], []

    for idx, item in enumerate(dataset_items):
        # Validate required keys
        required_keys = ["audio", "spectrogram", "text", "text_encoded", "audio_path"]
        if missing := set(required_keys) - set(item.keys()):
            raise KeyError(f"Item {idx} missing keys: {missing}")

        # Audio validation and processing
        audio = item["audio"]
        if not isinstance(audio, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for audio, got {type(audio)} at {idx}")
        if torch.isnan(audio).any() or torch.isinf(audio).any():
            raise ValueError(f"Invalid audio values at index {idx}")
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        if audio.size(0) == 0:
            raise ValueError(f"Zero-length audio at index {idx}")
        audios.append(audio)
        audio_lengths.append(audio.shape[0])

        # Spectrogram validation and processing
        spect = item["spectrogram"]
        if not isinstance(spect, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for spectrogram, got {type(spect)} at {idx}")
        if spect.ndim != 3:
            raise ValueError(f"Expected 3D spectrogram [1,F,T], got {spect.ndim}D at {idx}")
        if torch.isnan(spect).any() or torch.isinf(spect).any():
            raise ValueError(f"Invalid spectrogram values at index {idx}")
            
        spect = spect.squeeze(0)
        if spect.size(1) == 0:
            raise ValueError(f"Zero-length spectrogram at index {idx}")
            
        # Calculate statistics
        spect_mean = spect.mean()
        spect_std = spect.std().clamp(min=1e-6)
        spect_means.append(spect_mean)
        spect_stds.append(spect_std)
        
        spectrograms.append(spect)
        spectrogram_lengths.append(spect.shape[1])

        # Text processing
        text_encoded = torch.as_tensor(item["text_encoded"], dtype=torch.long).view(-1)
        if text_encoded.size(0) == 0:
            raise ValueError(f"Empty text encoding at index {idx}")
        text_encodeds.append(text_encoded)
        text_encoded_lengths.append(text_encoded.size(0))
        
        texts.append(item["text"])
        audio_paths.append(item["audio_path"])

    # Normalize spectrograms
    batch_mean = torch.tensor(spect_means).mean()
    batch_std = torch.tensor(spect_stds).mean()
    spectrograms_t = [(spect.permute(1, 0) - batch_mean) / batch_std for spect in spectrograms]

    # Padding with small random noise instead of zeros
    pad_noise_std = 1e-5
    padded_spectrograms_t = pad_sequence(
        spectrograms_t, 
        batch_first=True, 
        padding_value=0.0
    ) + torch.randn_like(pad_sequence(spectrograms_t, batch_first=True)) * pad_noise_std

    return {
        "audio": pad_sequence(audios, batch_first=True),
        "audio_length": torch.tensor(audio_lengths, dtype=torch.long),
        "spectrogram": padded_spectrograms_t.permute(0, 2, 1),
        "spectrogram_length": torch.tensor(spectrogram_lengths, dtype=torch.long),
        "text": texts,
        "text_encoded": pad_sequence(text_encodeds, batch_first=True, padding_value=0),
        "text_encoded_length": torch.tensor(text_encoded_lengths, dtype=torch.long),
        "audio_path": audio_paths
    }




# import torch
# from torch.nn.utils.rnn import pad_sequence
# from typing import List, Dict

# def collate_fn(dataset_items: List[Dict]):
#     """
#     Collate and pad fields in the dataset items.

#     Args:
#         dataset_items (List[Dict]): A list of dataset items, each containing
#             'audio', 'spectrogram', 'text', 'text_encoded', and 'audio_path'.

#     Returns:
#         Dict: A dictionary containing padded and collated tensors along with
#               necessary metadata.
#     """
#     if not dataset_items:
#         raise ValueError("Received an empty list of dataset_items.")

#     audios = []
#     audio_lengths = []
#     spectrograms = []
#     spectrogram_lengths = []
#     texts = []
#     text_encodeds = []
#     text_encoded_lengths = []
#     audio_paths = []

#     for idx, item in enumerate(dataset_items):
#         # Validate required keys
#         required_keys = ["audio", "spectrogram", "text", "text_encoded", "audio_path"]
#         for key in required_keys:
#             if key not in item:
#                 raise KeyError(f"Item at index {idx} is missing the key '{key}'.")

#         # Handle audio
#         audio = item["audio"]
#         if not isinstance(audio, torch.Tensor):
#             raise TypeError(f"'audio' should be a torch.Tensor, got {type(audio)} at index {idx}.")
#         if audio.ndim == 2:
#             audio = audio.mean(dim=0)
#         audios.append(audio)
#         audio_lengths.append(audio.shape[0])

#         # Handle spectrogram
#         spect = item["spectrogram"]
#         if not isinstance(spect, torch.Tensor):
#             raise TypeError(f"'spectrogram' should be a torch.Tensor, got {type(spect)} at index {idx}.")
#         if spect.ndim != 3:
#             raise ValueError(f"'spectrogram' should have 3 dimensions [1, F, T], got {spect.ndim} at index {idx}.")
#         spect = spect.squeeze(0)  # Shape: [F, T]
#         if spect.ndim != 2:
#             raise ValueError(f"Squeezed 'spectrogram' should have 2 dimensions [F, T], got {spect.ndim} at index {idx}.")
#         spectrograms.append(spect)
#         spectrogram_lengths.append(spect.shape[1])

#         # Handle text
#         texts.append(item["text"])
#         text_encoded = item["text_encoded"]
#         if not torch.is_tensor(text_encoded):
#             text_encoded = torch.tensor(text_encoded, dtype=torch.long)
#         if text_encoded.ndim > 1:
#             text_encoded = text_encoded.view(-1)
#         text_encodeds.append(text_encoded)
#         text_encoded_lengths.append(text_encoded.size(0))

#         # Handle audio_path
#         audio_paths.append(item["audio_path"])

#     # Pad audios
#     padded_audios = pad_sequence(audios, batch_first=True, padding_value=0.0)  # Shape: [B, max_audio_length]
#     audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)  # Shape: [B]

#     # Pad spectrograms
#     # Transpose spectrograms to [T, F] for pad_sequence
#     spectrograms_t = [spect.permute(1, 0) for spect in spectrograms]  # List of [T, F]
#     padded_spectrograms_t = pad_sequence(spectrograms_t, batch_first=True, padding_value=0.0)  # [B, max_T, F]
#     padded_spectrograms = padded_spectrograms_t.permute(0, 2, 1)  # [B, F, max_T]
#     spectrogram_lengths = torch.tensor(spectrogram_lengths, dtype=torch.long)  # [B]

#     # Pad text_encoded
#     padded_text_encoded = pad_sequence(text_encodeds, batch_first=True, padding_value=0)  # [B, max_text_length]
#     text_encoded_lengths = torch.tensor(text_encoded_lengths, dtype=torch.long)  # [B]

#     return {
#         "audio": padded_audios,  # [B, max_audio_length]
#         "audio_length": audio_lengths,  # [B]
#         "spectrogram": padded_spectrograms,  # [B, F, max_T]
#         "spectrogram_length": spectrogram_lengths,  # [B]
#         "text": texts,  # List of strings
#         "text_encoded": padded_text_encoded,  # [B, max_text_length]
#         "text_encoded_length": text_encoded_lengths,  # [B]
#         "audio_path": audio_paths  # List of strings
#     }
