import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict

def collate_fn(dataset_items: List[Dict]):
    """
    A safer, more efficient collate function for handling audio + spectrogram + text data.
    """

    if not dataset_items:
        raise ValueError("Received an empty list of dataset_items.")

    audios, audio_lengths = [], []
    spectrograms, spectrogram_lengths = [], []
    texts, text_encodeds, text_encoded_lengths = [], [], []
    audio_paths = []

    # Collect per-item spectrogram mean and std for later batch-level averaging
    spect_means, spect_stds = [], []

    for idx, item in enumerate(dataset_items):
        # ---------------------------------------------------------------------
        # 1) Validate required keys
        # ---------------------------------------------------------------------
        required_keys = ["audio", "spectrogram", "text", "text_encoded", "audio_path"]
        missing = set(required_keys) - set(item.keys())
        if missing:
            raise KeyError(f"Item {idx} missing keys: {missing}")

        # ---------------------------------------------------------------------
        # 2) Audio validation and collection
        # ---------------------------------------------------------------------
        audio = item["audio"]
        if not isinstance(audio, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for audio, got {type(audio)} at index {idx}")
        # Check NaNs and Infs
        if torch.isnan(audio).any() or torch.isinf(audio).any():
            raise ValueError(f"Invalid audio values at index {idx}")
        # If stereo or multi-channel, down-mix to mono
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        if audio.size(0) == 0:
            raise ValueError(f"Zero-length audio at index {idx}")
        audios.append(audio)
        audio_lengths.append(audio.shape[0])

        # ---------------------------------------------------------------------
        # 3) Spectrogram validation and collection
        # ---------------------------------------------------------------------
        spect = item["spectrogram"]
        if not isinstance(spect, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for spectrogram, got {type(spect)} at index {idx}")
        if spect.ndim != 3:
            raise ValueError(f"Expected spectrogram of shape [1, F, T], got {spect.ndim}D at index {idx}")

        # Check for NaNs, Infs
        if torch.isnan(spect).any() or torch.isinf(spect).any():
            raise ValueError(f"Invalid spectrogram values at index {idx}")

        # Remove the channel dimension => [F, T]
        spect = spect.squeeze(0)
        if spect.size(1) == 0:
            raise ValueError(f"Zero-length spectrogram at index {idx}")

        # Compute per-sample mean/std for normalization
        sample_mean = spect.mean()
        sample_std = spect.std().clamp(min=1e-6)  # Prevent zero division
        spect_means.append(sample_mean)
        spect_stds.append(sample_std)

        spectrograms.append(spect)  # shape [F, T]
        spectrogram_lengths.append(spect.shape[1])

        # ---------------------------------------------------------------------
        # 4) Text processing
        # ---------------------------------------------------------------------
        text_encoded = torch.as_tensor(item["text_encoded"], dtype=torch.long).view(-1)
        if text_encoded.size(0) == 0:
            raise ValueError(f"Empty text encoding at index {idx}")
        text_encodeds.append(text_encoded)
        text_encoded_lengths.append(text_encoded.size(0))

        texts.append(item["text"])
        audio_paths.append(item["audio_path"])

    # -------------------------------------------------------------------------
    # 5) Compute batch-level mean and std (averaging per-sample stats)
    # -------------------------------------------------------------------------
    batch_mean = torch.tensor(spect_means).mean().clamp(min=1e-6)
    batch_std = torch.tensor(spect_stds).mean().clamp(min=1e-6)

    # -------------------------------------------------------------------------
    # 6) Normalize spectrograms: 
    #    - Convert each spectrogram to shape [T, F] for consistent padding
    #    - Apply (spect - batch_mean) / batch_std
    # -------------------------------------------------------------------------
    spectrograms_t = []
    for spect in spectrograms:  
        # [F, T] -> [T, F]
        spect_t = spect.transpose(0, 1)  
        spect_t = (spect_t - batch_mean) / batch_std
        spectrograms_t.append(spect_t)

    # -------------------------------------------------------------------------
    # 7) Pad spectrograms (which are now [T, F]) with 0.0, then add small noise
    # -------------------------------------------------------------------------
    pad_noise_std = 1e-5
    padded_spectrograms_t = pad_sequence(
        spectrograms_t,  # list of [T, F]
        batch_first=True,  # => [B, max_T, F]
        padding_value=0.0
    )
    # Add small random noise to the padded region
    noise = torch.randn_like(padded_spectrograms_t) * pad_noise_std
    padded_spectrograms_t = padded_spectrograms_t + noise

    # Finally, transform back to [B, F, T]
    padded_spectrograms = padded_spectrograms_t.transpose(1, 2)  # => [B, F, max_T]

    # -------------------------------------------------------------------------
    # 8) Return dictionary of padded/processed tensors
    # -------------------------------------------------------------------------
    return {
        "audio": pad_sequence(audios, batch_first=True),  # [B, max_audio_len]
        "audio_length": torch.tensor(audio_lengths, dtype=torch.long),
        "spectrogram": padded_spectrograms,               # [B, F, max_T]
        "spectrogram_length": torch.tensor(spectrogram_lengths, dtype=torch.long),
        "text": texts,                                    # list of strings
        "text_encoded": pad_sequence(
            text_encodeds, batch_first=True, padding_value=0
        ),  # [B, max_text_len]
        "text_encoded_length": torch.tensor(text_encoded_lengths, dtype=torch.long),
        "audio_path": audio_paths
    }
