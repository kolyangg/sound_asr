import torch
from torch.nn.utils.rnn import pad_sequence # NEW
import torch.nn.functional as F # NEW

# def collate_fn(dataset_items: list[dict]):
#     """
#     Collate and pad fields in the dataset items.
#     Converts individual items into a batch.

#     Args:
#         dataset_items (list[dict]): list of objects from
#             dataset.__getitem__.
#     Returns:
#         result_batch (dict[Tensor]): dict, containing batch-version
#             of the tensors.
#     """

#     # pass  # TODO
    
#     ### TO CHECK A LOT !!! ###
    
#     audios = []
#     spectrograms = []
#     texts = []
#     text_encodeds = []
#     audio_paths = []
#     spectrogram_lengths = []
#     text_encoded_lengths = []

#     for item in dataset_items:
#         # Handle audio (convert to mono if needed)
#         audio = item["audio"]
#         if audio.ndim == 2:
#             audio = audio.mean(dim=0)  # [T]
#         audios.append(audio)

#         # Handle spectrogram
#         spect = item["spectrogram"]
#         # If spect is 3D: [C, F, T], reduce channels
#         if spect.ndim == 3:
#             # Average across channels to get [F, T]
#             spect = spect.mean(dim=0)
#         # Now spect should be [F, T]
#         spectrograms.append(spect)
#         spectrogram_lengths.append(spect.shape[1])  # Store the time dimension length

#         texts.append(item["text"])

#         text_encoded = item["text_encoded"]
#         if not torch.is_tensor(text_encoded):
#             text_encoded = torch.tensor(text_encoded, dtype=torch.long)
        
#         # Ensure it's 1D
#         if text_encoded.ndim > 1:
#             text_encoded = text_encoded.view(-1)
        
#         text_encodeds.append(text_encoded)
#         text_encoded_lengths.append(len(text_encoded))  # Store text length

#         audio_paths.append(item["audio_path"])

#     # Pad audio sequences: [B, max_T_audio]
#     padded_audios = pad_sequence(audios, batch_first=True)

#     # Ensure consistent frequency dimension for spectrograms
#     # All spectrograms are now [T, F]. We must ensure all have the same F.
#     max_freq = max(spect.shape[0] for spect in spectrograms)
#     max_time = max(spect.shape[1] for spect in spectrograms)
    
#     # # Pad both frequency and time dimensions
#     # padded_spectrograms = []
    
#     # # for i, spect in enumerate(spectrograms):
#     # #     if spect.shape[1] < max_freq:
#     # #         diff = max_freq - spect.shape[1]
#     # #         # Pad freq dimension on the right
#     # #         spectrograms[i] = F.pad(spect, (0, diff))

#     # aligned_spectrograms = []
    
#     # for spect in spectrograms:
#     #     # Print original spectrogram stats to verify data
#     #     print(f"Original spect shape: {spect.shape}, min: {spect.min()}, max: {spect.max()}")
        
#     #     if spect.shape[0] < max_freq:
#     #         # Use a different padding value (like -100) to clearly see what's being padded
#     #         padding = torch.zeros((max_freq - spect.shape[0], spect.shape[1]), 
#     #                             dtype=spect.dtype, 
#     #                             device=spect.device)
#     #         spect = torch.cat([spect, padding], dim=0)
            
#     #     # Don't transpose here - keep original layout
#     #     aligned_spectrograms.append(spect)  # Keep as [F, T]

#     # # # Now we can use pad_sequence since all frequency dimensions match
#     # # padded_spectrograms = pad_sequence(aligned_spectrograms, batch_first=True)  # [B, T, F]
#     # # # Transpose back to [B, F, T]
#     # # padded_spectrograms = padded_spectrograms.transpose(1, 2)  # [B, F, T]
    
    
#     # # Stack spectrograms directly - they're already aligned in frequency
#     # padded_spectrograms = torch.stack(aligned_spectrograms)  # [B, F, T]
    
#     # # Pad text encodings: [B, max_length_text]
#     # if not torch.is_tensor(text_encoded):
#     #     text_encoded = torch.tensor(text_encoded, dtype=torch.long)

#     # # Ensure it's 1D
#     # if text_encoded.ndim > 1:
#     #     # Flatten the tensor if it's accidentally multi-dimensional
#     #     text_encoded = text_encoded.view(-1)
        
#     # Create padded spectrograms tensor directly
#     batch_size = len(spectrograms)
#     freq_dim = spectrograms[0].shape[0]  # Should be 128 for all
#     padded_spectrograms = torch.zeros((batch_size, freq_dim, max_time),
#                                     dtype=spectrograms[0].dtype,
#                                     device=spectrograms[0].device)
    
#     # Fill in the actual values
#     for i, spect in enumerate(spectrograms):
#         padded_spectrograms[i, :, :spect.shape[1]] = spect
    
#     padded_text_encoded = pad_sequence(text_encodeds, batch_first=True, padding_value=0)

#     spectrogram_lengths = torch.tensor(spectrogram_lengths, dtype=torch.long)
#     text_encoded_lengths = torch.tensor(text_encoded_lengths, dtype=torch.long)  # Convert to tensor

    
#     return {
#         "audio": padded_audios,
#         "spectrogram": padded_spectrograms,
#         "text": texts,
#         "text_encoded": padded_text_encoded,
#         "audio_path": audio_paths,
#         "spectrogram_length": spectrogram_lengths,
#         "text_encoded_length": text_encoded_lengths  
#     }
    
#     ### TO CHECK A LOT !!! ###

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    """
    audios = []
    spectrograms = []
    texts = []
    text_encodeds = []
    audio_paths = []
    spectrogram_lengths = []
    text_encoded_lengths = []

    # print("\n=== Initial Spectrogram Check ===")
    for idx, item in enumerate(dataset_items):
        # Handle audio
        audio = item["audio"]
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        audios.append(audio)

        # Handle spectrogram with debug info
        spect = item["spectrogram"]  # Shape: [1, 128, T]
        # print(f"\nItem {idx}:")
        # print(f"Initial spect type: {type(spect)}, dtype: {spect.dtype}")
        # print(f"Initial shape: {spect.shape}")
        # print(f"Initial value range: min={spect.min().item():.4f}, max={spect.max().item():.4f}")
        
        # Remove the channel dimension (squeeze) and keep as [F, T]
        spect = spect.squeeze(0)  # Now shape: [128, T]
        # print(f"After squeeze - shape: {spect.shape}")
        # print(f"After squeeze - range: min={spect.min().item():.4f}, max={spect.max().item():.4f}")
        
        spectrograms.append(spect)
        spectrogram_lengths.append(spect.shape[1])  # T dimension

        # Rest of the item processing...
        texts.append(item["text"])
        text_encoded = item["text_encoded"]
        if not torch.is_tensor(text_encoded):
            text_encoded = torch.tensor(text_encoded, dtype=torch.long)
        if text_encoded.ndim > 1:
            text_encoded = text_encoded.view(-1)
        text_encodeds.append(text_encoded)
        text_encoded_lengths.append(len(text_encoded))
        audio_paths.append(item["audio_path"])

    # Find max dimensions for spectrograms
    max_time = max(spect.shape[1] for spect in spectrograms)
    
    # print("\n=== Before Padding ===")
    # print(f"Max time dimension: {max_time}")
    # print(f"Number of spectrograms: {len(spectrograms)}")
    # print(f"First spectrogram shape: {spectrograms[0].shape}")
    # print(f"First spectrogram stats: min={spectrograms[0].min().item():.4f}, max={spectrograms[0].max().item():.4f}")

    # Create padded spectrograms tensor
    batch_size = len(spectrograms)
    freq_dim = spectrograms[0].shape[0]  # Should be 128
    padded_spectrograms = torch.zeros((batch_size, freq_dim, max_time),
                                    dtype=spectrograms[0].dtype,
                                    device=spectrograms[0].device)
    
    # Fill in the actual values with debug info
    for i, spect in enumerate(spectrograms):
        # print(f"\nCopying spectrogram {i}:")
        # print(f"Source shape: {spect.shape}")
        # print(f"Source range: min={spect.min().item():.4f}, max={spect.max().item():.4f}")
        padded_spectrograms[i, :, :spect.shape[1]] = spect
        # print(f"After copy - range at position {i}: min={padded_spectrograms[i].min().item():.4f}, max={padded_spectrograms[i].max().item():.4f}")

    # print("\n=== Final Check ===")
    # print(f"Final tensor shape: {padded_spectrograms.shape}")
    # print(f"Final value range: min={padded_spectrograms.min().item():.4f}, max={padded_spectrograms.max().item():.4f}")
    # print(f"Non-zero elements: {torch.count_nonzero(padded_spectrograms).item()}")

    # Rest of the processing
    padded_audios = pad_sequence(audios, batch_first=True)
    padded_text_encoded = pad_sequence(text_encodeds, batch_first=True, padding_value=0)
    spectrogram_lengths = torch.tensor(spectrogram_lengths, dtype=torch.long)
    text_encoded_lengths = torch.tensor(text_encoded_lengths, dtype=torch.long)

    # print(audio_paths) # CHECK WHICH AUDIO FILES ARE BEING USED!!!
    
    return {
        "audio": padded_audios,
        "spectrogram": padded_spectrograms,  # [B, F, T]
        "text": texts,
        "text_encoded": padded_text_encoded,
        "audio_path": audio_paths,
        "spectrogram_length": spectrogram_lengths,
        "text_encoded_length": text_encoded_lengths
    }