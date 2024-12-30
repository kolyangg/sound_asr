import torch_audiomentations
from torch import Tensor, nn
import random
import torch
import torchaudio.functional as F
from torch import Tensor, nn


class Gain(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)

class PitchShift(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.PitchShift(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)

class AddBackgroundNoise(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.AddBackgroundNoise(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)


class Shift(nn.Module):
    """
    Applies time-domain shift augmentation to the audio waveform.

    Args:
        min_shift (float): Minimum shift in seconds (negative for left shift).
        max_shift (float): Maximum shift in seconds (positive for right shift).
        rollover (bool): If True, shifted-out samples will roll back in.
        sample_rate (int): Sample rate of the audio.
        p (float): Probability of applying the transform.
    """
    def __init__(self, min_shift: float = -0.1, shift_max: float = 0.1, rollover: bool = True, sample_rate: int = 16000, p: float = 0.5,*args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.Shift(
            min_shift=min_shift,
            max_shift=shift_max,
            rollover=rollover,
            sample_rate=sample_rate,
            p=p
        )

    def __call__(self, data: Tensor) -> Tensor:
        """
        Applies the Shift augmentation.

        Args:
            data (Tensor): Input tensor of shape (B, T).

        Returns:
            Tensor: Shifted tensor of shape (B, T).
        """
        # Unsqueeze to add channel dimension as expected by torch_audiomentations
        data = data.unsqueeze(1)  # Shape: (B, 1, T)
        augmented_data = self._aug(data)
        return augmented_data.squeeze(1)  # Back to (B, T)



class ApplyImpulseResponse(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.ApplyImpulseResponse(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
    



class SpeedPerturb(nn.Module):
    """
    Wave-level speed perturbation using torchaudio.functional.resample
    for each audio in the batch.

    Args:
        speeds (list of float): possible speed factors, e.g. [0.9, 1.0, 1.1]
        p (float): probability to apply speed perturbation
        sr (int): original sample rate, e.g. 16000
    """
    def __init__(self, speeds=[0.9, 1.0, 1.1], p=0.5, sample_rate=16000,*args, **kwargs):
        super().__init__()
        self.speeds = speeds
        self.p = p
        self.sr = sample_rate

    def __call__(self, data: Tensor) -> Tensor:
        """
        data: (B, num_samples)
        Returns a potentially padded (B, new_num_samples) wave batch.
        """
        # Randomly decide if we apply speed perturb to this entire batch
        if random.random() > self.p:
            return data

        # Pick one speed factor for the whole batch this time
        speed = random.choice(self.speeds)
        new_sr = int(self.sr * speed)

        # We'll process each sample in the batch individually,
        # then pad to max length so we can keep a consistent batch dimension.
        batch_size = data.shape[0]
        waves_out = []

        for b in range(batch_size):
            wave_b = data[b]  # shape (num_samples,)
            # Step 1: Resample from sr -> new_sr
            wave_res = F.resample(wave_b, self.sr, new_sr)
            # Step 2: Resample back from new_sr -> sr
            wave_out = F.resample(wave_res, new_sr, self.sr)
            waves_out.append(wave_out)

        # Now each wave_out may have a different length => we pad
        max_len = max(w.shape[0] for w in waves_out)
        out = torch.zeros((batch_size, max_len), dtype=data.dtype, device=data.device)

        for i, wave_out in enumerate(waves_out):
            length = wave_out.shape[0]
            out[i, :length] = wave_out

        return out
