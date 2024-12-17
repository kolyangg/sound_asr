import torch_audiomentations
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
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.Shift(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)

class ApplyImpulseResponse(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.ApplyImpulseResponse(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)