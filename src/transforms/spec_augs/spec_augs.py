import torch_audiomentations
import torchaudio.transforms as T
from torch import Tensor, nn


class FrequencyMasking(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = T.FrequencyMasking(*args, **kwargs)
        
    def __call__(self, data: Tensor):
        return self._aug(data)

class TimeMasking(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = T.TimeMasking(*args, **kwargs)
        
    def __call__(self, data: Tensor):
        return self._aug(data)