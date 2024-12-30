import torchaudio.transforms as T
from torch import Tensor, nn
import random


class FrequencyMasking(nn.Module):
    """
    Wrapper for torchaudio.transforms.FrequencyMasking with optional probability p.
    """
    def __init__(self, freq_mask_param: int, p: float = 1.0):
        super().__init__()
        self._aug = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.p = p

    def __call__(self, data: Tensor) -> Tensor:
        """
        Overrides __call__ for direct transform application with probability.
        """
        if random.random() < self.p:
            return self._aug(data)
        return data


class TimeMasking(nn.Module):
    """
    Wrapper for torchaudio.transforms.TimeMasking with optional probability p.
    """
    def __init__(self, time_mask_param: int, p: float = 1.0):
        super().__init__()
        self._aug = T.TimeMasking(time_mask_param=time_mask_param)
        self.p = p

    def __call__(self, data: Tensor) -> Tensor:
        """
        Overrides __call__ for direct transform application with probability.
        """
        if random.random() < self.p:
            return self._aug(data)
        return data
