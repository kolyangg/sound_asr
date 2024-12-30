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


# import torchaudio.transforms as T
# from torch import Tensor, nn
# import random


# class FrequencyMasking(nn.Module):
#     """
#     Wrapper for torchaudio.transforms.FrequencyMasking with optional probability p.
#     """
#     def __init__(self, freq_mask_param: int, p: float = 1.0, *args, **kwargs):
#         super().__init__()
#         self._aug = T.FrequencyMasking(freq_mask_param=freq_mask_param, *args, **kwargs)
#         self.p = p

#     def __call__(self, data: Tensor) -> Tensor:
#         """
#         Overrides __call__ for direct transform application with probability.
#         """
#         if random.random() < self.p:
#             return self._aug(data)
#         return data


# class TimeMasking(nn.Module):
#     """
#     Wrapper for torchaudio.transforms.TimeMasking with optional probability p.
#     """
#     def __init__(self, time_mask_param: int, p: float = 1.0, *args, **kwargs):
#         super().__init__()
#         self._aug = T.TimeMasking(time_mask_param=time_mask_param, *args, **kwargs)
#         self.p = p

#     def __call__(self, data: Tensor) -> Tensor:
#         """
#         Overrides __call__ for direct transform application with probability.
#         """
#         if random.random() < self.p:
#             return self._aug(data)
#         return data





# import torch_audiomentations
# import torchaudio.transforms as T
# from torch import Tensor, nn


# class FrequencyMasking(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self._aug = T.FrequencyMasking(*args, **kwargs)
        
#     def __call__(self, data: Tensor):
#         # Debug: print input shape
#         print(f"[DEBUG] FrequencyMasking input shape: {data.shape}")
        
#         output = self._aug(data)
        
#         # Debug: print output shape
#         print(f"[DEBUG] FrequencyMasking output shape: {output.shape}")
        
#         return output


# class TimeMasking(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self._aug = T.TimeMasking(*args, **kwargs)
        
#     def __call__(self, data: Tensor):
#         # Debug: print input shape
#         print(f"[DEBUG] TimeMasking input shape: {data.shape}")
        
#         output = self._aug(data)
        
#         # Debug: print output shape
#         print(f"[DEBUG] TimeMasking output shape: {output.shape}")
        
#         return output
