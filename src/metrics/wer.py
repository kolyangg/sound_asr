from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()  # Shape: [batch_size, time_steps]
        lengths = log_probs_length.detach().numpy()  # Shape: [batch_size]

        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            # Use decode_indices instead of ctc_decode since log_prob_vec are indices
            pred_text = self.text_encoder.decode_indices(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)

    

class BeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder, beam_size=10, use_lm=True, *args, **kwargs):       
        
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = self.text_encoder.beam_size
        beam_size = self.beam_size
        
        self.use_lm = self.text_encoder.lm is not None
        use_lm = self.use_lm
        
        # self.use_lm = True # TEMP FIX!!!
        self.use_lm = self.text_encoder.use_lm
        
        # debug part
        print(f"BeamSearchWERMetric: beam_size={self.beam_size}, use_lm={self.use_lm}")
        # self.use_lm = True

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        """
        Calculate WER using batched beam search decoding.
        """
        # Convert log_probs to probabilities
        probs = torch.exp(log_probs.cpu())
        
        # Decode predictions in batches
        predictions = [
            self.text_encoder.ctc_beam_search(
                probs[i, :log_probs_length[i].item()].numpy(),
                beam_size=self.beam_size,
                use_lm=self.use_lm,
            )[0][0]
            for i in range(len(text))
        ]

        # Normalize targets
        normalized_targets = [self.text_encoder.normalize_text(t) for t in text]

        # Batch WER evaluation
        wers = [
            calc_wer(target, pred) 
            for target, pred in zip(normalized_targets, predictions)
        ]
        return sum(wers) / len(wers)