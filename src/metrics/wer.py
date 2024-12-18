from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
            # print('wers:')
            # print(wers)
        return sum(wers) / len(wers)


# class BeamSearchWERMetric(BaseMetric):
#     def __init__(self, text_encoder, beam_size=100, use_lm=True, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.text_encoder = text_encoder
#         self.beam_size = self.text_encoder.beam_size
#         beam_size = self.beam_size
#         self.use_lm = self.text_encoder.lm is not None
        
#         # debug part
#         print(f"BeamSearchWERMetric: beam_size={self.beam_size}, use_lm={use_lm}")
#         # self.use_lm = True

#     def __call__(
#         self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
#     ):
#         """
#         Calculate WER using beam search decoding.
#         """
#         wers = []
        
#         debug = False
        
#         # Convert log_probs to probabilities
#         probs = torch.exp(log_probs.cpu())

#         # Loop through each example
#         for i in range(len(text)):
#             target_text = self.text_encoder.normalize_text(text[i])
#             seq_len = log_probs_length[i].item()
#             sequence_probs = probs[i, :seq_len]

#             # Decode using beam search
#             beam_results = self.text_encoder.ctc_beam_search(
#                 sequence_probs.numpy(),
#                 beam_size=self.beam_size,
#                 use_lm=self.use_lm,
#             )

#             # Best prediction from beam search
#             best_pred_text = beam_results[0][0]

#             # Calculate WER
#             wer = calc_wer(target_text, best_pred_text)
#             wers.append(wer)

#             # Debugging output
#             if debug:
#                 print(f"\n[DEBUG] Example {i}:")
#                 print(f"Target Text : '{target_text}'")
#                 print(f"Beam Prediction : '{best_pred_text}' (WER: {wer * 100:.2f})")
        
#         return sum(wers) / len(wers)
    
    

class BeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder, beam_size=10, use_lm=False, *args, **kwargs):       
        
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = self.text_encoder.beam_size
        beam_size = self.beam_size
        
        self.use_lm = self.text_encoder.lm is not None
        use_lm = self.use_lm
        
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