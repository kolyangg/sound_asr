from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer

# TODO add beam search/lm versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
            # print('cers:')
            # print(cers)
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BaseMetric):
    def __init__(self, text_encoder, beam_size=10, use_lm=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = self.text_encoder.beam_size
        beam_size = self.beam_size
        self.use_lm = self.text_encoder.lm is not None
                
        # debug part
        print(f"BeamSearchCERMetric: beam_size={self.beam_size}, use_lm={use_lm}")
        # self.use_lm = True

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        """
        Calculate CER using beam search decoding.
        """
        cers = []
        debug = False
        
        # Convert log_probs to probabilities
        probs = torch.exp(log_probs.cpu())

        # Loop through each example
        for i in range(len(text)):
            target_text = self.text_encoder.normalize_text(text[i])
            seq_len = log_probs_length[i].item()
            sequence_probs = probs[i, :seq_len]

            # Decode using beam search
            beam_results = self.text_encoder.ctc_beam_search(
                sequence_probs.numpy(),
                beam_size=self.beam_size,
                use_lm=self.use_lm,
            )

            # Best prediction from beam search
            best_pred_text = beam_results[0][0]

            # Calculate CER
            cer = calc_cer(target_text, best_pred_text)
            cers.append(cer)

            # Debugging output
            if debug:
                print(f"\n[DEBUG] Example {i}:")
                print(f"Target Text : '{target_text}'")
                print(f"Beam Prediction : '{best_pred_text}' (CER: {cer * 100:.2f})")
        
        return sum(cers) / len(cers)



        # beam_predictions = []
        # probs = torch.exp(log_probs)  # Convert log_probs to probabilities
        # for i in range(len(text)):
        #     seq_len = log_probs_length[i]
        #     sequence_probs = probs[i, :seq_len]

        #     # Perform beam search
        #     beam_results = self.text_encoder.ctc_beam_search(
        #         sequence_probs.numpy(), beam_size=10, use_lm=use_lm, debug=False
        #     )
        #     best_prediction = beam_results[0][0]  # Top beam result
        #     beam_predictions.append(best_prediction)
        #     print(f"Example {i}: Beam Prediction -> '{best_prediction}'")

        # # Compare and log results
        # for i in range(min(examples_to_log, len(text))):
        #     target = self.text_encoder.normalize_text(text[i])
        #     greedy_pred = greedy_predictions[i]
        #     beam_pred = beam_predictions[i]