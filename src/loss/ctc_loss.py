import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def __init__(self, **kwargs):
        """
        Initialize the CTCLoss with zero_infinity=True by default.
        Accepts additional keyword arguments to pass to the superclass.
        """
        super().__init__(zero_infinity=True, **kwargs)
        print("[DEBUG] CTCLossWrapper initialized with zero_infinity=True")
        if hasattr(self, 'blank'):
            print(f"[DEBUG] Blank index set to: {self.blank}")
            
    def forward(
        self, log_probs, log_probs_length, text_encoded, text_encoded_length, **batch
    ) -> Tensor:
        #  Add debug prints
        # print("\nCTC Loss Debug Info:")
        # print(f"log_probs shape: {log_probs.shape}")  # Should be (T, B, C)
        # print(f"log_probs_length: {log_probs_length}")  # Should be (B,)
        # print(f"log_probs_length shape: {log_probs_length.shape}")
        # print(f"text shape: {text_encoded.shape}")  # Should be (B, S) or (sum(text_length),)
        # print(f"text_length: {text_encoded_length}")  # Should be (B,)
        # print(f"text_length shape: {text_encoded_length.shape}")
        # print(f"Batch size from log_probs: {log_probs.shape[1]}")

        # print(log_probs)        
        
        log_probs_t = torch.transpose(log_probs, 0, 1)

        loss = super().forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
        )

        # print(loss)

        return {"loss": loss}
