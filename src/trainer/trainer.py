from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer

import torch

class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """
    
    def process_batch(self, batch, metrics: MetricTracker):
        """Process batch, similar to original but with added BPE support"""
        
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        
        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)
        
        all_losses = self.criterion(**batch)
        # print(f"\nLoss value: {all_losses['loss'].item()}")
         
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Update metrics the same way
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch


    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log batch with added support for beam search and LM during inference
        """
        if mode == "train":
            self.log_spectrogram(**batch)
        else:
            self.log_spectrogram(**batch)
            # During inference, try both greedy and beam search predictions
            
            # First log greedy search results
            self.log_predictions(**batch, use_beam_search=False)

            if self.config.trainer.use_beam_search:
            
                # Then log beam search results if available
                if hasattr(self.text_encoder, 'ctc_beam_search'):
                    self.log_predictions(**batch, use_beam_search=self.config.trainer.use_beam_search, use_lm=self.config.trainer.use_lm, 
                                        beam_size = self.config.trainer.beam_size)
                    
                    # If LM is available, also log beam search + LM results
                    if hasattr(self.text_encoder, 'lm') and self.text_encoder.lm is not None:
                        self.log_predictions(**batch, use_beam_search=True, use_lm=True, beam_size = self.config.trainer.beam_size)
                    
        
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log batch with added support for beam search and LM during inference
        """
        if mode == "train":
            self.log_spectrogram(**batch)
        else:
            self.log_spectrogram(**batch)
            
            # Log only the configured decoding method
            self.log_predictions(
                **batch,
                use_beam_search=self.config.trainer.use_beam_search,
                use_lm=self.config.trainer.use_lm,
                beam_size=self.config.trainer.beam_size
            )                
        
    
    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)    
    
    ### AFTER FIXING BEAM SEARCH ###
    
    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path,
        examples_to_log=10, use_beam_search=False, use_lm=False, 
        beam_size=10, **batch
    ):
        """
        Log predictions with corrected greedy and beam search handling.
        Consistency ensured with ctc_beam_search.
        """
        debug1 = False
        debug2 = True
        
        rows = {}
        log_probs = log_probs.cpu()
        log_probs_length = log_probs_length.cpu()

        # Greedy decoding
        if debug1:
            print("\n=== Debug: Greedy Decoding ===")
        argmax_inds = log_probs.argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)] for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        ]
        greedy_predictions = []
        for i, inds in enumerate(argmax_inds):
            # Convert list of indices to torch.Tensor
            inds_tensor = torch.tensor(inds)
            pred = self.text_encoder.ctc_decode(inds_tensor)
            greedy_predictions.append(pred)
            if debug1:
                print(f"Example {i}: Greedy Prediction -> '{pred}'")

        if self.config.trainer.use_beam_search:
            # Beam search decoding
            if debug1:
                print("\n=== Debug: Beam Search Decoding ===")
            beam_predictions = []
            probs = torch.exp(log_probs)  # Convert log_probs to probabilities
            for i in range(len(text)):
                seq_len = log_probs_length[i]
                sequence_probs = probs[i, :seq_len]

                # Perform beam search with dynamic beam_size
                beam_results = self.text_encoder.ctc_beam_search(
                    sequence_probs.numpy(), beam_size=beam_size, use_lm=use_lm, debug=False
                )
                best_prediction = beam_results[0][0]  # Top beam result
                beam_predictions.append(best_prediction)
                if debug1:
                    print(f"Example {i}: Beam Prediction -> '{best_prediction}'")

        # Compare and log results
        for i in range(min(examples_to_log, len(text))):
            target = self.text_encoder.normalize_text(text[i])
            greedy_pred = greedy_predictions[i]
            if self.config.trainer.use_beam_search:
                beam_pred = beam_predictions[i]

            # Calculate CER
            cer_greedy = calc_cer(target, greedy_pred) * 100
            
            if self.config.trainer.use_beam_search:
                cer_beam = calc_cer(target, beam_pred) * 100

            if debug2:
                print(f"\n=== Example {i} Comparison ===")
                print(f"Target Text      : '{target}'")
                print(f"Greedy Prediction: '{greedy_pred}' (CER: {cer_greedy:.2f})")
                if self.config.trainer.use_beam_search:
                    print(f"Beam Prediction  : '{beam_pred}' (CER: {cer_beam:.2f})")

            if self.config.trainer.use_beam_search:
                rows[Path(audio_path[i]).name] = {
                    "target": target,
                    "greedy_prediction": greedy_pred,
                    "beam_prediction": beam_pred,
                    "cer_greedy": cer_greedy,
                    "cer_beam": cer_beam,
                }
            else:
                rows[Path(audio_path[i]).name] = {
                    "target": target,
                    "greedy_prediction": greedy_pred,
                    "cer_greedy": cer_greedy
                }


        # Log table with predictions
        table_name = "predictions"
        if use_beam_search:
            table_name += "_beam_search"
            if use_lm:
                table_name += "_with_lm"

        self.writer.add_table(table_name, pd.DataFrame.from_dict(rows, orient="index"))

    
    
    
    
