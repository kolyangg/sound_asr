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

    # def process_batch(self, batch, metrics: MetricTracker):
    #     """
    #     Run batch through the model, compute metrics, compute loss,
    #     and do training step (during training stage).

    #     The function expects that criterion aggregates all losses
    #     (if there are many) into a single one defined in the 'loss' key.

    #     Args:
    #         batch (dict): dict-based batch containing the data from
    #             the dataloader.
    #         metrics (MetricTracker): MetricTracker object that computes
    #             and aggregates the metrics. The metrics depend on the type of
    #             the partition (train or inference).
    #     Returns:
    #         batch (dict): dict-based batch containing the data from
    #             the dataloader (possibly transformed via batch transform),
    #             model outputs, and losses.
    #     """
    #     batch = self.move_batch_to_device(batch)
    #     batch = self.transform_batch(batch)  # transform batch on device -- faster
        
    #     metric_funcs = self.metrics["inference"]
    #     if self.is_train:
    #         metric_funcs = self.metrics["train"]
    #         self.optimizer.zero_grad()

    #     outputs = self.model(**batch)
    #     batch.update(outputs)

    #     all_losses = self.criterion(**batch)
    #     batch.update(all_losses)

    #     if self.is_train:
    #         batch["loss"].backward()  # sum of all losses is always called loss
    #         self._clip_grad_norm()
    #         self.optimizer.step()
    #         if self.lr_scheduler is not None:
    #             self.lr_scheduler.step()

    #     # update metrics for each loss (in case of multiple losses)
    #     for loss_name in self.config.writer.loss_names:
    #         metrics.update(loss_name, batch[loss_name].item())

    #     for met in metric_funcs:
    #         metrics.update(met.name, met(**batch))
    #     return batch
    
    def process_batch(self, batch, metrics: MetricTracker):
        """Process batch, similar to original but with added BPE support"""
        
        # print("\nTrainer Process Batch Debug:")
        # print("Batch contents:")
        # for k, v in batch.items():
        #     if torch.is_tensor(v):
        #         print(f"{k}: shape {v.shape}")
        #     else:
        #         print(f"{k}: {v}")
        
        
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        
        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)
        
        # print("\nAfter model forward:")
        # print(f"log_probs shape: {batch['log_probs'].shape}")
        # print(f"log_probs_length: {batch['log_probs_length']}")


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

    # def _log_batch(self, batch_idx, batch, mode="train"):
    #     """
    #     Log data from batch. Calls self.writer.add_* to log data
    #     to the experiment tracker.

    #     Args:
    #         batch_idx (int): index of the current batch.
    #         batch (dict): dict-based batch after going through
    #             the 'process_batch' function.
    #         mode (str): train or inference. Defines which logging
    #             rules to apply.
    #     """
    #     # method to log data from you batch
    #     # such as audio, text or images, for example

    #     # logging scheme might be different for different partitions
    #     if mode == "train":  # the method is called only every self.log_step steps
    #         self.log_spectrogram(**batch)
    #     else:
    #         # Log Stuff
    #         self.log_spectrogram(**batch)
    #         self.log_predictions(**batch)

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

    # def log_predictions(
    #     self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch
    # ):
    #     # TODO add beam search
    #     # Note: by improving text encoder and metrics design
    #     # this logging can also be improved significantly

    #     argmax_inds = log_probs.cpu().argmax(-1).numpy()
    #     argmax_inds = [
    #         inds[: int(ind_len)]
    #         for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
    #     ]
    #     argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
    #     argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
    #     tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path))

    #     rows = {}
    #     for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
    #         target = self.text_encoder.normalize_text(target)
    #         wer = calc_wer(target, pred) * 100
    #         cer = calc_cer(target, pred) * 100

    #         rows[Path(audio_path).name] = {
    #             "target": target,
    #             "raw prediction": raw_pred,
    #             "predictions": pred,
    #             "wer": wer,
    #             "cer": cer,
    #         }
    #     self.writer.add_table(
    #         "predictions", pd.DataFrame.from_dict(rows, orient="index")
    #     )
        
    ### BEFORE FIXING BEAM SEARCH ###
    # def log_predictions(
    #     self, text, log_probs, log_probs_length, audio_path, 
    #     examples_to_log=10, use_beam_search=False, use_lm=False, **batch
    # ):
    #     """
    #     Log predictions with support for beam search, LM, and BPE
    #     """
    #     rows = {}
    #     # use_beam_search = self.config.trainer.use_beam_search
        
    #     if use_beam_search:
    #         # Convert log_probs to probabilities for beam search
    #         probs = torch.exp(log_probs.cpu())
            
    #         predictions = []
    #         raw_predictions = []
            
    #         debug_beam_search = True # was True
            
    #         if debug_beam_search:
    #             # Debug first example in detail
    #             print("\nDEBUG: Analyzing first example")
    #             print(f"True text: {text[0]}")
    #             sequence_probs = probs[0, :log_probs_length[0]]
    #             beam_results = self.text_encoder.ctc_beam_search(
    #                 sequence_probs, 
    #                 beam_size=10,
    #                 use_lm=use_lm,
    #                 debug=True
    #             )        
                
    #         for i in range(len(text)):
    #             seq_len = log_probs_length[i]
    #             sequence_probs = probs[i, :seq_len]
                
    #             if use_lm:
    #                 beam_results = self.text_encoder.ctc_beam_search(sequence_probs, beam_size=10, use_lm=True)
    #             else:
    #                 beam_results = self.text_encoder.ctc_beam_search(sequence_probs, beam_size=10, use_lm=False)
                
    #             # Get best prediction
    #             best_text = beam_results[0][0]
    #             predictions.append(best_text)
                
    #             # Store top-3 predictions as raw prediction
    #             raw_pred = " || ".join([f"{text} ({score:.3f})" 
    #                                 for text, score in beam_results[:3]])
    #             raw_predictions.append(raw_pred)
    #     else:
    #         # Regular greedy decoding
    #         argmax_inds = log_probs.cpu().argmax(-1).numpy()
    #         argmax_inds = [
    #             inds[: int(ind_len)]
    #             for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
    #         ]
    #         # Handle BPE if enabled
    #         predictions = []
    #         raw_predictions = []
    #         for inds in argmax_inds:
    #             raw_text = self.text_encoder.decode(inds)
    #             pred_text = self.text_encoder.ctc_decode(inds)
                
    #             if hasattr(self.text_encoder, 'use_bpe') and self.text_encoder.use_bpe:
    #                 raw_text = self.text_encoder.tokenizer.clean_up_tokenization(raw_text)
    #                 pred_text = self.text_encoder.tokenizer.clean_up_tokenization(pred_text)
                
    #             predictions.append(pred_text)
    #             raw_predictions.append(raw_text)

    #     # Create table rows
    #     tuples = list(zip(predictions, text, raw_predictions, audio_path))
    #     for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
    #         target = self.text_encoder.normalize_text(target)
    #         wer = calc_wer(target, pred) * 100
    #         cer = calc_cer(target, pred) * 100

    #         rows[Path(audio_path).name] = {
    #             "target": target,
    #             "raw prediction": raw_pred,
    #             "prediction": pred,
    #             "wer": wer,
    #             "cer": cer,
    #         }

    #     # Add decoding method to table name
    #     table_name = "predictions"
    #     if use_beam_search:
    #         table_name += "_beam_search"
    #         if use_lm:
    #             table_name += "_with_lm"

    #     self.writer.add_table(
    #         table_name, pd.DataFrame.from_dict(rows, orient="index")
    #     )

    # ### BEFORE FIXING BEAM SEARCH ###
    
    
    ### AFTER FIXING BEAM SEARCH ###
    
    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path,
        examples_to_log=10, use_beam_search=False, use_lm=False, 
        beam_size = 10, **batch
    ):
        """
        Log predictions with corrected greedy and beam search handling.
        Consistency ensured with ctc_beam_search.
        """
        debug = False
        
        rows = {}
        log_probs = log_probs.cpu()
        log_probs_length = log_probs_length.cpu()

        # Greedy decoding
        if debug:
            print("\n=== Debug: Greedy Decoding ===")
        argmax_inds = log_probs.argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)] for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        ]
        greedy_predictions = []
        for i, inds in enumerate(argmax_inds):
            pred = self.text_encoder.ctc_decode(inds)
            greedy_predictions.append(pred)
            if debug:
                print(f"Example {i}: Greedy Prediction -> '{pred}'")

        # Beam search decoding
        if debug:
            print("\n=== Debug: Beam Search Decoding ===")
        beam_predictions = []
        probs = torch.exp(log_probs)  # Convert log_probs to probabilities
        for i in range(len(text)):
            seq_len = log_probs_length[i]
            sequence_probs = probs[i, :seq_len]

            # Perform beam search
            beam_results = self.text_encoder.ctc_beam_search(
                sequence_probs.numpy(), beam_size=10, use_lm=use_lm, debug=False
            )
            best_prediction = beam_results[0][0]  # Top beam result
            beam_predictions.append(best_prediction)
            if debug:
                print(f"Example {i}: Beam Prediction -> '{best_prediction}'")

        # Compare and log results
        for i in range(min(examples_to_log, len(text))):
            target = self.text_encoder.normalize_text(text[i])
            greedy_pred = greedy_predictions[i]
            beam_pred = beam_predictions[i]

            # Calculate CER
            cer_greedy = calc_cer(target, greedy_pred) * 100
            cer_beam = calc_cer(target, beam_pred) * 100

            if debug:
                print(f"\n=== Example {i} Comparison ===")
                print(f"Target Text      : '{target}'")
                print(f"Greedy Prediction: '{greedy_pred}' (CER: {cer_greedy:.2f})")
                print(f"Beam Prediction  : '{beam_pred}' (CER: {cer_beam:.2f})")

            rows[Path(audio_path[i]).name] = {
                "target": target,
                "greedy_prediction": greedy_pred,
                "beam_prediction": beam_pred,
                "cer_greedy": cer_greedy,
                "cer_beam": cer_beam,
            }

        # Log table with predictions
        table_name = "predictions"
        if use_beam_search:
            table_name += "_beam_search"
            if use_lm:
                table_name += "_with_lm"

        self.writer.add_table(table_name, pd.DataFrame.from_dict(rows, orient="index"))



    ### AFTER FIXING BEAM SEARCH ###