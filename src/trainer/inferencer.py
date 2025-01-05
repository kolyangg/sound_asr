import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            text_encoder (CTCTextEncoder): text encoder.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.text_encoder = text_encoder

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

        #
        self.text_encoder.lm_weight = self.config.text_encoder.lm_weight

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part, use_beam_search=False, use_lm=False, beam_size=10):
        """
        Process batch during inference, with option to use beam search.
        Also handles BPE tokenization if text_encoder has use_bpe=True.
        
        Args:
            batch_idx (int): the index of the current batch
            batch (dict): dict-based batch containing the data
            metrics (MetricTracker): metrics tracker
            part (str): partition name
            use_beam_search (bool): whether to use beam search decoding
            use_lm (bool): whether to use language model with beam search
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        # lm_weight = self.text_encoder.lm_weight
        
        self.text_encoder.lm_weight = self.config.text_encoder.lm_weight
        use_beam_search = self.config.text_encoder.use_beam_search
        self.text_encoder.use_beam_search = use_beam_search
        beam_size = self.config.inferencer.beam_size

        self.text_encoder.use_lm = self.config.text_encoder.use_lm
        use_lm = self.text_encoder.use_lm

        self.text_encoder.use_bpe = self.config.text_encoder.use_bpe
        use_bpe = self.text_encoder.use_bpe

        # print('Inference:')
        # print("use_lm: ", use_lm)
        # print("beam_size: ", beam_size)
        # print("lm_weight: ", self.text_encoder.lm_weight)
        # print("use_bpe: ", use_bpe)

        with torch.no_grad():
            outputs = self.model(**batch)
            batch.update(outputs)
            
            if use_beam_search:
                log_probs = outputs["log_probs"]
                probs = torch.exp(log_probs)
                probs = probs.cpu() # TRY
                
                batch_predictions = []
                for i in range(probs.size(0)):
                    seq_len = batch["log_probs_length"][i]
                    sequence_probs = probs[i, :seq_len]
                    

                    # if use_lm:
                    #     # Use beam search with LM
                    #     beam_results = self.text_encoder.ctc_beam_search_with_lm(sequence_probs, beam_size=10)
                    # else:
                    #     # Use regular beam search
                    #     beam_results = self.text_encoder.ctc_beam_search(sequence_probs, beam_size=10)
                    

                    # if use_lm:
                    #     self.text_encoder._initialize_language_model() ### TESTING

                    beam_results = self.text_encoder.ctc_beam_search(
                        sequence_probs.numpy(), beam_size=beam_size, use_lm=use_lm, debug=False
                    )


                    best_text = beam_results[0][0]  # First beam result, text
                    batch_predictions.append(best_text)
                
                batch["predictions"] = batch_predictions
                
            else:
                # Regular greedy decoding
                argmax_indices = torch.argmax(outputs["log_probs"], dim=-1)
                batch_predictions = []
                for i in range(len(argmax_indices)):
                    sequence_length = batch["log_probs_length"][i]
                    sequence = argmax_indices[i, :sequence_length]
                    
                    # Use appropriate decoding based on whether BPE is enabled
                    text = self.text_encoder.decode(sequence)
                    if self.text_encoder.use_bpe:
                        # Clean up BPE tokens if needed
                        text = self.text_encoder._clean_decoded_text(text)
                    
                    batch_predictions.append(text)
                
                batch["predictions"] = batch_predictions
                
            
            # Decode ground truth
            batch["ground_truth"] = []
            for encoded in batch["text_encoded"]:
                text = self.text_encoder.ctc_decode(encoded)
                if self.text_encoder.use_bpe:
                    text = self.text_encoder._clean_decoded_text(text)
                batch["ground_truth"].append(text)

            # Update metrics
            if metrics is not None:
                for met in self.metrics["inference"]:
                    metrics.update(met.name, met(**batch))

            # Save predictions
            if self.save_path is not None:
                batch_size = len(batch_predictions)
                current_id = batch_idx * batch_size
                
                for i in range(batch_size):
                    output_id = current_id + i
                    output = {
                        "prediction": batch_predictions[i],
                        "ground_truth": batch["ground_truth"][i],
                        "audio_path": batch["audio_path"][i]
                    }
                    torch.save(output, self.save_path / part / f"output_{output_id}.pth")

        return batch



    def _inference_part(self, part, dataloader, use_beam_search=True, use_lm=True):
    # def _inference_part(self, part, dataloader, use_beam_search=True, use_lm=False):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=f"{part} ({'beam search + LM' if use_lm else 'beam search' if use_beam_search else 'greedy'})",
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                    use_beam_search=use_beam_search, 
                    beam_size=self.cfg_trainer.beam_size,
                    use_lm=use_lm
                )

        return self.evaluation_metrics.result()
