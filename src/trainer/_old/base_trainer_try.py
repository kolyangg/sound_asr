from abc import abstractmethod

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH

from src.datasets.sortagrad_dl import DataLoaderWrapper, SortaGradDataLoader

class BaseTrainer:
    """
    Base class for all trainers.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        metrics: dict,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        text_encoder,
        config,
        device: torch.device,
        dataloaders: dict,
        logger,
        writer,
        epoch_len: int = None,
        skip_oom: bool = True,
        batch_transforms: dict = None,
    ):
        """
        Initialization method.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config["trainer"]

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = self.cfg_trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.text_encoder = text_encoder
        self.batch_transforms = batch_transforms

        self.epoch_len = epoch_len

        # Define dataloaders
        self.train_dataloader = dataloaders["train"]

        # Check if train_dataloader is an instance of SortaGradDataLoader
        if isinstance(self.train_dataloader, SortaGradDataLoader):
            print("Using SortaGradDataLoader.")
            # No need to wrap with inf_loop since SortaGradDataLoader is infinite
            self.train_dataloader_wrapper = self.train_dataloader
            if self.epoch_len is None:
                # Set epoch_len manually to the desired number of batches (e.g., 200)
                self.epoch_len = config.trainer.epoch_len
                print(f"Epoch length set to: {self.epoch_len}")
        else:
            print("Using standard DataLoader with inf_loop.")
            # Wrap standard DataLoader with inf_loop to make it infinite
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.train_dataloader_wrapper = self.train_dataloader
            if self.epoch_len is None:
                # Set epoch_len based on the desired number of batches (e.g., 200)
                self.epoch_len = config.trainer.epoch_len
                print(f"Epoch length set to: {self.epoch_len}")

        print(f"Final epoch length: {self.epoch_len}")

        # Define evaluation dataloaders
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # Define epochs
        self._last_epoch = 0  # Required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs

        # Configuration to monitor model performance and save best
        self.save_period = self.cfg_trainer.save_period  # Checkpoint each save_period epochs
        self.monitor = self.cfg_trainer.get("monitor", "off")  # Format: "mnt_mode mnt_metric"

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"], "Monitor mode must be 'min' or 'max'."

            self.mnt_best = float('inf') if self.mnt_mode == "min" else -float('inf')
            self.early_stop = self.cfg_trainer.get("early_stop", float('inf'))
            if self.early_stop <= 0:
                self.early_stop = float('inf')

        # Setup visualization writer instance
        self.writer = writer

        # Define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics["inference"]],
            writer=self.writer,
        )

        # Define checkpoint directory and initialize if required
        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))
    
    
    
    
    
    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            # print logged information to the screen
            for key, value in logs.items():
                self.logger.info(f"    {key:15s}: {value}")

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

            if stop_process:  # early_stop
                break

    # def _train_epoch(self, epoch):
    #     """
    #     Training logic for an epoch, including logging and evaluation on
    #     non-train partitions.

    #     Args:
    #         epoch (int): current training epoch.
    #     Returns:
    #         logs (dict): logs that contain the average loss and metric in
    #             this epoch.
    #     """
    #     self.is_train = True
    #     self.model.train()
    #     self.train_metrics.reset()
    #     self.writer.set_step((epoch - 1) * self.epoch_len)
    #     self.writer.add_scalar("epoch", epoch)
        
    #     # Get the appropriate dataloader for this epoch
    #     train_dataloader = self.train_dataloader_wrapper.get_dataloader(epoch) # NEW ADDED!!!
        
    #     for batch_idx, batch in enumerate(
    #         # tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
    #         tqdm(train_dataloader, desc="train", total=self.epoch_len) # NEW ADDED!!!
    #     ):
    #         # print('train_dataloader')
    #         # print(self.train_dataloader)
    #         # print(self.train_dataloader.dataset[0])
    #         try:
    #             batch = self.process_batch(
    #                 batch,
    #                 metrics=self.train_metrics,
    #             )
    #         except torch.cuda.OutOfMemoryError as e:
    #             if self.skip_oom:
    #                 self.logger.warning("OOM on batch. Skipping batch.")
    #                 torch.cuda.empty_cache()  # free some memory
    #                 continue
    #             else:
    #                 raise e

    #         self.train_metrics.update("grad_norm", self._get_grad_norm())

    #         # log current results
    #         if batch_idx % self.log_step == 0:
    #             self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
    #             self.logger.debug(
    #                 "Train Epoch: {} {} Loss: {:.6f}".format(
    #                     epoch, self._progress(batch_idx), batch["loss"].item()
    #                 )
    #             )
    #             self.writer.add_scalar(
    #                 "learning rate", self.lr_scheduler.get_last_lr()[0]
    #             )
    #             self._log_scalars(self.train_metrics)
    #             self._log_batch(batch_idx, batch)
    #             # we don't want to reset train metrics at the start of every epoch
    #             # because we are interested in recent train metrics
    #             last_train_metrics = self.train_metrics.result()
    #             self.train_metrics.reset()
    #         if batch_idx + 1 >= self.epoch_len:
    #             break

    #     logs = last_train_metrics

    #     # Run val/test
    #     for part, dataloader in self.evaluation_dataloaders.items():
    #         val_logs = self._evaluation_epoch(epoch, part, dataloader)
    #         logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

    #     return logs
    
    def _train_epoch(self, epoch: int) -> dict:
        """
        Training logic for a single epoch, including logging and evaluation.

        Args:
            epoch (int): Current training epoch.
        Returns:
            logs (dict): Dictionary containing average loss and metrics for this epoch.
        """
        print(f"\nStarting training for epoch {epoch}")
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)
        
        # Create an iterator from the train_dataloader_wrapper
        train_dataloader_iter = iter(self.train_dataloader_wrapper)
        print(f"Training dataloader for epoch {epoch} has {self.epoch_len} batches")
        
        last_train_metrics = {}

        for batch_idx in range(self.epoch_len):
            try:
                batch = next(train_dataloader_iter)
            except StopIteration:
                # This should not happen as DataLoaders are either infinite or properly handled
                print("Dataloader exhausted unexpectedly.")
                break

            # print(f"Epoch {epoch}, Batch {batch_idx + 1}/{self.epoch_len}")
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    print("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  # Free some memory
                    continue
                else:
                    raise e

            self.train_metrics.update("grad_norm", self._get_grad_norm())

            # Log current results
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                print(
                    f"Train Epoch: {epoch} [{self._progress(batch_idx)}] Loss: {batch['loss'].item():.6f}"
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                # Reset train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.epoch_len:
                print(f"Reached epoch length: {self.epoch_len} batches")
                break

        print(f"Completed training for epoch {epoch}")

        logs = last_train_metrics

        # Run validation/test
        for part, dataloader in self.evaluation_dataloaders.items():
            print(f"Starting evaluation on {part} for epoch {epoch}")
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})
            print(f"Completed evaluation on {part} for epoch {epoch}")

        return logs


    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(
                batch_idx, batch, part
            )  # log only the last batch during inference

        return self.evaluation_metrics.result()

    def _monitor_performance(self, logs, not_improved_count):
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Args:
            logs (dict): logs after training and evaluating the model for
                an epoch.
            not_improved_count (int): the current number of epochs without
                improvement.
        Returns:
            best (bool): if True, the monitored metric has improved.
            stop_process (bool): if True, stop the process (early stopping).
                The metric did not improve for too much epochs.
            not_improved_count (int): updated number of epochs without
                improvement.
        """
        best = False
        stop_process = False
        if self.mnt_mode != "off":
            try:
                # check whether model performance improved or not,
                # according to specified metric(mnt_metric)
                if self.mnt_mode == "min":
                    improved = logs[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == "max":
                    improved = logs[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                self.logger.warning(
                    f"Warning: Metric '{self.mnt_metric}' is not found. "
                    "Model performance monitoring is disabled."
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                stop_process = True
        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch
        
        # for key, value in batch.items():
        #     if value is not None and hasattr(value, 'to'):
        #         batch[key] = value.to(self.device)
        # return batch

    def transform_batch(self, batch):
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        # do batch transforms on device
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch

    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _progress(self, batch_idx):
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
        """
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch, save_best=True, only_best=True):
    # def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
            or checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)