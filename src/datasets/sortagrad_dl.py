# src/datasets/sortagrad_dl.py

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
import random

class SortaGradDataLoader:
    """
    A wrapper for SortaGrad: sort dataset by length for the first N epochs,
    then shuffle randomly for subsequent epochs.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        collate_fn,
        num_workers,
        pin_memory,
        num_epochs_sorted=1,
        drop_last=True,
        shuffle=True,
        **kwargs
    ):
        """
        Args:
            dataset (Dataset): The dataset to load.
            batch_size (int): Batch size for training.
            collate_fn (callable): Function to collate data into batches.
            num_workers (int): Number of workers for DataLoader.
            pin_memory (bool): Whether to pin memory in DataLoader.
            num_epochs_sorted (int): Number of epochs to sort by length.
            drop_last (bool): Whether to drop the last incomplete batch.
            shuffle (bool): Whether to shuffle the dataset (overridden by SortaGrad).
            **kwargs: Additional DataLoader arguments.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_epochs_sorted = num_epochs_sorted
        self.drop_last = drop_last
        self.current_epoch = 0
        self.kwargs = kwargs

    def compute_length(self, spectrogram, threshold=1e-3):
        """
        Compute the length of the spectrogram by finding the last non-zero frame.

        Args:
            spectrogram (Tensor): Spectrogram tensor of shape (F, T).

        Returns:
            int: The length (number of time steps) of the spectrogram.
        """
        # Assuming spectrogram is a 2D tensor (F, T)
        # Compute the sum over frequency bins to get energy per time step
        energy = torch.sum(spectrogram, dim=0)
        # Find indices where energy is greater than a small threshold
        non_zero = energy > threshold
        if non_zero.any():
            length = non_zero.nonzero(as_tuple=False).max().item() + 1
        else:
            length = 0
        return length

    def __iter__(self):
        if self.current_epoch < self.num_epochs_sorted:
            # Sort dataset by spectrogram length
            sorted_indices = sorted(
                range(len(self.dataset)),
                key=lambda i: self.compute_length(self.dataset[i]['spectrogram'])
            )
            sampler = SubsetRandomSampler(sorted_indices)
            shuffle_flag = False
            sampler_arg = sampler
            print(f"Epoch {self.current_epoch + 1}: Sorting dataset by computed spectrogram lengths")
        else:
            # Shuffle dataset randomly
            sampler = RandomSampler(self.dataset)
            shuffle_flag = False  # Shuffle is controlled by sampler
            sampler_arg = sampler
            print(f"Epoch {self.current_epoch + 1}: Shuffling dataset randomly")

        # Create the DataLoader for the current epoch
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            sampler=sampler_arg,
            shuffle=shuffle_flag,
            **self.kwargs
        )

        self.current_epoch += 1  # Move to next epoch
        return iter(dataloader)

    def __len__(self):
        return len(self.dataset) // self.batch_size
