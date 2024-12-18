# src/datasets/sortagrad_dl.py

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
import random

# class SortaGradDataLoader:
#     """
#     A wrapper for SortaGrad: sort dataset by length for the first N epochs,
#     then shuffle randomly for subsequent epochs.
#     """

#     def __init__(
#         self,
#         dataset,
#         batch_size,
#         collate_fn,
#         num_workers,
#         pin_memory,
#         num_epochs_sorted=1,
#         drop_last=True,
#         shuffle=True,
#         **kwargs
#     ):
#         """
#         Args:
#             dataset (Dataset): The dataset to load.
#             batch_size (int): Batch size for training.
#             collate_fn (callable): Function to collate data into batches.
#             num_workers (int): Number of workers for DataLoader.
#             pin_memory (bool): Whether to pin memory in DataLoader.
#             num_epochs_sorted (int): Number of epochs to sort by length.
#             drop_last (bool): Whether to drop the last incomplete batch.
#             shuffle (bool): Whether to shuffle the dataset (overridden by SortaGrad).
#             **kwargs: Additional DataLoader arguments.
#         """
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.collate_fn = collate_fn
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.num_epochs_sorted = num_epochs_sorted
#         self.drop_last = drop_last
#         self.current_epoch = 0
#         self.kwargs = kwargs

#     def compute_length(self, spectrogram, threshold=1e-3):
#         """
#         Compute the length of the spectrogram by finding the last non-zero frame.

#         Args:
#             spectrogram (Tensor): Spectrogram tensor of shape (F, T).

#         Returns:
#             int: The length (number of time steps) of the spectrogram.
#         """
#         # Assuming spectrogram is a 2D tensor (F, T)
#         # Compute the sum over frequency bins to get energy per time step
#         energy = torch.sum(spectrogram, dim=0)
#         # Find indices where energy is greater than a small threshold
#         non_zero = energy > threshold
#         if non_zero.any():
#             length = non_zero.nonzero(as_tuple=False).max().item() + 1
#         else:
#             length = 0
#         return length

#     def __iter__(self):
#         if self.current_epoch < self.num_epochs_sorted:
#             # Sort dataset by spectrogram length
#             sorted_indices = sorted(
#                 range(len(self.dataset)),
#                 key=lambda i: self.compute_length(self.dataset[i]['spectrogram'])
#             )
#             sampler = SubsetRandomSampler(sorted_indices)
#             shuffle_flag = False
#             sampler_arg = sampler
#             print(f"Epoch {self.current_epoch + 1}: Sorting dataset by computed spectrogram lengths")
#         else:
#             # Shuffle dataset randomly
#             sampler = RandomSampler(self.dataset)
#             shuffle_flag = False  # Shuffle is controlled by sampler
#             sampler_arg = sampler
#             print(f"Epoch {self.current_epoch + 1}: Shuffling dataset randomly")

#         # Create the DataLoader for the current epoch
#         dataloader = DataLoader(
#             self.dataset,
#             batch_size=self.batch_size,
#             collate_fn=self.collate_fn,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             drop_last=self.drop_last,
#             sampler=sampler_arg,
#             shuffle=shuffle_flag,
#             **self.kwargs
#         )

#         self.current_epoch += 1  # Move to next epoch
#         return iter(dataloader)

#     def __len__(self):
#         return len(self.dataset) // self.batch_size
class DataLoaderWrapper:
    def __init__(self, dataloader):
        """
        Wraps a DataLoader instance. If a SortaGradDataLoader is passed, it will manage epoch-based DataLoaders.

        Args:
            dataloader: Either a standard DataLoader or a SortaGradDataLoader.
        """
        self.dataloader = dataloader
        self.needs_epoch = isinstance(dataloader, SortaGradDataLoader)
        
        if self.needs_epoch:
            print("DataLoaderWrapper detected SortaGradDataLoader.")
        else:
            print("DataLoaderWrapper detected standard DataLoader.")

    def get_dataloader(self, epoch=None):
        """
        Retrieves the appropriate DataLoader for the current epoch.

        Args:
            epoch (int, optional): The current epoch number.

        Returns:
            DataLoader: The DataLoader instance to be used for this epoch.
        """
        if self.needs_epoch:
            # For SortaGradDataLoader, retrieve the DataLoader for the current epoch
            dataloader_iter = self.dataloader.__iter__()
            return dataloader_iter
        else:
            # For standard DataLoader, return the DataLoader itself
            return self.dataloader

    def __iter__(self):
        return iter(self.get_dataloader())

    def __len__(self):
        return len(self.dataloader)




from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, RandomSampler
from itertools import repeat
import torch


class SortaGradDataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        collate_fn,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_epochs_sorted: int = 1,
        drop_last: bool = True,
        shuffle: bool = True,
        **kwargs
    ):
        """
        Custom DataLoader that sorts data for a specified number of epochs before shuffling.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_epochs_sorted = num_epochs_sorted
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.kwargs = kwargs

        self.current_epoch = 0  # Tracks the current epoch

        print("Initializing SortaGradDataLoader.")

    def __iter__(self):
        """
        Creates an infinite iterator over the DataLoader, sorting the dataset
        for the first `num_epochs_sorted` epochs before shuffling.
        """
        while True:
            if self.current_epoch < self.num_epochs_sorted:
                # Sort the dataset based on a custom criterion (e.g., spectrogram length)
                sorted_indices = sorted(
                    range(len(self.dataset)),
                    key=lambda i: self.compute_length(self.dataset[i]["spectrogram"])
                )
                sampler = SubsetRandomSampler(sorted_indices)
                print(f"Epoch {self.current_epoch + 1}: Sorting dataset by length.")
            else:
                # Shuffle the dataset randomly
                sampler = RandomSampler(self.dataset)
                print(f"Epoch {self.current_epoch + 1}: Shuffling dataset randomly.")

            # Create a new DataLoader instance for the current epoch
            dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                sampler=sampler,
                **self.kwargs
            )

            print(f"SortaGradDataLoader for epoch {self.current_epoch + 1} created with {len(dataloader)} batches.")
            self.current_epoch += 1

            # Yield all batches from the current DataLoader
            yield from dataloader

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        if self.drop_last:
            length = len(self.dataset) // self.batch_size
        else:
            length = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        print(f"SortaGradDataLoader length: {length} batches.")
        return length

    @staticmethod
    def compute_length(spectrogram: torch.Tensor, threshold: float = 1e-3) -> int:
        """
        Computes the length of a spectrogram based on a threshold.
        """
        energy = torch.sum(spectrogram, dim=0)
        non_zero = energy > threshold
        if non_zero.any():
            return non_zero.nonzero(as_tuple=False).max().item() + 1
        return 0
