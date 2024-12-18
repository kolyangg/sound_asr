from itertools import repeat

from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed

from src.datasets.sortagrad_dl import SortaGradDataLoader


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


# def get_dataloaders(config, text_encoder, device):
#     """
#     Create dataloaders for each of the dataset partitions.
#     Also creates instance and batch transforms.

#     Args:
#         config (DictConfig): hydra experiment config.
#         text_encoder (CTCTextEncoder): instance of the text encoder
#             for the datasets.
#         device (str): device to use for batch transforms.
#     Returns:
#         dataloaders (dict[DataLoader]): dict containing dataloader for a
#             partition defined by key.
#         batch_transforms (dict[Callable] | None): transforms that
#             should be applied on the whole batch. Depend on the
#             tensor name.
#     """
#     # transforms or augmentations init
#     batch_transforms = instantiate(config.transforms.batch_transforms)
#     move_batch_transforms_to_device(batch_transforms, device)

#     # dataloaders init
#     dataloaders = {}
#     for dataset_partition in config.datasets.keys():
#         # dataset partition init
#         dataset = instantiate(
#             config.datasets[dataset_partition], text_encoder=text_encoder
#         )  # instance transforms are defined inside

#         assert config.dataloader.batch_size <= len(dataset), (
#             f"The batch size ({config.dataloader.batch_size}) cannot "
#             f"be larger than the dataset length ({len(dataset)})"
#         )

#         partition_dataloader = instantiate(
#             config.dataloader,
#             dataset=dataset,
#             collate_fn=collate_fn,
#             drop_last=(dataset_partition == "train"),
#             shuffle=(dataset_partition == "train"),
#             worker_init_fn=set_worker_seed,
#         )
#         dataloaders[dataset_partition] = partition_dataloader

#     return dataloaders, batch_transforms


def get_dataloaders(config, text_encoder, device):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        text_encoder (CTCTextEncoder): instance of the text encoder
            for the datasets.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # Initialize transforms or augmentations
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # Initialize dataloaders
    dataloaders = {}

    # Check if 'train' key exists in config.dataloader
    # If not, assume config.dataloader is a single DataLoader config
    if isinstance(config.dataloader, dict) and 'train' in config.dataloader:
        # Scenario 1: Multiple DataLoaders (e.g., 'train', 'validation')
        for dataset_partition in config.datasets.keys():
            # Instantiate dataset
            dataset = instantiate(
                config.datasets[dataset_partition],
                text_encoder=text_encoder
            )

            # Retrieve batch_size for the partition
            batch_size = config.dataloader[dataset_partition].get('batch_size', 40)  # default to 40

            # Ensure batch_size does not exceed dataset length
            assert batch_size <= len(dataset), (
                f"The batch size ({batch_size}) cannot "
                f"be larger than the dataset length ({len(dataset)})"
            )

            # Instantiate DataLoader
            partition_dataloader = instantiate(
                config.dataloader[dataset_partition],
                dataset=dataset,
                collate_fn=collate_fn,
                drop_last=(dataset_partition == "train"),
                shuffle=(dataset_partition == "train"),
                worker_init_fn=set_worker_seed,
            )

            # For non-training partitions, wrap with inf_loop
            if dataset_partition != "train":
                partition_dataloader = inf_loop(partition_dataloader)

            dataloaders[dataset_partition] = partition_dataloader

    else:
        # Scenario 2: Single DataLoader config (assumed for 'train')
        # Instantiate 'train' dataset
        try:
            train_dataset = instantiate(config.datasets['train'], text_encoder=text_encoder)
        except KeyError:
            raise KeyError("The 'train' dataset is not defined in config.datasets.")

        # Override batch_size to 5 for 'train' to achieve 200 batches per epoch
        train_dataloader = SortaGradDataLoader(
            dataset=train_dataset,
            batch_size=5,  # Set to 5 to get 200 batches with 1000 samples
            collate_fn=collate_fn,
            num_workers=config.dataloader.get('num_workers', 16),
            pin_memory=config.dataloader.get('pin_memory', True),
            num_epochs_sorted=1,
            drop_last=True,
            shuffle=True
        )

        dataloaders['train'] = train_dataloader

        # Instantiate 'validation' dataset
        try:
            validation_dataset = instantiate(config.datasets['val'], text_encoder=text_encoder)
        except KeyError:
            raise KeyError("The 'validation' dataset is not defined in config.datasets.")

        # Instantiate 'validation' DataLoader using the same config but adjust parameters
        validation_dataloader = instantiate(
            config.dataloader,
            dataset=validation_dataset,
            collate_fn=collate_fn,
            batch_size=40,  # Set to 40 as per original config
            drop_last=False,
            shuffle=False,
            worker_init_fn=set_worker_seed,
        )

        # Wrap 'validation' DataLoader with inf_loop for continuous iteration
        validation_dataloader = inf_loop(validation_dataloader)

        dataloaders['validation'] = validation_dataloader

    return dataloaders, batch_transforms