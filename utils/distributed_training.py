import os
from typing import *

import torch

# Distributed training imports
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler, Dataset, Sampler
from torch.nn.parallel import DistributedDataParallel
from operator import itemgetter
from torch import autocast # type: ignore
import traceback


class DatasetFromSampler(Dataset):
    """
    Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler

    https://github.com/singlaayush/MINIT/blob/main/distributed_sampler_wrapper.py
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler) # type: ignore


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.

    https://github.com/singlaayush/MINIT/blob/main/distributed_sampler_wrapper.py
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int],
        rank: Optional[int],
        shuffle: bool,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        if len(self.dataset) == 1:
            return iter(list(self.sampler))
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes)) # type: ignore


def setup_distributed(rank: int, world_size: int, master_port: str):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port

    torch.cuda.set_device(rank)

    # Initialize distributed training.
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)