import os
import numpy as np
import torch
from torch.utils.data import DataLoader

class SyntheticDataset:
    """
    Dataset object to create a dataloader
    """

    def __init__(self, fpath, split="train", mode="step_by_step"):
        datafiles = {
            "train": os.path.join(fpath, "train_{}_corpus.npy".format(mode)),
            "test": os.path.join(fpath, "test_{}_corpus.npy".format(mode)),
            "train_heldout": os.path.join(
                fpath, "train_heldout_{}_corpus.npy".format(mode)
            ),
        }

        self.data = np.load(datafiles[split])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = torch.from_numpy(self.data[idx])
        dat, target = elem[:-1], elem[1:]
        return dat, target

class MappedSyntheticDataset:
    """
    Dataset wrapper that applies token mapping transformation to convert 
    synthetic vocab indices to model vocab indices.
    """

    def __init__(self, fpath, split="train", mode="step_by_step", token_map=None):
        datafiles = {
            "train": os.path.join(fpath, "train_{}_corpus.npy".format(mode)),
            "test": os.path.join(fpath, "test_{}_corpus.npy".format(mode)),
            "train_heldout": os.path.join(
                fpath, "train_heldout_{}_corpus.npy".format(mode)
            ),
        }

        self.data = np.load(datafiles[split])
        self.token_map = token_map
        
        if token_map is None:
            raise ValueError("token_map must be provided to MappedSyntheticDataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = torch.from_numpy(self.data[idx])
        dat, target = elem[:-1], elem[1:]
        
        dat = torch.from_numpy([self.token_map[int(idx)] for idx in dat])
        target = torch.from_numpy([self.token_map[int(idx)] for idx in target])
    
        return dat, target


def get_data_loader(dataset, batch_size, num_workers):
   
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    return dataloader

