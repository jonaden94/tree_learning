import os
import torch
import numpy as np
import torch.utils.data as data
from utils.logger import *
from utils.config import read_yaml

VAR = "treeset"

"""
Dataset for forest plots/single trees. 

For self supervised training: use whole dataset?!

for supervised training: How to split the data? On a specific plot? Or one plot for training and another for testing?
If on a specific plot: generate region based train test split


TBD

STRUCTURE:
TBD

"""

class Treeset(data.Dataset):
    def __init__(self, config):
        self.plot_folders = config.plot_folders
        self.npoints = config.npoints
        self.include_labels = config.include_labels
        self.mode = config.mode
        self.normalization = config.normalization
        self.transform = config.transformations
        self.normalization_pars = np.array(config.normalization_pars)
        self.permutation = np.arange(self.npoints)
        self.in_memory = config.in_memory

        self.samples_list = []
        for folder in self.plot_folders:
            files = os.listdir(folder)
            files = [os.path.join(folder, file) for file in files]
            self.samples_list += files
        if self.in_memory:
            self.samples_list = [np.load(sample).astype(np.float32) for sample in self.samples_list]
        print_log(f'[DATASET] sample out {self.npoints} points', logger = VAR)
        print_log(f'[DATASET] {len(self.samples_list)} samples from {len(self.plot_folders)} forest plots were loaded', logger = VAR)

    def normalize(self, pc):
        """ files were already centered, only correct by axis values and correct z axis to have data in unit cube"""
        pc = pc / self.normalization_pars
        pc[:, 2] = pc[:, 2] - 1
        return pc

    def random_sample(self, pc, num):
        if len(pc) < num:
            choice = np.random.choice(len(pc), num, replace=True)
        else:
            choice = np.random.choice(len(pc), num, replace=False)
        return  pc[choice]

    def __getitem__(self, idx):
        if self.in_memory:
            return self.in_memory_getitem(idx)
        else:
            return self.out_memory_getitem(idx)

    def in_memory_getitem(self, idx):
        data = self.samples_list[idx]
        return self.getitem_finish(data)

    def out_memory_getitem(self, idx):
        path = self.samples_list[idx]
        data = np.load(path).astype(np.float32)
        return self.getitem_finish(data)
        
    def getitem_finish(self, data):
        sample = self.random_sample(data, self.npoints)
        points, labels = sample[:, :3], sample[:, 3]
        if self.normalization:
            points = self.normalize(points)
        if self.transform:
            points, labels = self.transform(points, labels)
        points, labels = torch.from_numpy(points).float(), torch.from_numpy(labels).int()
        if not self.include_labels:
            labels = None
        if self.mode == "eval":
            return points, labels, data
        else:
            return points, labels

    def get_loader(self, args=None):
        if args: 
            loader = data.DataLoader(self, *args)
        else:
            loader = data.DataLoader(self)
        return loader

    def __len__(self):
        return len(self.samples_list)

if __name__ == '__main__':
    import sys
    config = read_yaml("datasets/treeset_config.yaml")
    print(config)
    dataset = Treeset(config)
    print(dataset[0][0].shape)
    loader = dataset.get_loader()
    out = next(iter(loader))
    print(out[0].shape, out[1].shape)
