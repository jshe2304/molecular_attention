import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class GaussianSamplingDataset(Dataset):
    def __init__(self, x, y, var):
        self.x = x
        self.y = y
        self.std = torch.sqrt(var)
        
        assert self.x.shape[0] == self.y.shape[0] == self.std.shape[0]
        self.n = self.x.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        std = self.std[idx]
        return self.x[idx], self.y[idx] + std * torch.randn_like(std)

class SiameseDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
        assert self.x.shape[0] == self.y.shape[0]
        self.n = self.x.shape[0]

    def __len__(self):
        return self.n ** 2

    def __getitem__(self, idx):
        
        idx1 = idx % self.n # repeat
        idx2 = idx // self.n # repeat_interleave
        return self.x[idx1], self.x[idx2], self.y[idx1], self.y[idx2]

class QM9Dataset(Dataset):
    def __init__(self, smiles, d, y):

        if type(smiles) is str:
            smiles = pd.read_csv(smiles).to_numpy().squeeze()
        if type(d) is str:
            d = np.load(d).astype('float32')
        if type(y) is str:
            y = pd.read_csv(y).to_numpy().astype('float32').squeeze()

        assert len(smiles) == len(d) == len(y)
        
        self.smiles = smiles
        self.d = d
        self.y = y
        
        self.n_properties = y.shape[1]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.d[idx], self.y[idx]

class 