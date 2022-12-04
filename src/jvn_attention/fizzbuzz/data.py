import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pt
import numpy as np

class FizzBuzzDataModule(pt.LightningDataModule):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(
                torch.Tensor(self.x),
                torch.Tensor(self.y),
                ),
            batch_size=4,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(TensorDataset(
                torch.Tensor(self.x),
                torch.Tensor(self.y),
                ),
            batch_size=4,
        )