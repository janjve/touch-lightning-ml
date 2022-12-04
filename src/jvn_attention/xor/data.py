import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

class XORDataset(pl.LightningDataModule):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.Tensor(self.x),
                torch.Tensor(self.y)
            ),
            batch_size=4,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.Tensor(self.x),
                torch.Tensor(self.y)
            ),
            batch_size=4
        )


