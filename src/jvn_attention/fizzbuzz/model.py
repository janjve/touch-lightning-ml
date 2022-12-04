import torch
from torch import nn
import pytorch_lightning as pt

class FizzBuzzModel(pt.LightningModule):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8,4) # 8-bit number
        self.fc2 = nn.Linear(4,4) # passthrough, fizz, buzz, fizzbuz

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())