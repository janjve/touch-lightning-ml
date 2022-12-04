import torch
import pytorch_lightning as pl
import numpy as np

from jvn_attention.xor import model, data


def train():
    xor_model = model.XORModel()

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    dataset = data.XORDataset(x,y)

    trainer = pl.Trainer(max_epochs=100, fast_dev_run=False)

    trainer.fit(xor_model, dataset)

    # Get the model's state dictionary
    state_dict = xor_model.state_dict()

    # Print the weights of the first linear layer
    print(state_dict['fc1.weight'])
    print(state_dict['fc1.bias'])

    # Print the weights of the second linear layer
    print(state_dict['fc2.weight'])
    print(state_dict['fc2.bias'])

    return xor_model


def save(xor_model):
    torch.save(xor_model.state_dict(), 'xor_model.pth')


def main():
    xor_model = train()
    save(xor_model)

if __name__ == "__main__":
    SystemExit(main())