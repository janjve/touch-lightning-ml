import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from jvn_attention.xor import model, data

model = model.XORModel()

def train():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    dataset = data.XORDataset(x,y)

    trainer = pl.Trainer(max_epochs=10, fast_dev_run=True)

    trainer.fit(model, dataset)

    # Get the model's state dictionary
    state_dict = model.state_dict()

    # Print the weights of the first linear layer
    print(state_dict['fc1.weight'])
    print(state_dict['fc1.bias'])

    # Print the weights of the second linear layer
    print(state_dict['fc2.weight'])
    print(state_dict['fc2.bias'])

def inference():
    # Set the model in evaluation mode
    model.eval()

    # Create some input data
    input_data = torch.Tensor([[0, 1]])

    # Run inference on the input data
    output = model(input_data)

    # Print the output
    print(output)

train()
inference()