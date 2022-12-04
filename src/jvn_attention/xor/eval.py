import torch
import pytorch_lightning as pl
import numpy as np

from jvn_attention.xor import model

def eval(xor_model):
    # Set the model in evaluation mode
    xor_model.eval()

    # Create some input data
    input_data = torch.Tensor([[0, 1], [1, 1], [0, 0], [1, 0]])

    # Run inference on the input data
    output = xor_model(input_data)

    # Round to get prediction
    predictions = torch.round(output)

    # Print the output
    result = torch.cat([input_data, output, predictions], axis=1)
    print(result)


def load():
    xor_model = model.XORModel()
    xor_model.load_state_dict(torch.load('xor_model.pth'))
    return xor_model


def main():
    xor_model = load()
    eval(xor_model)

if __name__ == "__main__":
    SystemExit(main())