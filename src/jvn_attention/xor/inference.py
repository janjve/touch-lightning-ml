import torch

from dataclasses import dataclass
from jvn_attention.xor import model


@dataclass
class InputRow:
    x1: int
    x2: int


@dataclass
class Prediction:
    predict: int
    probability: int


class XORPredictor:
    def __init__(self):
        self.model = model.XORModel()
        self.model.load_state_dict(torch.load("xor_model.pth"))

    def create_tensor(self, data: InputRow) -> torch.Tensor:
        return torch.Tensor([[data.x1, data.x2]])

    def __call__(self, data: InputRow) -> Prediction:
        input_tensor = self.create_tensor(data)
        output_tensor = self.model(input_tensor)
        pred_tensor = torch.round(output_tensor)

        pred = int(pred_tensor.tolist()[0][0])
        prop = output_tensor.tolist()[0][0]

        return Prediction(predict=pred, probability=prop)
