import torch
from jvn_attention.fizzbuzz.model import FizzBuzzModel
from jvn_attention.fizzbuzz.utils import as_bit_array
import numpy as np

def eval():
    fizzbuzz_model = FizzBuzzModel()
    fizzbuzz_model.load_state_dict(torch.load("fizzbuzz_model.pth"))
    fizzbuzz_model.eval()

    input_data = torch.Tensor([as_bit_array(1)])

    output = fizzbuzz_model(input_data)

    print(output)

if __name__ == "__main__":
    SystemExit(eval())