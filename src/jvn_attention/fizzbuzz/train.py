import torch
import numpy as np
from jvn_attention.fizzbuzz import model, data
from jvn_attention.fizzbuzz.utils import as_bit_array

import pytorch_lightning as pl

def train():
    fizzbuzz_model = model.FizzBuzzModel()
    
    # example of passthrough, fizz, buzz, fizzbuzz example
    x = np.array([as_bit_array(1), as_bit_array(3), as_bit_array(5), as_bit_array(15)])
    y = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])

    fizzbuzz_data = data.FizzBuzzDataModule(x, y)
    
    trainer = pl.Trainer(max_epochs=100, fast_dev_run=False)
    trainer.fit(fizzbuzz_model, fizzbuzz_data)

    state_dict = fizzbuzz_model.state_dict()

    print(state_dict)

    torch.save(state_dict, "fizzbuzz_model.pth")


if __name__ == "__main__":
    SystemExit(train())