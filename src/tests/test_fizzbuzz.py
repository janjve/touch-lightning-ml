import pytest
from jvn_attention.fizzbuzz import utils


@pytest.mark.parametrize("input_num, expected_array", [
    (0, [0,0,0,0,0,0,0,0]),
    (1, [0,0,0,0,0,0,0,1]),
    (7, [0,0,0,0,0,1,1,1]),
    (8, [0,0,0,0,1,0,0,0]),
    (255, [1,1,1,1,1,1,1,1]),
])
def test_as_bit_array(input_num, expected_array):
    assert utils.as_bit_array(input_num) == expected_array
