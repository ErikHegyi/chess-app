import numpy as np


def int_to_bits(integers: np.ndarray) -> np.ndarray:
    return np.array([[int(i) for i in bin(x)[2:].zfill(64)] for x in integers])