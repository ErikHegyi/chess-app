import numpy as np


def bits_to_int(bits: np.ndarray) -> np.ndarray:
    return np.array([int(''.join(map(str, a)), 2) for a in bits])