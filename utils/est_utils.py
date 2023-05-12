import numpy as np


def cal_l1_distance(arr_x: np.ndarray, arr_y: np.ndarray) -> float:

    assert arr_x.shape == arr_y.shape, 'The shapes are different.'
    diff = float(np.sum(np.abs(arr_x - arr_y)))
    return diff