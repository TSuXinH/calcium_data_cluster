from scipy.optimize import linear_sum_assignment
import numpy as np


def align_label(raw, next):
    length = np.max(raw).item() + 1
    G = np.zeros((length, length))
    for x in range(length):
        idx_raw = raw == x
        for y in range(length):
            idx_next = next == y
            G[x, y] = np.sum((idx_raw & idx_next).astype(np.int_))
    _, new_col = linear_sum_assignment(G)
    return new_col
