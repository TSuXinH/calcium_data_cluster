import numpy as np


def z_score(data):
    if len(data.shape) == 1:
        return (data - np.mean(data)) / np.std(data) if np.std(data).item() != 0 else data
    else:
        res = (data - np.mean(data, axis=-1).reshape(-1, 1)) / np.std(data, axis=-1).reshape(-1, 1)
        res[np.isnan(res)] = 0
        return res


def normalize(data):
    if len(data.shape) == 1:
        res = (data - np.min(data)) / (np.max(data) - np.min(data))
        res[np.isnan(res)] = 0
    else:
        res = (data - np.min(data, axis=-1).reshape(-1, 1)) / (np.max(data, axis=-1) - np.min(data, axis=-1)).reshape(-1, 1)
        res[np.isnan(res)] = 0
    return res
