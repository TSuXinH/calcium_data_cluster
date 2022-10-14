import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def z_score(data):
    return (data - np.mean(data)) / np.std(data)


def normalize(data):
    if len(data.shape) == 1:
        res = (data - np.min(data)) / (np.max(data) - np.min(data))
        res[np.isnan(res)] = 0
    else:
        res = (data - np.min(data, axis=-1).reshape(-1, 1)) / (np.max(data, axis=-1) - np.min(data, axis=-1)).reshape(-1, 1)
        res[np.isnan(res)] = 0
    return res
