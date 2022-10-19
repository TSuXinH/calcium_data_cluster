import numpy as np
from sklearn.decomposition import PCA


def choose_pca_component(mat, thr):
    assert 0 < thr <= 1, 'the threshold should be between 0 and 1.'
    pca = PCA(n_components=np.min(mat.shape).item())
    _ = pca.fit_transform(mat)
    var_ratio = pca.explained_variance_ratio_
    var_ratio_cumsum = np.cumsum(var_ratio)
    return np.where(var_ratio_cumsum > thr)[0][0] + 1
