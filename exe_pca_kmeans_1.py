import matplotlib.pyplot as plt

from base_data_two_photo import *

component = min(f_long_stim.shape)
pca1 = PCA(n_components=component)
_ = pca1.fit_transform(f_long_stim)
exp_var = pca1.explained_variance_ratio_
sum = 0
key = 0
for idx in range(len(exp_var)):
    sum += exp_var[idx]
    if sum > .9:
        key = idx
        break

clusters = 4
comp = key
pca2 = PCA(n_components=comp)
pca_res = pca2.fit_transform(f_long_stim)

k_means = KMeans(n_clusters=clusters, init='k-means++', max_iter=1000)
res_kmeans = k_means.fit_predict(pca_res)


pca2 = PCA(n_components=2)
pca_res = pca2.fit_transform(pca_res)

visualize_cluster_2d(clusters, pca_res, res_kmeans, title='PCA-kmeans')
visualize_sampled_spikes(f_dff, res_kmeans, clusters)


