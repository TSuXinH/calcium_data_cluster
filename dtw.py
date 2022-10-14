import numpy as np
from sklearn.manifold import TSNE
from tslearn.clustering import TimeSeriesKMeans
from tslearn.generators import random_walks
from sklearn.cluster import KMeans

from base_data_two_photo import *
from utils import normalize


f_test = f_dff[:, final_index[0][0]: final_index[39][1]]
f_test_n = normalize(f_test)
print('start cluster')
# cluster_model = TimeSeriesKMeans(n_clusters=4, max_iter=100, metric='dtw')
cluster_model = KMeans(n_clusters=4, max_iter=100)
res_cluster = cluster_model.fit_predict(f_test_n)
print('finish cluster')

dim_reduce_model = PCA(n_components=3)
res_visualization = dim_reduce_model.fit_transform(f_test_n)
print('finish dimension reduction')

visualize_cluster_3d(4, res_visualization, res_cluster)
visualize_cluster_2d(4, res_visualization, res_cluster)
visualize_sampled_spikes(f_dff, res_cluster, 4)

index = np.where(res_cluster == 0)[0]
for idx, item in enumerate(index):
    plt.subplot(len(index), 1, idx+1)
    plt.plot(f_test[item])
    plt.axis('off')
plt.show()

x = np.random.choice(index, 10)
visualize_firing_curve(f_dff[x], 0, 10, final_index)
