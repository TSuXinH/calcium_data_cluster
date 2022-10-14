import matplotlib.pyplot as plt

from base_data_two_photo import *

pca_2d = PCA(n_components=2, whiten=True)
pca_2d_res = pca_2d.fit_transform(f_long_stim)
exp_var = pca_2d.explained_variance_ratio_
print('exp_var: ', exp_var)

# # pick the best cluster number
inertia_list = []
for cluster in range(2, len(color_map) + 2):
    cluster_kmeans = KMeans(n_clusters=cluster, init='k-means++', max_iter=1000)
    res_kmeans = cluster_kmeans.fit_predict(pca_2d_res)
    inertia_list.append(cluster_kmeans.inertia_)
plt.plot(list(range(len(inertia_list))), inertia_list)
plt.xticks(list(range(len(inertia_list))), list(range(2, len(inertia_list) + 2)))
plt.title('cluster inertia via number of cluster centers')
plt.show(block=True)

clusters = 3
k_means = KMeans(n_clusters=clusters, init='k-means++', max_iter=1000)
res_kmeans = k_means.fit_predict(pca_2d_res)

for item in range(clusters):
    index = np.where(res_kmeans == item)[0]
    tmp_result = pca_2d_res[index]
    color = color_map[item]
    plt.scatter(tmp_result[:, 0], tmp_result[:, 1], c=color, label='cluster {}'.format(item), s=6)
plt.title('PCA + K Means cluster')
plt.legend()
plt.show(block=True)

for item in range(clusters):
    index = np.where(res_kmeans == item)[0]
    print(len(index))
    tmp_f = f_dff[index]
    if len(index) > 10:
        x = np.random.randint(0, len(tmp_f)-10)
        visualize_firing_curve(tmp_f, x, 10, final_index)
    else:
        visualize_firing_curve(tmp_f, 0, len(index), final_index)
