import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from base_data_two_photo import *
from utils import visualize_3d_cluster, visualize_sampled_spikes
from autoencoder import CustomDataset, train, ZScore, test, normalize, AETest1, cal_err


def show_reconstruction(f_dff, recon, idx, trans):
    raw = f_dff[idx]
    raw = trans(raw)
    plt.plot(raw, 'g', label='raw')
    plt.plot(recon[idx], 'r', label='recovered')
    plt.legend()
    plt.show(block=True)


f_train = f_mean_stim

args = edict
# network structure
args.input_dim = f_train.shape[-1]
args.latent_dim = 2
args.hidden_dim = [256, 64]
args.drop_ratio = .15
# network hyperparameters
args.lr = 1e-4
args.wd = 1e-5
args.max_epoch = 200
args.batch_size = 256
# other configurations
args.device = 'cuda'
args.tb_path = './tensorboard'
args.save_path = ''
z_score = ZScore(f_train)

model = AETest1(args).to(args.device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
dataset = CustomDataset(f_train, normalize)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

print(args.input_dim)
print('training')
loss_list = train(model, loader, criterion, optimizer, args)
print('training done')
plt.plot(loss_list)
plt.title('non-linear autoencoder loss-epoch')
plt.show(block=True)
print('testing')
recon_res, rdc_res = test(model, test_loader, args)
print('testing done')
x = np.random.randint(0, len(f_train))
show_reconstruction(f_train, recon_res, x, trans=normalize)
f_pro = normalize(f_train)
print(cal_err(recon_res, f_pro))

clusters = 3
k_means = KMeans(n_clusters=clusters, init='k-means++', max_iter=1000)
res_kmeans = k_means.fit_predict(rdc_res)

dim_red_tsne = TSNE(n_components=3)
res_tsne = dim_red_tsne.fit_transform(rdc_res)

visualize_3d_cluster(clusters, res_tsne, res_kmeans, title='autoencoder normal')
visualize_sampled_spikes(f_train, res_kmeans, clusters, show_all=True)
