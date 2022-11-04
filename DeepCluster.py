import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans


class DeepClusterNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=args.conv_ker, stride=args.stride)
        self.act = nn.LeakyReLU(.1, True)
        self.mlp = nn.Sequential(
            nn.Linear(args.input_dim, args.hidden_dim),
            nn.LeakyReLU(.1, True),
            nn.Dropout(.2),
            nn.Linear(args.hidden_dim, args.latent_dim),
        )
        self.cls = nn.Sequential(
            nn.Linear(args.latent_dim, args.clus_num)
        )

    def forward(self, x):
        if self.args.mode == 'linear':
            lat = self.mlp(x)
            out = self.cls(lat)
            return lat, out
        elif self.args.mode == 'conv':
            bs, length = x.shape
            x = x.reshape(bs, 1, length)
            x = self.conv1d(x).reshape(bs, -1)
            lat = self.act(x)
            out = self.cls(lat)
            return lat, out
        else:
            raise NotImplementedError


class Train:
    def __init__(self, model, train_loader, optimizer, criterion, args):
        self.model = model.to(args.device)
        self.args = args
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion

    def train_once(self, epoch):
        loss_once = .0
        acc_once = .0
        self.model.train()
        for idx, (data, label) in enumerate(self.train_loader):
            data, label = data.to(self.args.device), label.to(self.args.device)
            label = label.reshape(-1)
            _, pred = self.model(data)
            loss = self.criterion(pred, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            label = label.cpu().numpy()
            p = np.argmax(pred.detach().cpu().numpy(), axis=-1)
            acc_once += np.sum(p == label).astype(np.float)
            loss_once += loss.item()
        acc_once /= len(self.train_loader.dataset)
        loss_once /= len(self.train_loader)
        print('epoch: {}, acc: {:.4f}, loss: {:.4f}'.format(epoch+1, acc_once, loss_once))
        if epoch % self.args.fixed_epoch == 0:
            if epoch // self.args.fixed_epoch <= 5:
                clus_num = int(self.args.clus_num - int(epoch // self.args.fixed_epoch * 10))
            else:
                clus_num = int(self.args.clus_num - 50)
            print('current cluster num: {}'.format(clus_num))
            self.model.eval()
            clus = KMeans(n_clusters=clus_num)
            data = self.train_loader.dataset.return_all().to(self.args.device)
            features, _ = self.model(data)
            features = features.detach().cpu()
            clus_label = clus.fit_predict(features)
            train_set = CustomDataset(data.cpu().numpy(), clus_label, self.args.trans)
            self.train_loader = DataLoader(train_set, batch_size=self.args.bs, shuffle=True)
        return acc_once, loss_once

    def train(self):
        acc_list = []
        loss_list = []
        loss_interval_list = []
        loss_min = np.inf
        save = False
        for epoch in range(self.args.max_epoch):
            acc, loss = self.train_once(epoch)
            acc_list.append(acc)
            loss_list.append(loss)
            if epoch % self.args.fixed_epoch == 1 and epoch >= self.args.max_epoch // 2:
                save = False if loss_min < loss else True
                loss_min = loss_min if loss_min < loss else loss
            if self.args.path != '' and (epoch % self.args.fixed_epoch == self.args.fixed_epoch-1) \
                    and save and (epoch >= self.args.max_epoch // 2):
                print('current epoch: {}'.format(epoch))
                state = {
                    'model': self.model.state_dict(),
                    'epoch': epoch,
                    'args': self.args,
                }
                torch.save(state, self.args.path)
            if epoch == self.args.max_epoch-1:
                state = {
                    'model': self.model.state_dict(),
                    'epoch': epoch,
                    'args': self.args,
                }
                torch.save(state, self.args.final_path)
        return acc_list, loss_list


class CustomDataset(Dataset):
    def __init__(self, data, label, trans=None):
        super().__init__()
        self.data = data
        self.label = label
        self.trans = trans

    def __getitem__(self, item):
        piece = self.trans(self.data[item]) if self.trans is not None else self.data[item]
        return torch.FloatTensor(piece), torch.LongTensor(self.label[item].reshape(-1))

    def __len__(self):
        return len(self.data)

    def return_all(self):
        return torch.FloatTensor(self.data)
