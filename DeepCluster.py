import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans


class DeepClusterNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.back_bone = nn.Sequential(
            nn.Linear(args.input_dim, args.hidden_dim),
            nn.LeakyReLU(.1, True),
            nn.Dropout(.3),
            nn.Linear(args.hidden_dim, args.latent_dim),
        )
        self.cls = nn.Sequential(
            nn.Linear(args.latent_dim, args.clus_num)
        )

    def forward(self, x):
        lat = self.back_bone(x)
        out = self.cls(lat)
        return lat, out


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
            self.model.eval()
            clus = KMeans(n_clusters=self.args.clus_num)
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
        for epoch in range(self.args.max_epoch):
            acc, loss = self.train_once(epoch)
            acc_list.append(acc)
            loss_list.append(loss)
            if self.args.path != '':
                state = {
                    'model': self.model.state_dict(),
                    'epoch': epoch
                }
                torch.save(state, self.args.path)
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
