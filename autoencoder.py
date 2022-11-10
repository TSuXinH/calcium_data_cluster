import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from copy import deepcopy

from utils import generate_contrast


class AutoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        enc_list = []
        dec_list = []
        length = len(args.hidden_dim)
        for idx in range(length+1):
            if idx == 0:
                enc_list.extend([
                    nn.Linear(args.input_dim, args.hidden_dim[0]),
                    nn.LeakyReLU(.1, True),
                    nn.Dropout(args.drop_ratio)
                ])
                dec_list.extend([
                    nn.Linear(args.latent_dim, args.hidden_dim[-1]),
                    nn.LeakyReLU(.1, True),
                    nn.Dropout(args.drop_ratio)
                ])
            elif idx == len(args.hidden_dim):
                enc_list.extend([
                    nn.Linear(args.hidden_dim[-1], args.latent_dim),
                ])
                dec_list.extend([
                    nn.Linear(args.hidden_dim[0], args.input_dim)
                ])
            else:
                enc_list.extend([
                    nn.Linear(args.hidden_dim[idx-1], args.hidden_dim[idx]),
                    nn.LeakyReLU(.1, True),
                    nn.Dropout(args.drop_ratio)
                ])
                dec_list.extend([
                    nn.Linear(args.hidden_dim[length-idx], args.hidden_dim[length-idx-1]),
                    nn.LeakyReLU(.1, True),
                    nn.Dropout(args.drop_ratio)
                ])
        self.enc = nn.Sequential(*enc_list)
        self.dec = nn.Sequential(*dec_list)

    def forward(self, neu_data):
        latent = self.enc(neu_data)
        decoded = self.dec(latent)
        return latent, decoded


class AETest1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.enc = nn.Sequential(
            nn.Linear(args.input_dim, args.input_dim // 4),
            nn.LeakyReLU(.1, True),
            nn.Linear(args.input_dim // 4, args.input_dim // 16),
        )
        self.dec = nn.Sequential(
            nn.Linear(args.input_dim // 16, args.input_dim // 4),
            nn.LeakyReLU(.1, True),
            nn.Linear(args.input_dim // 4, args.input_dim)
        )
        self.dis = nn.Sequential(
            nn.Linear(args.input_dim // 16, args.pred_dim),
        )

    def forward(self, x):
        lat = self.enc(x)
        dec = self.dec(lat)
        if self.args.contrast:
            dis_res = self.dis(lat)
            return lat, dec, dis_res
        else:
            return lat, dec

    def info_nce_loss(self, x):
        labels = torch.cat([torch.arange(self.args.batch_size) for _ in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        x = F.normalize(x, dim=1)
        similarity_matrix = torch.matmul(x, x.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        logits = logits / self.args.temperature
        return logits, labels


class CustomDataset(Dataset):
    def __init__(self, neu_data, args):
        super().__init__()
        self.args = args
        self.data = neu_data

    def __getitem__(self, item):
        piece = self.data[item]
        if self.args.aug is not None:
            piece = self.args.aug(piece)
        if self.args.trans is not None:
            piece = self.args.trans(piece)
        return torch.FloatTensor(piece)

    def __len__(self):
        return len(self.data)


class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.sum = 0
        self.length = 0
        self.ave = 0.
        self.last_ave = 0.

    def __repr__(self):
        return 'the average {} is {:.4f}'.format(self.name, self.ave)

    def update(self, data):
        self.sum += data
        self.length += 1
        self.ave = self.sum / self.length

    def clear(self):
        self.sum = self.length = self.ave = self.last_ave = 0.

    def clear_last(self):
        self.last_ave = deepcopy(self.ave)
        print(self.last_ave)
        self.sum = self.length = self.ave = 0.

    def is_better(self):
        return self.ave < self.last_ave


def train(model, train_loader, crit, optim, args, valid_loader=None):
    model.train()
    min_loss = np.inf
    train_loss_list = []
    valid_loss_list = []
    if valid_loader is not None:
        for epoch in range(args.max_epoch):
            model.train()
            train_loss = .0
            valid_loss = .0
            for idx, data in enumerate(train_loader):
                data = data.to(args.device)
                _, decoded = model(data)
                loss = crit(decoded, data)
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss += loss.item()
                print('epoch: {}, iteration: {}, loss: {:.6f}'.format(epoch, idx, loss.item()))
            train_loss /= len(train_loader)
            train_loss_list.append(train_loss)
            model.eval()
            for idx, data in enumerate(valid_loader):
                data = data.to(args.device)
                _, decoded = model(data)
                loss = crit(decoded, data)
                valid_loss += loss.item()
            valid_loss /= len(valid_loader)
            valid_loss_list.append(valid_loss)
            print('train loss: {:.6f}, valid loss: {:.6f}'.format(train_loss, valid_loss))
            if args.save_path:
                if min_loss > valid_loss:
                    min_loss = valid_loss
                    state = {
                        'model': model.state_dict(),
                        'epoch': epoch,
                        'args': args
                    }
                    torch.save(state, args.save_path)
                    print('saving, current epoch: {}'.format(epoch))
        return train_loss_list, valid_loss_list
    else:
        for epoch in range(args.max_epoch):
            model.train()
            train_loss = .0
            for idx, data in enumerate(train_loader):
                data = data.to(args.device)
                _, decoded = model(data)
                loss = crit(decoded, data)
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss += loss.item()
                print('epoch: {}, iteration: {}, loss: {:.6f}'.format(epoch, idx, loss.item()))
            train_loss /= len(train_loader)
            train_loss_list.append(train_loss)
            print('train loss: {:.6f}'.format(train_loss))
            if args.save_path:
                if min_loss > train_loss:
                    min_loss = train_loss
                    state = {
                        'model': model.state_dict(),
                        'epoch': epoch,
                        'args': args
                    }
                    torch.save(state, args.save_path)
                    print('saving, current epoch: {}'.format(epoch))
        return train_loss_list, valid_loss_list


def test(model, test_loader, args):
    model.to(args.device)
    model.eval()
    recon_list = []
    latent_list = []
    for idx, data in enumerate(test_loader):
        data = data.to(args.device)
        hid, out = model(data)
        latent_list.append(hid.detach().cpu())
        recon_list.append(out.detach().cpu())
    latent_array = np.concatenate(latent_list, axis=0)
    recon_array = np.concatenate(recon_list)
    return np.squeeze(recon_array), latent_array


def cal_err(recon, raw):
    return np.mean((recon - raw) ** 2)

