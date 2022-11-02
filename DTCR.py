import torch
import numpy as np
from copy import deepcopy
from torch import nn
from torch.utils.data import Dataset
from easydict import EasyDict


class DTCR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # encoder
        self.enc_conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=args.up_sample_chn,
                kernel_size=args.conv_ker,
                stride=args.conv_stride
            ),
            nn.LeakyReLU(.1, True),
        )
        self.enc_gru = nn.GRU(
            input_size=args.up_sample_chn,
            hidden_size=args.hidden_size,
            bidirectional=args.bidirectional,
            num_layers=args.num_rnn_layer,
            batch_first=True,
            dropout=args.dropout
        )
        # decoder
        self.dec_gru = nn.GRU(
            input_size=args.hidden_size*2 if args.bidirectional else args.hidden_size,
            hidden_size=args.up_sample_chn,
            bidirectional=args.bidirectional,
            num_layers=args.num_rnn_layer,
            batch_first=True,
            dropout=args.dropout
        )
        self.dec_deconv1d = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=args.up_sample_chn*2 if args.bidirectional else args.up_sample_chn,
                out_channels=1,
                kernel_size=args.deconv_ker,
                stride=args.conv_stride
            ),
        )
        # classifier
        self.cls = nn.Sequential(
            nn.Linear(args.hidden_size*2, args.cls_hid_size) if args.bidirectional else nn.Linear(args.hidden_size, args.cls_hid_size),
            nn.LeakyReLU(.1, True),
            nn.Dropout(args.dropout),
            nn.Linear(args.cls_hid_size, 2)
        )
        # cluster related
        self.act1 = nn.LeakyReLU(.1)
        self.act2 = nn.LeakyReLU(.1)

    def forward(self, x):
        bs, t0, n0 = x.shape
        x = x.reshape(-1, n0, t0)
        x = self.enc_conv1d(x)  # shape: [bs, n1, t1]
        _, n1, t1 = x.shape
        x = x.reshape(-1, t1, n1)
        hid_n = self.args.num_rnn_layer*2 if self.args.bidirectional else self.args.num_rnn_layer
        hid_enc_init = torch.zeros(hid_n, bs, self.args.hidden_size).to(self.args.device)
        lat, hid_enc = self.enc_gru(x, hid_enc_init)  # lat: [bs, t1, hid_size*2], hid_enc: [n*2, bs, hid_size]
        lat, hid_enc = self.act1(lat), self.act2(hid_enc)
        hid_cls = torch.cat((hid_enc[-2], hid_enc[-1]), dim=-1)  # shape: [bs, hidden_size*2]
        hid_dec_init = torch.zeros(hid_n, bs, self.args.up_sample_chn).to(self.args.device)
        out_dec, hid_dec = self.dec_gru(lat, hid_dec_init)  # out_dec: [bs, t1, up_sample_chn*2], hid_dec: [n*2, t1, up_sample_chn]
        out_dec = out_dec.reshape(bs, -1, t1)
        out = self.dec_deconv1d(out_dec)  # shape: [bs, n0, t0]
        out = out.reshape(bs, t0, n0)
        if self.args.use_cls:
            out_cls = self.cls(hid_cls)
            return hid_cls, out, out_cls
        return hid_cls, out


class DTCRDataset(Dataset):
    def __init__(self, data, args, trans=None):
        super().__init__()
        self.data = data
        self.args = args
        self.trans = trans

    def __getitem__(self, item):
        piece = self.data[item]
        if self.trans:
            piece = self.trans(piece)
        if self.args.generate_fake:
            fake_piece = self._generate_fake_sample(piece)
            return torch.FloatTensor(piece), torch.FloatTensor(fake_piece), torch.IntTensor([1, 0])
        else:
            return torch.FloatTensor(piece)

    def __len__(self):
        return len(self.data)

    def _generate_fake_sample(self, real):
        index = list(range(len(real)))
        len_change = int(self.args.alpha * len(index))
        index_change = np.random.choice(index, len_change, replace=False)
        index_raw = np.sort(index_change)
        fake = deepcopy(real)
        fake[index_raw] = real[index_change]
        return fake


class Train:
    def __init__(self, model, loader, criterion, optimizer, args):
        super().__init__()
        self.loader = loader
        self.model = model.to(args.device)
        self.args = args
        self.criterion = criterion.to(args.device)
        self.optimizer = optimizer

    def train(self):
        last_loss = torch.inf
        loss_list = []
        for epoch in range(self.args.max_epoch):
            res = self.train_once()
            print('epoch: {}'.format(epoch))
            if 'loss' in res:
                print('loss: {:.6f}'.format(res.loss))
                loss_list.append(res.loss)
                if self.args.tensorboard:
                    self.args.tensorboard.add_scalar(tag='train loss', scalar_value=res.loss, global_step=epoch+1)
                if res.loss < last_loss:
                    state = {'model': self.model.state_dict(), 'epoch': epoch}
                    torch.save(state, self.args.ckpt_path)
                    last_loss = res.loss
        return loss_list

    def train_once(self):
        if self.args.with_label:
            return self.train_with_label()
        else:
            if self.args.generate_fake:
                return self.train_with_fake()
            return self.train_without_label()

    def train_with_label(self):
        raise NotImplementedError

    def train_without_label(self):
        self.model.train()
        loss_epoch = .0
        for idx, data in enumerate(self.loader):
            data = data.to(self.args.device)
            data = torch.unsqueeze(data, dim=-1)
            hid, out = self.model(data)
            loss = self.criterion(out, data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_epoch += loss.cpu().item()
            print('step: [{}]/[{}], loss: {:.6f}'.format(idx, len(self.loader), loss.cpu().item()))
        rec = EasyDict()
        rec.loss = loss_epoch / len(self.loader)
        return rec

    def train_with_fake(self):
        self.model.train()
        loss_epoch = 0
        for idx, (data, fake, label) in enumerate(self.loader):
            data, fake, label = data.to(self.args.device), fake.to(self.args.device), label.to(self.args.device)
            data = torch.unsqueeze(data, dim=-1)
            fake = torch.unsqueeze(fake, dim=-1)
            hid1, out1 = self.model(data)
            hid2, out2 = self.model(fake)
            loss = self.criterion(out1, data)
            loss += self.criterion(out2, fake)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_epoch += loss.cpu().item()
            print('step: [{}]/[{}], loss: {:.6f}'.format(idx, len(self.loader), loss.cpu().item()))
        rec = EasyDict()
        rec.loss = loss_epoch / len(self.loader)
        return rec


class Transform:
    def __init__(self, trans_list):
        self.trans_list = trans_list

    def __call__(self, data):
        for t in self.trans_list:
            data = t(data)
        return data


def test(model, test_loader, args):
    model.to(args.device)
    model.eval()
    recon_list = []
    latent_list = []
    for idx, data in enumerate(test_loader):
        data = data.to(args.device)
        data = torch.unsqueeze(data, dim=-1)
        hid, out = model(data)
        latent_list.append(hid.detach().cpu())
        recon_list.append(out.detach().cpu())
    latent_array = np.concatenate(latent_list, axis=0)
    recon_array = np.concatenate(recon_list)
    return np.squeeze(recon_array), latent_array


def z_score(data):
    return (data - np.mean(data)) / np.std(data)


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
