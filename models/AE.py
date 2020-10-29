import time
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import device_utils
import sys

sys.path.append('../')
from architectures import FC_Encoder, FC_Decoder, CNN_Encoder, CNN_Decoder
from datasets import MNIST, EMNIST, FashionMNIST, XRAY


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        #output_size = args.embedding_size
        output_size = 4096
        self.input_size = (1, 128, 128)
        self.encoder = CNN_Encoder(output_size, self.input_size)

        self.decoder = CNN_Decoder(output_size, self.input_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class AE(object):
    def __init__(self, dataset, log_interval, is_cuda, batch_size):

        self.csv = ""
        self.root = ""
        self.dataset = dataset
        self.batch_size = batch_size
        self.log_interval =log_interval
        if is_cuda:
            self.device = device_utils.get_freeish_gpu()
        else:
            self.device = device_utils.get_cpu()
        print("going to use: %s" % self.device)
        self._init_dataset()
        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader

        self.model = Network()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def state_dict(self):
        return self.model.state_dict()

    @classmethod
    def load_state_dict(cls, checkpoint):
        model = Network()
        device = device_utils.get_freeish_gpu()
        model = model.cuda(device=device)
        return model.load_state_dict(checkpoint)

    def load_state_dict(self, checkpoint):
        return self.model.load_state_dict(checkpoint)

    def _init_dataset(self):
        if self.dataset == 'MNIST':
            self.data = MNIST(self.batch_size)
        elif self.dataset == 'EMNIST':
            self.data = EMNIST(self.batch_size)
        elif self.dataset == 'FashionMNIST':
            self.data = FashionMNIST(self.batch_size)
        elif str(self.dataset) == 'XRAY':
            self.data = XRAY(self.batch_size)

        else:
            print("Dataset not supported")
            sys.exit()

    def loss_function(self, recon_x, x):
        BCE = F.l1_loss(input=recon_x, target=x)
        return BCE

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch = self.model(data)
            loss = self.loss_function(recon_batch, data)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader),
                           loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(self.train_loader.dataset)))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device)
                recon_batch = self.model(data)
                test_loss += self.loss_function(recon_batch, data).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
