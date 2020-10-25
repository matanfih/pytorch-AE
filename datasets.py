from __future__ import print_function, division
from glob import glob
from os import listdir
from os.path import join
import csv, time
import PIL
import torch
from PIL.Image import Image
from torchvision import datasets, transforms
import os
import torch
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import Rescale, ToTensor
from sklearn.model_selection import train_test_split


class MNIST(object):
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist', train=False, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)


class EMNIST(object):
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.EMNIST('data/emnist', train=True, download=True, split='byclass',
                            transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.EMNIST('data/emnist', train=False, split='byclass',
                            transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)


class FashionMNIST(object):
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data/fmnist', train=True, download=True,
                                  transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data/fmnist', train=False, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)


class XrayDataset(Dataset):
    CSV_FIXED_PATH_FILE_NAME_PREFIX = "fixed_path_"
    TRAIN = "train_"
    TEST = "test_"

    FIXED_TRAIN = TRAIN + CSV_FIXED_PATH_FILE_NAME_PREFIX
    FIXED_TEST = TEST + CSV_FIXED_PATH_FILE_NAME_PREFIX

    def __pre_process(self):
        images = glob("{}/images_*/*.png".format(self.root_dir))
        image_list = self.xray_frame["Image Index"].tolist()
        labels = self.xray_frame["Finding Labels"].tolist()
        self_len = self.__len__()
        index = 1

        for img in images:
            name = os.path.basename(img)

            i = image_list.index(name)
            image_list[i] = img

            print("images left: %s" % (self_len - index))
            index += 1

        X_train, X_test, y_train, y_test = train_test_split(image_list, labels, test_size=0.2, random_state=42)

        for X, y, filename in [(X_train, y_train, self.train_csv_fixed_path), (X_test, y_test, self.test_csv_fixed_path)]:
            print("writing the X: %s, y: %s -> %s" % (len(X), len(y), filename))
            with open(filename, 'w') as csvf:
                writer = csv.writer(csvf)
                writer.writerow(["Image Index", "Finding Labels"])
                writer.writerows(zip(X, y))
        print("%s finished" % self.__pre_process.__name__)

    def __init__(self, csv_file='Data_Entry_2017_v2020.csv', root_dir='/data/matan/nih', transform=None, train=True):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.orig_csv_path = os.path.join(root_dir, csv_file)
        self.test_csv_fixed_path = os.path.join(root_dir, "{}{}".format(self.FIXED_TEST, csv_file))
        self.train_csv_fixed_path = os.path.join(root_dir, "{}{}".format(self.FIXED_TRAIN, csv_file))
        self.root_dir = root_dir

        if not os.path.exists(self.test_csv_fixed_path) or not os.path.exists(self.train_csv_fixed_path):
            self.xray_frame = pd.read_csv(self.orig_csv_path)
            self.__pre_process()

        elif train:
            self.xray_frame = pd.read_csv(self.train_csv_fixed_path)
        else: # test
            self.xray_frame = pd.read_csv(self.test_csv_fixed_path)

        self.transform = transform

    def __len__(self):
        return len(self.xray_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.xray_frame.iloc[idx]["Image Index"]
        image = io.imread(img_name)
        xray = self.xray_frame.iloc[idx, 1]
        xray = np.array([xray])
        # xray = xray.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': xray}

        if self.transform:
            sample = self.transform(sample)

        return sample['image'], sample['landmarks']


class XRAY(object):
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            XrayDataset(train=True, transform=transforms.Compose([Rescale(128), transforms.ToTensor()])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            XrayDataset(train=False, transform=transforms.Compose([Rescale(128), transforms.ToTensor()])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
