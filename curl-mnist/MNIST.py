import datetime

import matplotlib.pyplot as plt
import torch
import torch.distributions as D
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import MNIST

from data import ContrastiveDataset, build_CURL_dataset


def MNIST_pre_processing(dataset: torch.utils.data.Dataset):
    """Input MNIST Dataset object and return stacked X, y as tuple."""
    n = len(dataset)
    feature_size = dataset.data[0][0].shape[0] ** 2

    X_, y_ = [], []
    for X, y in dataset:
        X_.append(X.view(1, feature_size))
        y_.append(y)
    X = torch.cat(X_)
    assert X.shape == (n, feature_size)
    y = torch.Tensor(y_)
    assert y.shape == (n,)
    return X, y.long()
