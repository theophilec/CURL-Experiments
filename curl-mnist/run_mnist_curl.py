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

from data import ContrastiveDataset, build_CURL_dataset, Dataset
from MNIST import MNIST_pre_processing
from model import (
    ClassificationNet,
    FCLayer,
    Representation,
    train_curl,
)

torch.manual_seed(0)

DIM = 28 * 28

CURL_TRAIN_SIZE = 10000
SUP_SIZE = 100
BATCH_SIZE = 1000

# Model
INPUT_DIM = DIM
HIDDEN_DIM = int(0.70 * INPUT_DIM)
OUT_DIM = 20

# Training
N_EPOCH = 1000
LR = 1e-3

ROOT_DIR = "./data/MNIST"
DOWNLOAD = True
train_data = MNIST(
    ROOT_DIR, train=True, download=DOWNLOAD, transform=transforms.ToTensor()
)
test_data = MNIST(
    ROOT_DIR, train=False, download=DOWNLOAD, transform=transforms.ToTensor()
)

X_train, y_train = MNIST_pre_processing(train_data)
X_test, y_test = MNIST_pre_processing(test_data)

train_CURL = ContrastiveDataset(*build_CURL_dataset(X_train, y_train, CURL_TRAIN_SIZE))
test_CURL = ContrastiveDataset(*build_CURL_dataset(X_test, y_test, CURL_TRAIN_SIZE))


curl_train_loader = DataLoader(train_CURL, shuffle=True, batch_size=BATCH_SIZE)
curl_test_loader = DataLoader(train_CURL, shuffle=True, batch_size=BATCH_SIZE)

curl_model = Representation(INPUT_DIM, HIDDEN_DIM, OUT_DIM)


writer_str = (
    "CURL/MNIST-CURL-"
    + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    + "-"
    + str(INPUT_DIM)
    + "-"
    + str(HIDDEN_DIM)
    + "-"
    + str(OUT_DIM)
    + "-"
    + str(LR)
    + "-"
    + str(BATCH_SIZE)
)
writer = SummaryWriter(writer_str)
curl_model = train_curl(
    curl_train_loader,
    curl_test_loader,
    curl_model,
    writer,
    N_EPOCH,
    LR,
    verbose=True,
    visualize=True,
    test_X=X_test,
    test_y=y_test,
)
