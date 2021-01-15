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
    train_multiclass_sup,
)

ROOT_DIR = "./data/MNIST"
DOWNLOAD = True

torch.manual_seed(0)

DIM = 784
N_LABELS = 10

SUP_TRAIN_SIZE = 100
BATCH_SIZE = 1000

# Model
INPUT_DIM = DIM
HIDDEN_DIM = int(0.70 * INPUT_DIM)
OUT_DIM = 20

# Training
N_EPOCH = 1000
LR = 1e-3

train_data = MNIST(
    ROOT_DIR, train=True, download=DOWNLOAD, transform=transforms.ToTensor()
)
test_data = MNIST(
    ROOT_DIR, train=False, download=DOWNLOAD, transform=transforms.ToTensor()
)
X_train, y_train = MNIST_pre_processing(train_data)
X_test, y_test = MNIST_pre_processing(test_data)

mnist_train = Dataset(X_train[:SUP_TRAIN_SIZE], y_train[:SUP_TRAIN_SIZE])
train_loader = DataLoader(mnist_train, shuffle=True, batch_size=BATCH_SIZE)

sup_model = ClassificationNet(
    Representation(INPUT_DIM, HIDDEN_DIM, OUT_DIM), FCLayer(OUT_DIM, N_LABELS)
)

writer_str = (
    "CURL/MNIST-Supervised-"
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
sup_model = ClassificationNet(
    Representation(INPUT_DIM, HIDDEN_DIM, OUT_DIM), FCLayer(OUT_DIM, N_LABELS)
)
sup_model = train_multiclass_sup(
    train_loader,
    sup_model,
    writer,
    N_EPOCH,
    LR,
    verbose=True,
    visualize=True,
    X_test=X_test,
    y_test=y_test,
)
