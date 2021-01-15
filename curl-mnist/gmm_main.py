import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import ContrastiveDataset, build_CURL_dataset
from GMM import GMM, GMMDataset, visualize
from model import (
    ClassificationNet,
    FCLayer,
    Representation,
    train_curl,
    train_multiclass_sup,
)

torch.manual_seed(0)

N_CENTERS = 30
DIM = 10
TRAIN_SAMPLES = 5000
TEST_SAMPLES = 5000
VARIANCE = 1

CURL_TRAIN_SIZE = 50 * N_CENTERS
CURL_TEST_SIZE = 1000
BATCH_SIZE = 100

# Model
INPUT_DIM = DIM
HIDDEN_DIM = 20
OUT_DIM = HIDDEN_DIM

# Training
N_EPOCH = 1000
LR = 1e-3

# Data generation
CENTERS = torch.randn(N_CENTERS * DIM).view(N_CENTERS, DIM)
gmm = GMM(DIM, CENTERS, VARIANCE)

X_train, y_train = gmm.sample(TRAIN_SAMPLES)
X_test, y_test = gmm.sample(TEST_SAMPLES)

train_CURL = ContrastiveDataset(*build_CURL_dataset(X_train, y_train, CURL_TRAIN_SIZE))
assert len(train_CURL) == CURL_TRAIN_SIZE
test_CURL = ContrastiveDataset(*build_CURL_dataset(X_test, y_test, CURL_TEST_SIZE))

train_data = GMMDataset(X_train, y_train)
test_data = GMMDataset(X_test, y_test)

train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)

curl_train_loader = DataLoader(train_CURL, shuffle=True, batch_size=BATCH_SIZE)
curl_test_loader = DataLoader(test_CURL, shuffle=False, batch_size=BATCH_SIZE)

# Model
curl_model = Representation(INPUT_DIM, HIDDEN_DIM, OUT_DIM)
sup_model = ClassificationNet(
    Representation(INPUT_DIM, HIDDEN_DIM, OUT_DIM), FCLayer(OUT_DIM, N_CENTERS)
)

writer_str = (
    "CURL/GMM-"
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
"""
sup_model = train_multiclass_sup(
    train_loader,
    sup_model,
    writer,
    N_EPOCH,
    LR,
    verbose=True,
    visualize=True,
)
"""
curl_model = train_curl(
    curl_train_loader,
    curl_model,
    writer,
    N_EPOCH,
    LR,
    verbose=True,
    visualize=True,
    test_X=X_test,
    test_y=y_test,
)
