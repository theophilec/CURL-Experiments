import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
from sklearn.decomposition import PCA
from torchvision.utils import make_grid


# Inspired by:
# https://github.com/blakebordelon/CURL-Efficiency/blob/master/GMM_SYNTH/gmm_contrastive.py

class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            *[
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, latent_dim),
            ]
        )
        self.decoder = nn.Sequential(
            *[
                nn.Linear(latent_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            ]
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, y):
        return self.decoder(y)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Representation(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.layers = nn.Sequential(
            *[
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, out_dim),
            ]
        )

    def forward(self, x):
        return self.layers(x)


class FCLayer(nn.Module):
    def __init__(self, input_dim: int, out_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(self.input_dim, self.out_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        y_hat = self.log_softmax(self.fc(x))
        return y_hat


class ClassificationNet(nn.Module):
    def __init__(self, repNet: Representation, fcNet: FCLayer):
        super().__init__()
        self.repNet = repNet
        self.fcNet = fcNet

    def forward(self, x):
        return self.fcNet(self.repNet(x))


def accuracy(y_hat, y):
    return torch.eq(y_hat, y).sum()


def selection(X_scores, y):
    # inspired by code from PB-CURL
    # X_scores batch_size, dim
    # y batch_size,
    batch_size, n_classes = X_scores.shape
    ids = torch.Tensor(np.tile(np.arange(10), batch_size).reshape(batch_size, 10))
    targets = y.long().unsqueeze(1)
    not_targets = torch.where(ids != targets)[1].view(batch_size, 9)
    target_scores = X_scores.gather(0, targets)
    not_target_scores = X_scores.gather(0, not_targets)

    assert target_scores.shape == (batch_size, 1)
    assert not_target_scores.shape == (batch_size, n_classes - 1)

    delta = target_scores - not_target_scores
    return delta


def logistic_loss(v):
    # v is batch, dim
    return torch.log2(1 + torch.exp(-v).sum(1))


def curl_metrics(model, X, y, X_pos, X_neg, y_neg):
    """Compute empirical quantities in the Arora et al. 2019 paper.

    Args:
        model: nn.Module Representation
        X, y, X_pos, X_neg, y_neg: tensors
    """
    f_X, f_pos, f_neg = model(X), model(X_neg)
    contrast = (f_X * (f_pos - f_neg)).sum(1)
    sample_losses = logistic_loss(contrast)
    same = torch.eq(y, y_neg)
    different = torch.eq(y, y_neg)
    L_un = sample_losses.mean()
    L_un_same = sample_losses[same].mean()
    L_un_different = sample_losses[different].mean()
    tau = same.sum() / len(same)
    return L_un, L_un_same, L_un_different, tau




def train_multiclass_sup(
    dataloader,
    model,
    writer,
    n_epoch,
    lr,
    verbose=False,
    visualize=True,
    X_test=None,
    y_test=None,
):
    optim = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()
    for epoch in range(n_epoch):
        train_losses = []
        for x, y in dataloader:
            v = model(x)
            loss = loss_fn(v, y)
            loss = loss.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
        writer.add_scalar("SupLoss/train", np.mean(train_losses), epoch)
        if verbose:
            print(f"Epoch {epoch}: {np.mean(train_losses)}")
        if visualize and ((epoch % 100 == 1) or (epoch < 50)):
            fig = plt.figure()
            pca = PCA(n_components=2)
            with torch.no_grad():
                X_ = model.repNet(X_test)
                X_ = pca.fit_transform(X_)
            ax = plt.gca()
            subset = (
                torch.eq(y_test, 0)
                | torch.eq(y_test, 1)
                | torch.eq(y_test, 2)
                | torch.eq(y_test, 3)
                | torch.eq(y_test, 4)
            )
            ax.scatter(X_[subset, 0], X_[subset, 1], c=y_test[subset])
            ax.set_aspect("equal")
            writer.add_figure("rep-mc-sup", fig, epoch)
    return model


def train_curl(
    train_loader,
    test_loader,
    model,
    writer,
    n_epoch,
    lr,
    verbose=False,
    visualize=False,
    test_X=None,
    test_y=None,
):
    optim = torch.optim.Adam(params=model.parameters(), lr=lr)
    for epoch in range(n_epoch):
        train_losses = []
        for x, y, x_pos, x_neg, y_neg in train_loader:
            v = model(x) * (model(x_pos) - model(x_neg))
            v = v.sum(dim=1)
            assert v.shape == y.shape
            v = v.unsqueeze(-1)
            assert len(v.shape) == 2
            loss = logistic_loss(v)
            loss = loss.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
        writer.add_scalar("Loss/train", np.mean(train_losses), epoch)
        if epoch % 10 == 1:
            test_losses = []
            with torch.no_grad():
                X_ = model(test_X)
                for x, y, x_pos, x_neg, y_neg in test_loader:
                    v = model(x) * (model(x_pos) - model(x_neg))
                    v = v.sum(dim=1)
                    assert v.shape == y.shape
                    v = v.unsqueeze(-1)
                    assert len(v.shape) == 2
                    loss = logistic_loss(v)
                    loss = loss.mean()
                    test_losses.append(loss.item())
                writer.add_scalar("Loss/test", np.mean(test_losses), epoch)

        if verbose:
            print(f"Epoch {epoch}: {np.mean(train_losses)}")
        if visualize and ((epoch % 50 == 1) or (epoch < 50)):
            fig = plt.figure()
            pca = PCA(n_components=2)
            with torch.no_grad():
                X_ = model(test_X)
                X_ = pca.fit_transform(X_)
            ax = plt.gca()
            subset = (
                torch.eq(test_y, 0)
                | torch.eq(test_y, 1)
                | torch.eq(test_y, 2)
                | torch.eq(test_y, 3)
                | torch.eq(test_y, 4)
            )
            ax.scatter(X_[subset, 0], X_[subset, 1], c=test_y[subset])
            # ax.set_aspect("equal")
            writer.add_figure("rep/curl", fig, epoch)
    return model


def train_ae(
    train_loader,
    test_loader,
    model,
    writer,
    n_epoch,
    lr,
    verbose=False,
    visualize=False,
    X_test=None,
    y_test=None,
):
    images_for_diagnostics, _ = next(iter(test_loader))
    GRID_SIZE = 16
    IMAGE_SIZE = 28
    images_grid = make_grid(
        images_for_diagnostics[:GRID_SIZE]
        .reshape(GRID_SIZE, IMAGE_SIZE, IMAGE_SIZE)
        .unsqueeze(1)
        .repeat(1, 3, 1, 1)
    )
    writer.add_image("images", images_grid, 0)
    optim = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    for epoch in range(n_epoch):
        train_losses = []
        for x, _ in train_loader:
            x_ = model(x)
            loss = loss_fn(x_, x)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
        writer.add_scalar("loss/train", np.mean(train_losses), epoch)
        if visualize and ((epoch % 100 == 1) or (epoch < 50)):
            fig = plt.figure()
            pca = PCA(n_components=2)
            with torch.no_grad():
                X_batch, y_batch = X_test, y_test
                X_ = model.encoder(X_batch)
                X_ = pca.fit_transform(X_)
            subset = (
                torch.eq(y_test, 0)
                | torch.eq(y_test, 1)
                | torch.eq(y_test, 2)
                | torch.eq(y_test, 3)
                | torch.eq(y_test, 4)
            )
            ax = plt.gca()
            ax.scatter(X_[subset, 0], X_[subset, 1], c=y_batch[subset])
            writer.add_figure("rep/ae", fig, epoch)
        if epoch % 10 == 1:
            test_losses = []
            with torch.no_grad():
                for x, _ in test_loader:
                    x_ = model(x)
                    loss = loss_fn(x_, x)
                    test_losses.append(loss.item())
                writer.add_scalar("loss/test", np.mean(test_losses), epoch)
            with torch.no_grad():
                pred_images = (
                    model.forward(images_for_diagnostics[:GRID_SIZE].view(GRID_SIZE, -1))
                    .reshape(GRID_SIZE, IMAGE_SIZE, IMAGE_SIZE)
                    .unsqueeze(1)
                    .repeat(1, 3, 1, 1)
                )
            pred_images_grid = make_grid(pred_images)
            writer.add_image("pred_images", pred_images_grid, epoch)
    return model
