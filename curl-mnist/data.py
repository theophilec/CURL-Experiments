import torch
import torch.distributions as D


class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return self.X.shape[0]


def build_CURL_dataset(X_data, y_data, n_samples: int, collision=True):
    """Build a CURL dataset from X_data, y_data.

    Args:
        X_data: Tensor, N, dim
        y_data: Tensor, N
        n_samples: int

    If collision, class collision is allowed.
    """
    n_latent = len(torch.unique(y_data))

    X = []
    X_pos = []
    X_neg = []

    mix_dist = D.Categorical(torch.ones((n_latent,)))

    # sorting for efficient sampling
    y, argsort = mix_dist.sample((n_samples,)).sort()
    for i in range(n_latent):
        sub = torch.eq(y_data, i)
        n_max = sub.sum()
        n = torch.eq(y, i).sum()
        if n > 0:
            # assume we can sample x, x+ faithfully
            X_ind = torch.randint(0, n_max, (n,))
            X_pos_ind = torch.randint(0, n_max, (n,))
            X.append(X_data[sub][X_ind])
            X_pos.append(X_data[sub][X_pos_ind])
    X = torch.cat(X)
    X_pos = torch.cat(X_pos)

    if collision:
        y_neg, neg_argsort = mix_dist.sample((n_samples,)).sort()
        for i in range(n_latent):
            sub = torch.eq(y_data, i)
            n_max = sub.sum()
            n = torch.eq(y_neg, i).sum()
            X_neg_ind = torch.randint(0, n_max, (n,))
            if n > 0:
                X_neg.append(X_data[sub][X_neg_ind])
        X_neg = torch.cat(X_neg)
    else:
        raise NotImplementedError

    return (
        X[argsort],
        y[argsort],
        X_pos[argsort],
        X_neg[neg_argsort],
        y_neg[neg_argsort],
    )


class ContrastiveDataset:
    def __init__(self, X, y, X_pos, X_neg, y_neg):
        # validation
        assert X.shape == X_pos.shape
        assert X.shape == X_neg.shape
        assert y.shape == y_neg.shape
        assert y.shape[0] == X.shape[0]

        self.X = X
        self.y = y
        self.X_pos = X_pos
        self.X_neg = X_neg
        self.y_neg = y_neg

    def n_collisions(self):
        return torch.eq(self.y, self.y_neg).sum()

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.X_pos[i], self.X_neg[i], self.y_neg[i]

    def __len__(self):
        return self.X.shape[0]
