import numpy as np
from sklearn.metrics import r2_score

from .utils import r_metric, shuffle_c


def _get_permutation_mask(hierarchy, mode="permute_current"):
    """
    From [0, 0, 0, 1, 1, 2] shape (x_dim,) to
    [[True, True, True, False, False, False],
     [False, False, False, True, True, False],
     [False, False, False, False, False, True,]]
     of shape (n_groups, x_dim)
    """
    assert mode in ["permute_current", "permute_below", "permute_above"]
    permutation_mask = np.zeros((len(np.unique(hierarchy)), len(hierarchy)), dtype=bool)
    for i, level in enumerate(np.unique(hierarchy)):
        if mode == "permute_current":
            permutation_mask[i] = hierarchy == level
        elif mode == "permute_below":
            permutation_mask[i] = hierarchy <= level
        elif mode == "permute_above":
            permutation_mask[i] = hierarchy > level
    return permutation_mask


def _get_permutation_score(
    estimator, X, y, permutation_mask, metric=r_metric, n_repeats=10
):
    """
    Compute permutation score, given permutation_mask.
    Ex: if permutation mask = 
    [[True, True, True, False, False, False],
        [False, False, False, True, True, False],
        [False, False, False, False, False, True,]]
        of shape (n_groups, x_dim)
    Compute the permutation importance, successively for:
        - cols 0-2
        - cold 3-5
        - cols 6
    To compute the permutation importance:
        First compute the score for X.
        Then, for each group i:
            - permute the columns permutation_mask[i] n_repeats time
            - compute the score for permuted X each time
            - average scores over repeats
            - take the difference between baseline score and
            scores with permutation (averaged on repeats)  
    Returns
        array of shape (n_groups, y_dim)
            with n_groups = len(unique(hierarchy))
            importance[i, j] gives the contribution of features
            group i for th prediction of y_j.
    """

    # Scores without permutation
    pred = estimator.predict(X)
    baseline_score = metric(pred, y)

    # Scores with permutation
    score = np.zeros((n_repeats, len(permutation_mask), *baseline_score.shape))
    for level, mask in enumerate(permutation_mask):
        for repeat in range(n_repeats):

            # Split
            to_shuffle = np.where(mask)[0]

            # Shuffle
            X_permuted = X.copy()
            X_permuted[:, to_shuffle] = shuffle_c(X[:, to_shuffle])

            # Predict
            pred = estimator.predict(X_permuted)

            # Score
            score[repeat, level] = metric(pred, y)

    # Aggregate
    score = score.mean(0)

    # Importance
    importance = baseline_score[None] - score

    return importance


class Scorer(object):
    def __init__(self, metric=r2_score):
        self.metric = metric

    def score(self, model, X, y):
        pred = model.predict(X)
        r = self.metric(pred, y)
        return r


class PermutationScorer(Scorer):
    def __init__(
        self, hierarchy, metric=r2_score, mode="permute_current", n_repeats=50
    ):
        assert mode in ["permute_current", "permute_below", "permute_above"]
        self.permutation_mask = _get_permutation_mask(hierarchy, mode=mode)
        self.n_repeats = n_repeats
        self.metric = metric
        self.mode = mode

    def score(self, model, X, y):
        r = _get_permutation_score(
            model,
            X,
            y,
            self.permutation_mask,
            metric=self.metric,
            n_repeats=self.n_repeats,
        )
        if self.mode in ["permute_below", "permute_above"]:
            r[1:] -= r[:-1]

        return r
