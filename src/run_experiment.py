import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .get_contrib import (
    contrib_concat,
    contrib_permute_above_train,
    contrib_permute_below,
    contrib_permute_current,
)


def get_column_transformers(hierarchy, scale=True, pca=20):
    levels, counts = np.unique(hierarchy, return_counts=True)
    transformers = []
    hiearchy_trf = []
    for level, count in zip(levels, counts):
        cols = np.where(hierarchy == level)[0]
        pipes = []
        if scale:
            pipes.append(StandardScaler())
        if pca > 0:
            count = min(count, pca)
            pipes.append(PCA(count))
        trf = make_pipeline(*pipes)
        transformers.append((trf, cols))
        hiearchy_trf.extend([level] * count)
    transformers = make_column_transformer(*transformers)
    hiearchy_trf = np.array(hiearchy_trf)
    return transformers, hiearchy_trf


def run_experiment(
    exp,
    X,
    y,
    hierarchy,
    model=RidgeCV(np.logspace(-2, 8, 20)),
    scale=True,
    n_repeats=50,
    pca=20,
):
    """Compute the contribution of each feature group, given in hierarchy,
    to predict (given X and model) each dimension of the target y.
    Ex: if hierarchy = [0, 0, 0, 1, 1, 2] (n_groups=3), and y is shape (n, 10)
    (y_dim=10),
    returns an array of shape (n_groups, y_dim) = (3, 10), giving
    the contribution of each group to the prediction of a particular dimension.

    4 possible methods can be used

    Parameters
    ----------
    exp : str
        one of ["permute_current", "permute_below", "concat", "permute_above_train"]
        * "permute_current"
            + orthogonalise
            + concatenate features
            + fit on train
            + compute permutation importance (n_repeats=50), for each feature group separately, on test
            + re-iterate on folds

        * "permute_below":
            + concatenate features
            + fit on train
            + compute permutation importance (n_repeats=50), for 
                - the first group A: importance[A] = r - r_with_A_shuffled
                - the first + second group (importance(A+B) = r - r_with_A_and_B_shuffled)
                - the first + second + third group (importance(A+B+C) = r - r_with_A_and_B_and_C_shuffled)
                - ... 
                - all groups shuffled
            + extract specific contribution for each level
                - contrib_A = importance(A)
                - contrib_B = importance(A+B) - importance of first group (A)
                - ...
            + re-iterate on folds
        * "concat":
            + concatenate features
            + in order to predict Y, fit in hiearchical order:
                * the first group A -> r(A)
                * the first + second group -> r(A+B)
                * ... 
            + extract specific contribution for each level
                * contrib_A = r(A)
                * contrib_B = r(A+B) - r(A)
                * contrib_C = r(A+B+C) - r(A+B)
        * "permute_above_train":
            same as "concat", but fill the missing dimension with random noise,
            so that X in every experiment has the same dimension.
    X : array of shape (n, x_dim)
        features
    y : array of shape (n, y_dim)
        target
    hierarchy : array of int of shape (x_dim)
        indices of the group for each feature.
        Ex: [0, 0, 0, 1, 1, 2] refers to 3 groups,
        the first with dim=3, second with dim=2, last dim=1
    model : sklearn model, optional,
        by default RidgeCV(np.logspace(-2, 8, 20))
        predictive model used
        - to predict y given X,
        - is exp="permute_one", the model
        is also used to orthogonlize features
    scale : bool, optional
        whether to scale each group of features (independantly)
        before fitting, by default True
    n_repeats : int, optional
        number of repeats for exp "permute_xxx", by default 50
        Scores are always averaged across repeats,
    pca : int, optional
        if > 0, PCA is applied to each group of features
        (independantly) before fitting, refers to the number
        of components to use in the PCA, by default 20

    Returns
    -------
    array of shape (n_groups, y_dim)
        with n_groups = len(unique(hierarchy))
        importance[i, j] gives the contribution of features
        group i for th prediction of y_j.
    """
    assert exp in ["permute_current", "permute_below", "concat", "permute_above_train"]

    # Apply scaling and PCA
    if scale or pca:
        transformers, hierarchy = get_column_transformers(
            hierarchy, scale=scale, pca=pca
        )
        X = transformers.fit_transform(X)
    assert X.shape[1] == len(hierarchy)

    if exp == "permute_current":
        importance = contrib_permute_current(
            X, y, hierarchy=hierarchy, model=model, n_repeats=n_repeats
        )
    if exp == "permute_below":
        importance = contrib_permute_below(
            X, y, hierarchy=hierarchy, model=model, n_repeats=n_repeats
        )
    if exp == "concat":
        importance = contrib_concat(X, y, hierarchy=hierarchy, model=model)

    if exp == "permute_above_train":
        importance = contrib_permute_above_train(
            X, y, hierarchy=hierarchy, model=model, n_repeats=n_repeats
        )

    return importance
