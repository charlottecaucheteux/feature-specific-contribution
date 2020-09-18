import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .orthogonalizer import HierarchOrthogonalizer
from .utils import r_metric, shuffle_c


def _get_permutation_mask(hierarchy, mode="permute_current"):
    """
    From [0, 0, 0, 1, 1, 2] shape (x_dim,) to
    [[True, True, True, False, False, False],
     [False, False, False, True, True, False],
     [False, False, False, False, False, True,]]
     of shape (n_groups, x_dim)
    """
    assert mode in ["permute_current", "permute_below"]
    permutation_mask = np.zeros((len(np.unique(hierarchy)), len(hierarchy)), dtype=bool)
    for i, level in enumerate(np.unique(hierarchy)):
        if mode == "permute_current":
            permutation_mask[i] = hierarchy == level
        elif mode == "permute_below":
            permutation_mask[i] = hierarchy <= level
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


def contrib_permute_current(
    X,
    y,
    hierarchy=None,
    cv=KFold(5),
    metric=r_metric,
    groups=None,
    model=RidgeCV(np.logspace(-2, 8, 20)),
    n_repeats=50,
):
    """
    Method A

    cf. run_experiment
    features groups are indicated in hierarchy (e.g [0, 0, 1, 2, 2]),
    of the same len as x_dim.
    (groups refers to CV groups for GroupKFold)

    Returns
    array of shape (n_groups, y_dim)
        with n_groups = len(unique(hierarchy))
        importance[i, j] gives the contribution of features
        group i for th prediction of y_j.
    """
    # Init model
    orthogonalizer = HierarchOrthogonalizer(model, hierarchy)
    pipeline = make_pipeline(orthogonalizer, model)

    # Cross-validation loop
    importance = []
    for train, test in cv.split(y, groups=groups):
        print(".", end="")

        # Fit model
        pipeline.fit(X[train], y[train])

        # Compute permutation importance on test
        permutation_mask = _get_permutation_mask(hierarchy, mode="permute_current")
        score = _get_permutation_score(
            pipeline,
            X[test],
            y[test],
            permutation_mask,
            metric=metric,
            n_repeats=n_repeats,
        )
        importance.append(score)

    # Average accross splits
    importance = np.stack(importance)
    importance = importance.mean(0)

    return importance


def contrib_permute_below(
    X,
    y,
    cv=KFold(5),
    hierarchy=None,
    metric=r_metric,
    groups=None,
    model=RidgeCV(np.logspace(-2, 8, 20)),
    n_repeats=50,
):
    """
    Method B

    cf. run_experiment
    features groups are indicated in hierarchy (e.g [0, 0, 1, 2, 2]),
    of the same len as x_dim.
    (groups refers to CV groups for GroupKFold)
    
    Returns
    array of shape (n_groups, y_dim)
        with n_groups = len(unique(hierarchy))
        importance[i, j] gives the contribution of features
        group i for th prediction of y_j.
    """

    # Cross-validation loop
    importance = []
    for train, test in cv.split(y, groups=groups):
        print(".", end="")

        # Fit model
        model.fit(X[train], y[train])

        # Compute permutation importance on test
        permutation_mask = _get_permutation_mask(hierarchy, mode="permute_below")
        score = _get_permutation_score(
            model,
            X[test],
            y[test],
            permutation_mask,
            metric=metric,
            n_repeats=n_repeats,
        )
        # Orthogonalize contribution
        score[1:] -= score[:-1]

        importance.append(score)

    # Average accross splits
    importance = np.stack(importance)
    importance = importance.mean(0)

    return importance


def contrib_concat(
    X,
    y,
    cv=KFold(5),
    hierarchy=None,
    metric=r_metric,
    groups=None,
    model=RidgeCV(np.logspace(-2, 8, 20)),
    pca=0,
):
    """
    Method C

    cf. run_experiment
    features groups are indicated in hierarchy (e.g [0, 0, 1, 2, 2]),
    same len as x_dim.
    (not in groups, which refers to CV groups for GroupKFold)

    Returns
    array of shape (n_groups, y_dim)
        with n_groups = len(unique(hierarchy))
        importance[i, j] gives the contribution of features
        group i for th prediction of y_j.
    """

    # Cross validation loop
    n_levels = len(np.unique(hierarchy))
    score = np.zeros((cv.n_splits, n_levels, *y.shape[1:]))
    for i, level in enumerate(np.unique(hierarchy)):

        # Select
        below = np.where(hierarchy <= level)[0]
        X_below = X[:, below].copy()

        # Apply PCA if needed
        if pca > 0 and pca < X_below.shape[1]:
            # X_below = StandardScaler().fit_transform(X_below)
            X_below = PCA(pca).fit_transform(X_below)

        for split, (train, test) in enumerate(cv.split(y, groups=groups)):

            # Fit
            model.fit(X_below[train], y[train])

            # Predict
            pred = model.predict(X_below[test])

            # Score
            score[split, i] = metric(pred, y[test])

    # Average accross splits
    score = score.mean(0)

    # Orthogonalize contributions
    score[1:] -= score[:-1]

    return score


def contrib_permute_above_train(
    X,
    y,
    cv=KFold(5),
    hierarchy=None,
    metric=r_metric,
    groups=None,
    n_repeats=10,
    model=RidgeCV(np.logspace(-2, 8, 20)),
):
    """
    Method D

    Returns
    array of shape (n_groups, y_dim)
        with n_groups = len(unique(hierarchy))
        importance[i, j] gives the contribution of features
        group i for th prediction of y_j.
    """

    # Cross validation loop
    n_levels = len(np.unique(hierarchy))
    score = np.zeros((n_repeats, cv.n_splits, n_levels, *y.shape[1:]))

    for split, (train, test) in enumerate(cv.split(y, groups=groups)):

        for i, level in enumerate(np.unique(hierarchy)):

            for repeat in range(n_repeats):

                # Permute
                X_permuted = X.copy()
                above = np.where(hierarchy > level)[0]
                X_permuted[:, above] = shuffle_c(X[:, above])

                # Fit
                model.fit(X_permuted[train], y[train])

                # Predict
                pred = model.predict(X_permuted[test])

                # Score
                score[repeat, split, i] = metric(pred, y[test])

    # Average accross splits and repeats
    score = score.mean((0, 1))

    # Orthogonalize contributions
    score[1:] -= score[:-1]

    return score
