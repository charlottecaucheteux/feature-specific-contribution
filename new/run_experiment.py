import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .cross_validate import cross_validate
from .orthogonalizer import HierarchOrthogonalizer
from .scorer import PermutationScorer, Scorer
from .utils import r_metric


def apply_scaler(X, hierarchy, scaler=StandardScaler()):
    assert len(hierarchy) == X.shape[1]
    X_scale = np.zeros_like(X)
    for level in np.unique(hierarchy):
        current = np.where(hierarchy == level)[0]
        X_scale[:, current] = scaler.fit_transform(X[:, current])
    return X_scale


def apply_pca(X, hierarchy, pca):
    assert len(hierarchy) == X.shape[1]
    hierarchy_pca = []

    X_pca = np.zeros_like(X)
    start = 0
    for level in np.unique(hierarchy):
        current = np.where(hierarchy == level)[0]
        dim = min(len(current), pca)
        hierarchy_pca.extend([level] * dim)
        end = start + dim
        X_pca[:, start:end] = PCA(dim).fit_transform(X[:, current])
        start += dim
    hierarchy_pca = np.array(hierarchy_pca)
    return X_pca, hierarchy_pca


def run_experiment(
    exp,
    X,
    y,
    hierarchy,
    ridge=RidgeCV(np.logspace(-2, 8, 20)),
    scale=True,
    n_repeats=50,
    pca=20,
    cv=KFold(3),
    groups=None,
    metric=r_metric,
):

    assert exp in [
        "orth_permute_current",
        "permute_below",
        "permute_above",
        "fit_hierch",
        "fit_hierch_pca",
    ]
    ridge = RidgeCV(np.logspace(-3, 8, 20))

    if scale:
        X = apply_scaler(X, hierarchy)

    if pca > 0:
        X, hierarchy = apply_pca(X, hierarchy, pca)

    if exp == "orth_permute_current":

        pipeline = make_pipeline(HierarchOrthogonalizer(ridge, hierarchy), ridge)
        scorer = PermutationScorer(
            hierarchy, metric=metric, mode="permute_current", n_repeats=n_repeats
        )
        r = cross_validate(pipeline, X, y, scorer=scorer, cv=cv, groups=groups)
        return r

    if exp in ["permute_below", "permute_above"]:
        scorer = PermutationScorer(
            hierarchy, metric=metric, mode=exp, n_repeats=n_repeats
        )
        r = cross_validate(ridge, X, y, scorer=scorer, cv=cv, groups=groups)

    if "fit_hierch" in exp:
        scorer = Scorer(metric=metric)
        r = []
        for level in np.unique(hierarchy):
            current = np.where(hierarchy <= level)[0]
            X_current = X[:, current].copy()
            if "pca" in exp:
                X_current = PCA(50).fit_transform(X_current)
            r_current = cross_validate(
                ridge, X_current, y, scorer=scorer, cv=cv, groups=groups
            )
            r.append(r_current)
        r = np.stack(r)
        r[1:] -= r[:-1]

    return r
