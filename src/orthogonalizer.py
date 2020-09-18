import numpy as np
from sklearn.base import TransformerMixin, clone


class HierarchOrthogonalizer(TransformerMixin):
    def __init__(self, model, hierarchy):
        self.model = model
        self.hierarchy = hierarchy  # [0, 0, 0, 1, 1, 1, 2, 3, 4,]

    def fit(self, X, y=None):
        self.models_ = list()
        for level in np.unique(self.hierarchy)[:-1]:
            below = np.where(self.hierarchy == level)[0]
            current = np.where(self.hierarchy == (level + 1))[0]

            model = clone(self.model)
            model.fit(X[:, below], X[:, current])

            self.models_.append(model)
        return self

    def transform(self, X, y=None):
        X_hierarch = np.zeros_like(X)
        first_level = np.where(self.hierarchy == 0)[0]
        X_hierarch[:, first_level] = X[:, first_level].copy()
        for level, model in enumerate(self.models_):

            below = np.where(self.hierarchy == level)[0]
            current = np.where(self.hierarchy == (level + 1))[0]

            X_hierarch[:, current] = X[:, current] - model.predict(X[:, below])
        return X_hierarch

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
