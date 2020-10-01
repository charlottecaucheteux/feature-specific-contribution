import numpy as np
from sklearn.model_selection import KFold


def cross_validate(pipeline, X, y, cv=KFold(3), scorer=None, groups=None):
    score = []
    for train, test in cv.split(y, groups=groups):

        print(".", end="")

        # Fit pipeline
        pipeline.fit(X[train], y[train])

        # Score
        r = scorer.score(pipeline, X[test], y[test])

        score.append(r)

    score = np.stack(score)

    # Average across splits
    score = score.mean(0)
    return score
