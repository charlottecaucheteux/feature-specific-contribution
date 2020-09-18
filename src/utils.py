import numpy as np
from scipy.stats import pearsonr


def r_metric(y_pred, y_true):
    _, nc = y_pred.shape
    r = np.zeros(y_pred.shape[1])
    for i in range(nc):
        true = y_true[:, i]
        pred = y_pred[:, i]
        r[i] = pearsonr(true, pred)[0]
    return r


def shuffle_c(x):
    if x.shape[1] == 0:
        return x
    else:
        cols = [np.random.permutation(col) for col in x.T]
        cols = np.stack(cols).T
        return cols
