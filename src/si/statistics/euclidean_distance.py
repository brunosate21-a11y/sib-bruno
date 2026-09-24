import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the euclidean distance between a single sample x and multiple samples y.

    Parameters
    ----------
    x: numpy.ndarray (n_features,)
        A single sample.
    y: numpy.ndarray (n_samples, n_features)
        Multiple samples.

    Returns
    -------
    distances: numpy.ndarray (n_samples,)
        The euclidean distance between x and each sample in y.
    """
    return np.sqrt(((x - y) ** 2).sum(axis=1))
