import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the accuracy between the true and predicted labels.

    Parameters
    ----------
    y_true: numpy.ndarray
        The true label values.
    y_pred: numpy.ndarray
        The predicted label values.

    Returns
    -------
    accuracy: float
        The portion of well classified samples.
    """
    # divide the number of well classified labels by the total number of labels
    return np.sum(y_true == y_pred) / len(y_true)
