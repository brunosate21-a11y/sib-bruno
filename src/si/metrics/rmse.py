import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Root Mean Squared Error (RMSE) between the true and predicted values.

    RMSE = sqrt( sum((y_true_i - y_pred_i)^2) / N )

    Parameters
    ----------
    y_true: numpy.ndarray
        The real values of y.
    y_pred: numpy.ndarray
        The predicted values of y.

    Returns
    -------
    rmse: float
        The error between y_true and y_pred.
    """
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))
