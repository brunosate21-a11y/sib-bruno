import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula o erro quadrático médio (Mean Squared Error) entre os valores reais e os previstos.

    MSE = (1/n) * soma((y_true_i - y_pred_i)^2)

    Parameters
    ----------
    y_true: np.ndarray
        Os valores reais de y.
    y_pred: np.ndarray
        Os valores previstos de y.

    Returns
    -------
    mse: float
        O erro entre y_true e y_pred.
    """
    # (y_true - y_pred): diferença entre real e previsto para cada amostra (operação vetorizada do numpy)
    # ** 2: eleva ao quadrado para que erros positivos e negativos não se anulem
    #   e para penalizar mais os erros grandes
    # np.sum(...) / len(y_true): soma os erros e divide pelo número de amostras (n) -> média
    return np.sum((y_true - y_pred) ** 2) / len(y_true)