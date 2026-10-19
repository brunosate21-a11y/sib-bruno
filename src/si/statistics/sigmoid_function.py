import numpy as np


def sigmoid_function(X: np.ndarray) -> np.ndarray:
    """
    Calcula a função sigmoide para os valores de X.

    Parameters
    ----------
    X: np.ndarray
        Os valores de entrada.

    Returns
    -------
    sigmoid: np.ndarray
        A probabilidade de cada valor ser 1 (valores entre 0 e 1).
    """
    # f(x) = 1 / (1 + e^(-x))
    # transforma qualquer número real num valor entre 0 e 1:
    # x muito negativo -> perto de 0; x = 0 -> 0.5; x muito positivo -> perto de 1
    # é esta função que faz a regressão logística devolver probabilidades
    # (uma SVM, por exemplo, devolve só uma distância à fronteira, não uma probabilidade)
    return 1 / (1 + np.exp(-X))

