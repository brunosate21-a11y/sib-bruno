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
    # a rede neuronal devolve probabilidades em matrizes 2D, e não rótulos:
    # _to_labels converte-as em rótulos; os vetores 1D (como os dos outros modelos) ficam iguais
    y_true = _to_labels(y_true)
    y_pred = _to_labels(y_pred)

    # divide the number of well classified labels by the total number of labels
    return np.sum(y_true == y_pred) / len(y_true)


def _to_labels(y: np.ndarray) -> np.ndarray:
    """
    Converte previsões da rede neuronal (matrizes 2D) em rótulos (vetor 1D).

    Parameters
    ----------
    y: np.ndarray
        Rótulos (n,), probabilidades da classe 1 (n, 1) ou uma probabilidade por classe (n, k).

    Returns
    -------
    labels: np.ndarray
        Os rótulos, num vetor (n,).
    """
    y = np.asarray(y)
    if y.ndim == 2:
        if y.shape[1] == 1:
            # binária: uma coluna com a probabilidade da classe 1 -> arredondar (0.95 -> 1, 0.05 -> 0)
            # ravel passa de (n, 1) para (n,)
            return np.round(y).ravel()
        # multiclasse: uma coluna por classe -> o índice da coluna com a maior probabilidade
        return np.argmax(y, axis=1)
    # já são rótulos: ficam iguais
    return y