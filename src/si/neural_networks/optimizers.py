from abc import ABCMeta, abstractmethod

import numpy as np


class Optimizer(metaclass=ABCMeta):
    """
    Classe base (abstrata) para os optimizers.

    Um optimizer decide como os parâmetros (pesos ou biases) de uma camada mudam
    a partir do gradiente da loss. Cada DenseLayer recebe duas cópias do optimizer
    no initialize (w_opt para os pesos, b_opt para os biases) e chama o update
    no backward propagation.

    Parameters
    ----------
    learning_rate: float
        A taxa de aprendizagem: o tamanho do passo em cada atualização.
    """

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Atualiza os parâmetros (pesos ou biases) de uma camada.

        Parameters
        ----------
        w: np.ndarray
            Os parâmetros atuais.
        grad_loss_w: np.ndarray
            O gradiente da loss em relação a esses parâmetros (mesma shape de w).

        Returns
        -------
        w: np.ndarray
            Os parâmetros atualizados.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent com momentum.

    Em vez de usar só o gradiente atual, usa uma média dos gradientes anteriores
    (o retained_gradient). Se os gradientes apontam sempre para o mesmo lado, a média
    mantém essa direção; se trocam de sinal a cada passo (zigue-zague), anulam-se em
    parte e o treino fica mais estável.

    Parameters
    ----------
    learning_rate: float
        A taxa de aprendizagem (α).
    momentum: float
        O momentum (β), entre 0 e 1. Com 0 é o gradient descent normal.

    Attributes
    ----------
    retained_gradient: np.ndarray
        A média dos gradientes (V), com a mesma shape dos parâmetros.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        # o __init__ da Optimizer guarda o learning_rate
        super().__init__(learning_rate)
        self.momentum = momentum

        # atributo estimado: a shape só se sabe quando chegar o primeiro gradiente
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Atualiza o retained_gradient e devolve os parâmetros atualizados:
            V_t = β * V_(t-1) + (1 - β) * dL/dW
            W_t = W_(t-1) - α * V_t

        Parameters
        ----------
        w: np.ndarray
            Os parâmetros atuais (pesos ou biases).
        grad_loss_w: np.ndarray
            O gradiente da loss em relação a w.

        Returns
        -------
        w: np.ndarray
            Os parâmetros atualizados.
        """
        # na primeira chamada ainda não há gradientes anteriores: começa a zeros,
        # com a shape dos parâmetros (pesos (n_features, n_units) ou biases (1, n_units))
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))

        # media móvel: uma parte β do que vinha de trás + uma parte (1 - β) do gradiente atual
        # com β = 0 fica só o gradiente atual (gradient descent sem momentum)
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w

        # passo no sentido contrário: o gradiente aponta para onde a loss sobe, por isso subtrai-se
        return w - self.learning_rate * self.retained_gradient