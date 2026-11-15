from abc import ABCMeta, abstractmethod

import numpy as np


class LossFunction(metaclass=ABCMeta):
    """
    Classe base (abstrata) para as funções de perda (loss).

    A loss mede a diferença entre o que a rede previu (y_pred) e o valor real (y_true).
    O treino tenta baixá-la: a derivada da loss é o primeiro erro (dE/dY) que entra
    no backward propagation, pela última camada.

    Não tem __init__: não guarda parâmetros, só define os dois métodos
    que cada loss concreta é obrigada a implementar.
    """

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula o valor da loss a partir dos valores reais e dos previstos.

        Parameters
        ----------
        y_true: np.ndarray
            Os valores reais.
        y_pred: np.ndarray
            Os valores previstos pela rede.

        Returns
        -------
        loss: float
            O valor da loss (quanto menor, melhor).
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calcula a derivada da loss em relação às previsões (dE/dY).

        Parameters
        ----------
        y_true: np.ndarray
            Os valores reais.
        y_pred: np.ndarray
            Os valores previstos pela rede.

        Returns
        -------
        derivative: np.ndarray
            Matriz com a mesma shape de y_pred: o erro que entra no backward
            propagation pela última camada.
        """
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """
    Erro quadrático médio: L = (1/n) * soma((y_true - y_pred)^2).

    Usa-se em regressão: mede a distância média (ao quadrado) entre o previsto e o real.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula o MSE entre os valores reais e os previstos.
        """
        # o quadrado impede que erros positivos e negativos se anulem e castiga mais os erros grandes
        # np.mean divide pelo número total de valores (n)
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Derivada do MSE em relação às previsões: (2/n) * (y_pred - y_true).
        """
        # derivada de (y_true - y_pred)^2 em ordem a y_pred = 2 * (y_true - y_pred) * (-1)
        # escrever (y_pred - y_true) já inclui esse -1
        # y_true.size = n, o mesmo n pelo qual o np.mean divide na loss
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    """
    Entropia cruzada binária: L = -soma(y * log(p) + (1 - y) * log(1 - p)).

    Usa-se em classificação binária, com uma sigmoide na última camada:
    y_true é 0 ou 1 e y_pred (p) é a probabilidade prevista de ser 1.
    Castiga muito as previsões confiantes e erradas (ex.: p = 0.01 quando y = 1).
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula a entropia cruzada binária (soma, como na fórmula do slide).
        """
        # np.clip afasta p de 0 e de 1: log(0) = -inf
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # para cada amostra só um dos termos conta:
        # y = 1 -> -log(p)      (p perto de 1 = loss perto de 0)
        # y = 0 -> -log(1 - p)  (p perto de 0 = loss perto de 0)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Derivada da entropia cruzada binária em relação às previsões: -y/p + (1 - y)/(1 - p).
        """
        # o mesmo clip: aqui evita dividir por 0 quando p = 0 ou p = 1
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # derivada de log(p) = 1/p; derivada de log(1 - p) = -1/(1 - p) (o -1 vem da regra da cadeia)
        # com o sinal - da frente da loss: -y/p + (1 - y)/(1 - p)
        return -y_true / p + (1 - y_true) / (1 - p)


class CategoricalCrossEntropy(LossFunction):
    """
    Entropia cruzada categórica: L = -soma(y * log(p)).

    Usa-se em classificação multiclasse, com uma softmax na última camada:
    y_true está em one-hot (uma coluna por classe, 1 na classe certa e 0 nas outras)
    e y_pred (p) tem a probabilidade prevista de cada classe.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula a entropia cruzada categórica (soma, como na fórmula do slide).
        """
        # np.clip afasta p de 0 e de 1: log(0) = -inf
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # com y em one-hot, em cada amostra só conta a coluna da classe certa (as outras têm y = 0):
        # a loss de cada amostra é -log(probabilidade dada à classe certa)
        return -np.sum(y_true * np.log(p))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Derivada da entropia cruzada categórica em relação às previsões: -y/p.
        """
        # o mesmo clip: aqui evita dividir por 0 quando p = 0
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # derivada de log(p) = 1/p; com o sinal - da frente da loss: -y/p
        # nas colunas das classes erradas (y = 0) a derivada é 0
        return -y_true / p