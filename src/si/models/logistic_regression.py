import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegression(Model):
    """
    A LogisticRegression é um modelo logístico com regularização L2.
    Resolve o problema da regressão logística com uma versão adaptada do Gradient Descent.

    Parameters
    ----------
    l2_penalty: float
        O parâmetro de regularização L2 (lambda).
    alpha: float
        A taxa de aprendizagem (learning rate).
    max_iter: int
        O número máximo de iterações.
    patience: int
        O número máximo de iterações seguidas sem melhoria do custo.
    scale: bool
        Se os dados devem ser normalizados ou não.

    Attributes
    ----------
    theta: np.ndarray
        Os coeficientes do modelo logístico, um por feature.
        Por exemplo: sigmoid(x0 * theta[0] + x1 * theta[1] + ...)
    theta_zero: float
        O coeficiente zero do modelo (intercept).
    mean: np.ndarray
        A média de cada feature do dataset de treino.
    std: np.ndarray
        O desvio padrão de cada feature do dataset de treino.
    cost_history: dict
        O valor da função de custo em cada iteração {iteração: custo}.
    """

    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000,
                 patience: int = 5, scale: bool = True, **kwargs):
        # chama o __init__ do Model (e, através dele, do Estimator),
        # que dá início ao estado de "treinado" do modelo
        super().__init__(**kwargs)

        # parâmetros: definidos pelo utilizador ao criar o modelo (iguais aos da RidgeRegression)
        self.l2_penalty = l2_penalty  # força da penalização L2: quanto maior, mais pequenos ficam os thetas
                                      # no sklearn usa-se C = 1 / l2_penalty, por isso lá C maior = MENOS penalização
        self.alpha = alpha            # tamanho de cada passo do gradient descent
        self.max_iter = max_iter      # limite de iterações do treino
        self.patience = patience      # early stopping: pára se o custo não melhorar durante 'patience' iterações seguidas
        self.scale = scale            # normalizar as features antes de treinar

        # atributos estimados: começam a None e só são preenchidos no _fit
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}