import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
    Ridge Regression (modelo linear com regularização L2) resolvida de forma analítica,
    com a solução fechada dos mínimos quadrados:

        theta = (X^T X + lambda * I')^-1 X^T y

    em que I' é a matriz identidade com I'[0, 0] = 0, para o intercept não ser penalizado.

    Parameters
    ----------
    l2_penalty: float
        O parâmetro de regularização L2 (lambda).
    scale: bool
        Se os dados devem ser normalizados ou não.

    Attributes
    ----------
    theta: np.ndarray
        Os coeficientes do modelo, um por feature.
    theta_zero: float
        O coeficiente zero do modelo (intercept).
    mean: np.ndarray
        A média de cada feature do dataset de treino.
    std: np.ndarray
        O desvio padrão de cada feature do dataset de treino.
    """

    def __init__(self, l2_penalty: float = 1, scale: bool = True, **kwargs):
        # chamar o __init__ do Model, início ao estado de "treinado" do modelo
        super().__init__(**kwargs)

        # parâmetros: não há alpha, max_iter nem patience,
        # porque a solução é calculada de uma vez só (não é iterativa)
        self.l2_penalty = l2_penalty
        self.scale = scale

        # atributos estimados: começam None e só são preenchidos no _fit
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Estima theta, theta_zero, mean e std com a solução fechada dos mínimos quadrados.

        Parameters
        ----------
        dataset: Dataset
            O dataset de treino.

        Returns
        -------
        self: RidgeRegressionLeastSquares
            O modelo treinado.
        """
        # normalizar os dados, igual à RidgeRegression
        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            # evitar divisão por zero em features constantes
            self.std[self.std == 0] = 1
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        # acrescentar uma coluna de uns no início de X:
        # assim o theta_zero passa a ser só mais um coeficiente (multiplicado por x0 = 1)
        # e todos os thetas são calculados juntos, numa única fórmula
        X = np.c_[np.ones(X.shape[0]), X]

        # matriz de penalização: lambda * matriz identidade
        # tamanho (n_features + 1) x (n_features + 1), por causa da coluna de uns
        penalty_matrix = self.l2_penalty * np.eye(X.shape[1])

        # pôr a primeira posição a 0, para o theta_zero (intercept) não ser penalizado,
        # tal como na RidgeRegression, em que o theta_zero era atualizado sem o termo L2
        penalty_matrix[0, 0] = 0

        # solução fechada: thetas = (X^T X + penalização)^-1 X^T y
        thetas = np.linalg.inv(X.T.dot(X) + penalty_matrix).dot(X.T).dot(dataset.y)

        # o primeiro elemento corresponde à coluna de uns (theta_zero); os restantes são os theta
        self.theta_zero = thetas[0]
        self.theta = thetas[1:]

        return self

    