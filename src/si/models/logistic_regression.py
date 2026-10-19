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

    def _fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Estima theta, theta_zero, mean, std e cost_history com gradient descent.

        Parameters
        ----------
        dataset: Dataset
            O dataset de treino.

        Returns
        -------
        self: LogisticRegression
            O modelo treinado.
        """
        # normalizar os dados, se pedido: x' = (x - média) / desvio padrão
        # a média e o std são calculados SÓ no treino e guardados no modelo,
        # para que o _predict use exatamente os mesmos valores (evita data leakage)
        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            # uma feature constante tem std = 0, o que daria divisão por zero (NaN)
            self.std[self.std == 0] = 1
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X
        y = dataset.y

        # m = número de amostras, n = número de features
        m, n = X.shape

        # iniciar os parâmetros a zero: com theta = 0, sigmoid(0) = 0.5 para todas as amostras,
        # ou seja, o modelo começa "sem opinião" (50% de probabilidade para cada classe)
        self.theta = np.zeros(n)
        self.theta_zero = 0
        self.cost_history = {}

        i = 0
        early_stopping = 0  # contador de iterações seguidas sem melhoria do custo

        # repetir até atingir max_iter ou até o custo deixar de melhorar durante 'patience' iterações
        while i < self.max_iter and early_stopping < self.patience:

            # prever y com a sigmoide: ÚNICA diferença em relação à RidgeRegression
            #    z = X · theta + theta_zero é a parte linear (igual à Ridge);
            #    a sigmoide transforma z na probabilidade de cada amostra ser da classe 1
            y_pred = sigmoid_function(np.dot(X, self.theta) + self.theta_zero)

            # gradiente do erro, já multiplicado pela learning rate:
            #    alpha * (1/m) * soma((y_pred - y) * x_j) -> a fórmula é IGUAL à da Ridge,
            #    porque a derivada da sigmoide "cancela-se" com a derivada do log da função de custo
            gradient = (self.alpha / m) * np.dot(y_pred - y, X)

            # termo da regularização L2: "encolhe" cada theta um pouco em cada iteração
            #    theta_j * (1 - alpha * lambda / m)
            #    na regressão logística isto é essencial: se as classes forem separáveis,
            #    sem L2 os thetas cresceriam sem limite (para empurrar as probabilidades para 0 e 1)
            penalization_term = self.theta * (1 - self.alpha * (self.l2_penalty / m))

            # atualizar theta: primeiro encolhe (L2), depois dá o passo do gradiente
            self.theta = penalization_term - gradient

            # atualizar theta_zero: não é penalizado (o intercept não deve ser encolhido);
            #    como x0 = 1 para todas as amostras, o gradiente fica só a soma dos erros
            self.theta_zero = self.theta_zero - (self.alpha / m) * np.sum(y_pred - y)

            # calcular o custo (log-loss + L2) com os novos thetas e guardá-lo no histórico
            self.cost_history[i] = self.cost(dataset)

            # patience: se o custo não desceu em relação à iteração anterior, conta mais uma;
            # se desceu, o contador volta a zero (só contam iterações SEGUIDAS sem melhoria)
            if i > 0 and self.cost_history[i] >= self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0

            i += 1

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Prevê a classe (0 ou 1) de cada amostra com os coeficientes estimados no _fit.

        Parameters
        ----------
        dataset: Dataset
            O dataset para o qual queremos prever y.

        Returns
        -------
        predictions: np.ndarray
            As classes previstas (0 ou 1).
        """
        # normalizar com a média e o std guardados no _fit (NÃO recalcular com os dados novos):
        # o modelo aprendeu os thetas numa escala concreta, por isso os dados novos
        # têm de ser transformados exatamente da mesma forma
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        # probabilidade de cada amostra ser da classe 1: sigmoid(X · theta + theta_zero)
        probabilities = sigmoid_function(np.dot(X, self.theta) + self.theta_zero)

        # converter as probabilidades em classes: prob >= 0.5 -> 1, prob < 0.5 -> 0
        # o limiar 0.5 corresponde a z = 0, ou seja, à fronteira de decisão X · theta + theta_zero = 0
        return np.where(probabilities >= 0.5, 1, 0)