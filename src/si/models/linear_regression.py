import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegression(Model):
    """
    A RidgeRegression é um modelo linear com regularização L2.
    Resolve o problema da regressão linear com uma versão adaptada do Gradient Descent.

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
        Os coeficientes do modelo linear, um por feature.
        Por exemplo: x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        O coeficiente zero do modelo (intercept).
        Por exemplo: theta_zero * 1
    mean: np.ndarray
        A média de cada feature do dataset de treino.
    std: np.ndarray
        O desvio padrão de cada feature do dataset de treino.
    cost_history: dict
        O valor da função de custo em cada iteração {iteração: custo}.
    """

    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000,
                 patience: int = 5, scale: bool = True, **kwargs):
        # chamar o __init__ do Model (e, através dele, do Estimator),
        # que da start ao estado de "treinado" do modelo
        super().__init__(**kwargs)

        # parâmetros: definidos pelo utilizador ao criar o modelo
        self.l2_penalty = l2_penalty  # força da penalização L2: quanto maior, mais pequenos ficam os thetas
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


    def _fit(self, dataset: Dataset) -> 'RidgeRegression':
        """
        Estima theta, theta_zero, mean, std e cost_history com gradient descent.

        Parameters
        ----------
        dataset: Dataset
            O dataset de treino.

        Returns
        -------
        self: RidgeRegression
            O modelo treinado.
        """
        # 1. normalizar os dados, se pedido: x' = (x - média) / desvio padrão
        # a média e o std são calculados SÓ no treino e guardados no modelo,
        # para que o _predict use exatamente os mesmos valores (evita data leakage)
        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            # uma feature constante tem std = 0, o que daria divisão por zero (NaN);
            # pôr std = 1 deixa essa feature apenas centrada em 0
            self.std[self.std == 0] = 1
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X
        y = dataset.y

        # m = número de amostras, n = número de features
        m, n = X.shape

        # iniciar os parâmetros do modelo: começa com todos os coeficientes a zero
        self.theta = np.zeros(n)
        self.theta_zero = 0
        self.cost_history = {}

        i = 0
        early_stopping = 0  # contador de iterações seguidas sem melhoria do custo

        # 8. repetir até atingir max_iter ou até o custo deixar de melhorar durante 'patience' iterações
        while i < self.max_iter and early_stopping < self.patience:

            # 2. prever y com os thetas atuais: y_pred = X · theta + theta_zero
            y_pred = np.dot(X, self.theta) + self.theta_zero

            # 3. gradiente do erro, já multiplicado pela learning rate:
            #    alpha * (1/m) * soma((y_pred - y) * x_j), calculado para todas as features de uma vez
            gradient = (self.alpha / m) * np.dot(y_pred - y, X)

            # 4. termo da regularização L2: "encolhe" cada theta um pouco em cada iteração
            #    theta_j * (1 - alpha * lambda / m)
            penalization_term = self.theta * (1 - self.alpha * (self.l2_penalty / m))

            # 5. atualizar theta: primeiro encolhe (L2), depois dá o passo do gradiente
            self.theta = penalization_term - gradient

            # 6. atualizar theta_zero: não é penalizado (o intercept não deve ser encolhido);
            #    como x0 = 1 para todas as amostras, o gradiente fica só a soma dos erros
            self.theta_zero = self.theta_zero - (self.alpha / m) * np.sum(y_pred - y)

            # 7. calcular o custo com os novos thetas e guardá-lo no histórico
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
        Prevê a variável dependente (y) com os coeficientes estimados no _fit.

        Parameters
        ----------
        dataset: Dataset
            O dataset para o qual queremos prever y.

        Returns
        -------
        predictions: np.ndarray
            Os valores previstos de y.
        """
        # normalizar com a média e o std guardados no _fit (NÃO recalcular com os dados novos):
        # o modelo aprendeu os thetas numa escala concreta, por isso os dados novos
        # têm de ser transformados exatamente da mesma forma
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        # h(x) = theta_zero + theta_1 * x_1 + ... + theta_n * x_n
        # np.dot faz esta soma para todas as amostras de uma vez
        return np.dot(X, self.theta) + self.theta_zero

    def _score(self, dataset: Dataset) -> float:
        """
        Calcula o erro (MSE) entre os valores reais e os previstos de y.

        Parameters
        ----------
        dataset: Dataset
            O dataset com os valores reais de y.

        Returns
        -------
        mse: float
            O erro quadrático médio do modelo.
        """
        # 1. prever y com os thetas estimados
        predictions = self.predict(dataset)

        # 2. comparar com os valores reais através do MSE (quanto mais baixo, melhor)
        return mse(dataset.y, predictions)

    def cost(self, dataset: Dataset) -> float:
        """
        Calcula a função de custo J (erro + penalização L2) do modelo no dataset.

        Parameters
        ----------
        dataset: Dataset
            O dataset onde calcular o custo.

        Returns
        -------
        cost: float
            O valor da função de custo.
        """
        # 1. prever y com os thetas atuais
        # usa-se _predict (e não predict) porque o cost é chamado DENTRO do _fit,
        # quando o modelo ainda não está marcado como treinado: o predict() lançaria
        # o ValueError do is_fitted() a meio do treino
        y_pred = self._predict(dataset)

        # m = número de amostras
        m = len(dataset.y)

        # 2. J = (1 / 2m) * [soma dos erros ao quadrado + lambda * soma dos thetas ao quadrado]
        # o theta_zero não entra na penalização (o somatório começa em j = 1)
        return (np.sum((y_pred - dataset.y) ** 2) + self.l2_penalty * np.sum(self.theta ** 2)) / (2 * m)