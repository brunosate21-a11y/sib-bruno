import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse
from si.neural_networks.losses import MeanSquaredError
from si.neural_networks.optimizers import SGD


class NeuralNetwork(Model):
    """
    Rede neuronal: uma sequência de camadas treinada com backpropagation.

    Junta todas as peças: as camadas (DenseLayer, ativações, Dropout), a loss e o optimizer.
    Em cada mini-batch faz: forward -> derivada da loss -> backward (que atualiza os pesos).

    Parameters
    ----------
    epochs: int
        O número de épocas (passagens completas pelos dados de treino).
    batch_size: int
        O número de amostras de cada mini-batch.
    optimizer: type
        A classe do optimizer (ex.: SGD), e não um objeto: a rede cria-o com o learning_rate.
    learning_rate: float
        A taxa de aprendizagem passada ao optimizer.
    verbose: bool
        Se True, mostra a loss e a métrica no fim de cada época.
    loss: type
        A classe da loss (ex.: MeanSquaredError), e não um objeto.
    metric: callable
        A função de avaliação (ex.: accuracy).
    **kwargs
        Argumentos extra para o optimizer (ex.: momentum=0.9).

    Attributes
    ----------
    layers: list
        As camadas da rede, pela ordem em que foram adicionadas.
    history: dict
        A loss e a métrica de cada época: {época: {'loss': ..., 'metric': ...}}.
    """

    def __init__(self, epochs: int = 100, batch_size: int = 128, optimizer=SGD,
                 learning_rate: float = 0.01, verbose: bool = False, loss=MeanSquaredError,
                 metric=mse, **kwargs):
        # o __init__ do Model põe _is_fitted = False (o predict e o score só funcionam depois do fit)
        super().__init__()

        # parâmetros
        self.epochs = epochs
        self.batch_size = batch_size
        # recebe-se a classe (SGD) e cria-se aqui o objeto, com o learning_rate e os kwargs (ex.: momentum)
        self.optimizer = optimizer(learning_rate=learning_rate, **kwargs)
        self.verbose = verbose
        # o mesmo para a loss: recebe-se a classe e cria-se o objeto
        self.loss = loss()
        self.metric = metric

        # atributos estimados
        self.layers = []
        self.history = {}

    def add(self, layer) -> 'NeuralNetwork':
        """
        Adiciona uma camada ao fim da rede.

        Parameters
        ----------
        layer: Layer
            A camada a adicionar.

        Returns
        -------
        self: NeuralNetwork
            A própria rede.
        """
        # ligar a camada à anterior: o input desta é o output da anterior
        # (a primeira camada não tem anterior: o input_shape tem de ser dado ao criá-la)
        if self.layers:
            layer.set_input_shape(self.layers[-1].output_shape())

        # só as camadas com parâmetros (DenseLayer) têm initialize: criam os pesos e recebem
        # o optimizer; as ativações e o Dropout não têm nada para inicializar
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)

        # guardar a camada
        self.layers.append(layer)
        return self

    def _get_mini_batches(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> list:
        """
        Divide os dados em mini-batches de batch_size amostras.

        Returns
        -------
        batches: list
            Lista de pares (X_batch, y_batch).
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        # baralhar em cada época: os batches mudam de época para época,
        # e a rede não aprende nada que dependa da ordem das amostras
        if shuffle:
            np.random.shuffle(indices)

        batches = []
        # start = 0, batch_size, 2*batch_size, ...; o último batch pode ficar mais pequeno
        for start in range(0, n_samples, self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batches.append((X[batch_indices], y[batch_indices]))
        return batches

    def _forward_propagation(self, X: np.ndarray, training: bool) -> np.ndarray:
        """
        Passa o input por todas as camadas, da primeira para a última.
        """
        output = X
        for layer in self.layers:
            # o output de uma camada é o input da seguinte
            output = layer.forward_propagation(output, training)
        return output

    def _backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Passa o erro por todas as camadas, da última para a primeira.
        """
        error = output_error
        # reversed: o erro anda para trás; cada camada devolve o dE/dX, que é o dE/dY da anterior
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error

    def _fit(self, dataset: Dataset) -> 'NeuralNetwork':
        """
        Treina a rede: para cada época, para cada mini-batch, faz
        forward -> derivada da loss -> backward (que atualiza os pesos).

        Parameters
        ----------
        dataset: Dataset
            O dataset de treino.

        Returns
        -------
        self: NeuralNetwork
            A rede treinada.
        """
        X = dataset.X
        y = dataset.y
        # y vem como vetor (n,), mas a rede devolve matrizes (n, 1): sem isto,
        # (n,) - (n, 1) dava uma matriz (n, n) por broadcasting, sem nenhum erro
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.history = {}
        for epoch in range(1, self.epochs + 1):
            # outputs e y de todos os batches da época, para calcular a loss da época no fim
            output_epoch = []
            y_epoch = []

            for X_batch, y_batch in self._get_mini_batches(X, y):
                # forward: previsão do batch (training=True por causa do Dropout)
                output = self._forward_propagation(X_batch, training=True)

                # derivada da loss: o erro inicial dE/dY, que entra pela última camada
                error = self.loss.derivative(y_batch, output)

                # backward: da última camada para a primeira; cada DenseLayer atualiza W e b
                # (os pesos mudam depois de cada mini-batch, e não só uma vez por época)
                self._backward_propagation(error)

                output_epoch.append(output)
                y_epoch.append(y_batch)

            # juntar os batches outra vez numa matriz só
            # nota: cada output foi calculado antes da atualização do seu batch,
            # por isso a loss da época é medida durante o treino
            output_epoch = np.concatenate(output_epoch)
            y_epoch = np.concatenate(y_epoch)

            # loss e métrica da época, guardadas no history
            loss = self.loss.loss(y_epoch, output_epoch)
            metric = self.metric(y_epoch, output_epoch) if self.metric is not None else None
            self.history[epoch] = {'loss': loss, 'metric': metric}

            if self.verbose:
                texto_metric = f" - {self.metric.__name__}: {metric:.4f}" if metric is not None else ""
                print(f"Época {epoch}/{self.epochs} - loss: {loss:.4f}{texto_metric}")

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Faz previsões: forward por todas as camadas, em modo de previsão.

        Parameters
        ----------
        dataset: Dataset
            O dataset a prever.

        Returns
        -------
        predictions: np.ndarray
            O output da última camada, matriz (n_amostras, n_outputs).
        """
        # training=False: o Dropout não desliga neurónios
        return self._forward_propagation(dataset.X, training=False)

    def _score(self, dataset: Dataset) -> float:
        """
        Avalia a rede com a métrica (ex.: accuracy) no dataset.

        Parameters
        ----------
        dataset: Dataset
            O dataset a avaliar.

        Returns
        -------
        score: float
            O valor da métrica.
        """
        # o mesmo cuidado do _fit: y em (n, 1), como as previsões
        y = dataset.y
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)
        return self.metric(y, self._predict(dataset))