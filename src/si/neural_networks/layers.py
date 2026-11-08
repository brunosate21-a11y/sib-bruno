import copy
from abc import ABCMeta, abstractmethod

import numpy as np


class Layer(metaclass=ABCMeta):
    """
    Classe base (abstrata) para todas as camadas de uma rede neuronal.

    Não tem __init__: só define os métodos que todas as camadas partilham
    (input_shape, layer_name) e os métodos abstratos que cada camada concreta
    (DenseLayer, Dropout, ...) é obrigada a implementar.
    Por ser abstrata, não se pode fazer Layer() diretamente: dá TypeError.
    """

    @abstractmethod
    def forward_propagation(self, input, training):
        """
        Calcula o output da camada a partir do input.

        Parameters
        ----------
        input: np.ndarray
            O input da camada.
        training: bool
            Se a rede está em treino ou não (algumas camadas, como o Dropout,
            comportam-se de forma diferente em treino e em previsão).

        Returns
        -------
        output: np.ndarray
            O output da camada.
        """
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error):
        """
        Recebe o erro do output da camada (dE/dY), atualiza os parâmetros
        (se a camada os tiver) e devolve o erro do input (dE/dX),
        que passa a ser o dE/dY da camada anterior.

        Parameters
        ----------
        output_error: np.ndarray
            O gradiente do erro em relação ao output da camada (dE/dY).

        Returns
        -------
        input_error: np.ndarray
            O gradiente do erro em relação ao input da camada (dE/dX).
        """
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        """
        Devolve a shape do output da camada (sem a dimensão das amostras).
        """
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        """
        Devolve o número de parâmetros treináveis da camada.
        """
        raise NotImplementedError

    def set_input_shape(self, input_shape):
        """
        Define a shape do input da camada.
        A rede vai usar isto para ligar cada camada ao output_shape da anterior.
        """
        self._input_shape = input_shape

    def input_shape(self):
        """
        Devolve a shape do input da camada.
        """
        return self._input_shape

    def layer_name(self):
        """
        Devolve o nome da camada (o nome da classe, ex.: 'DenseLayer').
        """
        return self.__class__.__name__


class DenseLayer(Layer):
    """
    Camada densa (fully connected): cada neurónio está ligado a todos os
    neurónios da camada anterior.

    Parameters
    ----------
    n_units: int
        O número de neurónios da camada.
    input_shape: tuple
        A shape do input da camada. Para dados tabulares é (n_features,).

    Attributes
    ----------
    input: np.ndarray
        O último input recebido (fica guardado para o backward propagation).
    output: np.ndarray
        O último output calculado.
    weights: np.ndarray
        Os pesos da camada, matriz (n_features, n_units).
        Linha i = pesos que saem da feature i; coluna j = pesos que entram no neurónio j.
    biases: np.ndarray
        Os biases da camada, matriz (1, n_units): um bias por neurónio.
            w_opt: Optimizer
                O optimizer que atualiza os pesos no backward propagation.
            b_opt: Optimizer
                O optimizer que atualiza os biases no backward propagation. 
    """

    def __init__(self, n_units: int, input_shape: tuple = None):
        # parâmetros: definidos pelo utilizador ao criar a camada
        self.n_units = n_units
        self._input_shape = input_shape  # mesmo atributo que o Layer usa no input_shape()

        # atributos estimados: começam no None e são preenchidos no initialize / forward
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None
        self.w_opt = None  # optimizer dos pesos (criado no initialize)
        self.b_opt = None  # optimizer dos biases (criado no initialize)

    def initialize(self, optimizer=None) -> 'DenseLayer':
        """
        Inicializa os pesos com valores aleatórios em [-0.5, 0.5[, os biases a zero
        e guarda uma cópia do optimizer para os pesos e outra para os biases.

        Parameters
        ----------
        optimizer: Optimizer
            O optimizer que vai atualizar os pesos e os biases no backward propagation.
            É opcional (None) para a camada continuar a funcionar só com o forward,
            como nos testes da Aula 4, antes de existirem optimizers.

        Returns
        -------
        self: DenseLayer
            A própria camada (permite fazer DenseLayer(...).initialize(optimizer)).
        """
        if self._input_shape is None:
            raise ValueError("A camada precisa do input_shape antes de ser inicializada.")

        n_features = self.input_shape()[0]

        # np.random.rand dá valores em [0, 1[; subtrair 0.5 centra-os em zero --> [-0.5, 0.5[
        # pesos centrados em zero, assim os neurónios não começam todos cruzados no mesmo sentido
        self.weights = np.random.rand(n_features, self.n_units) - 0.5

        # biases a zero: os pesos aleatórios são suficientes para os neurónios aprenderem coisas diferentes
        self.biases = np.zeros((1, self.n_units))

        # uma cópia independente do optimizer para os pesos e outra para os biases:
        # optimizers como o SGD com momentum ou o Adam guardam estado interno com a shape
        # dos parâmetros, por isso pesos e biases não podem partilhar o mesmo objeto
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)

        return self



    def parameters(self) -> int:
        """
        Devolve o número de parâmetros da camada (pesos + biases).

        Returns
        -------
        int
            n_features * n_units + n_units
        """
        # .size = número total de elementos de cada matriz
        return int(self.weights.size + self.biases.size)

    def forward_propagation(self, input: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Calcula o output da camada: Y = X·W + B.

        Parameters
        ----------
        input: np.ndarray
            O input da camada, matriz (n_amostras, n_features).
        training: bool
            Não é usado na camada densa (comporta-se igual em treino e previsão).

        Returns
        -------
        output: np.ndarray
            O output da camada, matriz (n_amostras, n_units).
        """
        # guarda o input: o backward propagation vai precisar dele para calcular o gradiente dos pesos
        self.input = input

        # (n_amostras, n_features) · (n_features, n_units) -> (n_amostras, n_units)
        # o bias (1, n_units) é somado a todas as linhas por broadcasting
        self.output = np.dot(input, self.weights) + self.biases
        return self.output


    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:

        """
        Calcula os gradientes da camada, atualiza os pesos e os biases
        e devolve o erro do input para a camada anterior.

        Parameters
        ----------
        output_error: np.ndarray
            dE/dY, matriz (n_amostras, n_units), vinda da camada seguinte.

        Returns
        -------
        input_error: np.ndarray
            dE/dX, matriz (n_amostras, n_features); vai ser o dE/dY da camada anterior.
            
        """
        # 1. dE/dX = dE/dY · Wᵀ  -->  (n_amostras, n_units) · (n_units, n_features) = (n_amostras, n_features)
        # tem de ser calculado ANTES de atualizar os pesos: senão usava os pesos novos
        # e o resultado ficava errado sem dar nenhuma mensagem de erro
        input_error = np.dot(output_error, self.weights.T)

        # 2. dE/dW = Xᵀ · dE/dY  -->  (n_features, n_amostras) · (n_amostras, n_units) = (n_features, n_units)
        # o produto de matrizes já soma as contribuições de todas as amostras do batch
        weights_error = np.dot(self.input.T, output_error)

        # 3. dE/dB = soma de dE/dY ao longo das amostras  ->  (1, n_units)
        # o mesmo bias é somado a todas as amostras (broadcasting), por isso o gradiente junta-as todas
        # keepdims=True mantém a shape (1, n_units), igual à de self.biases
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        # 4. atualizar os parâmetros com o optimizer (SGD, Adam, ... na próxima aula)
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)

        # 5. devolver dE/dX para a camada anterior continuar o backward
        return input_error

    

    def output_shape(self) -> tuple:
        """
        Devolve a shape do output da camada: (n_units,).
        """
        return (self.n_units,)