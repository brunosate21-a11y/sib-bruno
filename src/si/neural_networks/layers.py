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

    # nota: o backward_propagation também vai ser abstrato, mas só é adicionado
    # quando a DenseLayer o implementar; se fosse abstrato já, a DenseLayer
    # não podia ser instanciada (faltava-lhe esse método)

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

    def initialize(self) -> 'DenseLayer':
        """
        Inicializa os pesos com valores aleatórios em [-0.5, 0.5[ e os biases a zero.

        Returns
        -------
        self: DenseLayer
            A própria camada (permite fazer DenseLayer(...).initialize()).
        """
        if self._input_shape is None:
            raise ValueError("A camada precisa do input_shape antes de ser inicializada.")

        n_features = self.input_shape()[0]

        # np.random.rand dá valores em [0, 1[; subtrair 0.5 centra-os em zero -> [-0.5, 0.5[
        # pesos centrados em zero, assim os neurónios não começam todos enviesados no mesmo sentido
        self.weights = np.random.rand(n_features, self.n_units) - 0.5

        # biases a zero: os pesos aleatórios são suficientes para os neurónios aprenderem coisas diferentes
        self.biases = np.zeros((1, self.n_units))

        return self