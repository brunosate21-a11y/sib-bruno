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