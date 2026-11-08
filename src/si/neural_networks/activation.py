from abc import abstractmethod

import numpy as np

from si.neural_networks.layers import Layer
from si.statistics.sigmoid_function import sigmoid_function

class ActivationLayer(Layer):
    """
    Classe base (abstrata) para as camadas de ativação.

    Uma camada de ativação aplica uma função não linear a cada valor do input,
    elemento a elemento. Sem ativações, várias camadas densas seguidas equivaliam
    a uma única transformação linear, e a rede não conseguia aprender padrões não lineares.

    Não tem pesos nem biases: só guarda o input para o backward propagation.
    Cada ativação concreta (ReLU, Sigmoid, ...) só tem de implementar
    activation_function e derivative.

    Attributes
    ----------
    input: np.ndarray
        O último input recebido (fica guardado para o backward propagation).
    output: np.ndarray
        O último output calculado.
    """

    def forward_propagation(self, input: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Aplica a função de ativação ao input.

        Parameters
        ----------
        input: np.ndarray
            O input da camada.
        training: bool
            Não é usado (as ativações comportam-se igual em treino e em previsão).

        Returns
        -------
        output: np.ndarray
            O output da camada, com a mesma shape do input.
        """
        # guarda o input: no backward, a derivada é calculada no input, não no output
        self.input = input
        self.output = self.activation_function(input)
        return self.output

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Multiplica o erro que vem da camada seguinte pela derivada da ativação.

        Parameters
        ----------
        output_error: np.ndarray
            dE/dY, o erro em relação ao output desta camada.

        Returns
        -------
        input_error: np.ndarray
            dE/dX = f'(X) * dE/dY, o erro em relação ao input (para a camada anterior).
        """
        # regra da cadeia: dE/dX = dE/dY * dY/dX, e dY/dX = f'(X)
        # multiplicação elemento a elemento (*), não np.dot: cada output depende só
        # do seu próprio input (y_i = f(x_i)), por isso não há somas entre neurónios
        # não há pesos para atualizar: esta camada só passa o erro para trás
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Aplica a função de ativação ao input (usada no forward propagation).
        Cada ativação concreta implementa a sua.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Aplica a derivada da função de ativação ao input (usada no backward propagation).
        Cada ativação concreta implementa a sua.
        """
        raise NotImplementedError

    def output_shape(self) -> tuple:
        """
        Devolve a shape do output: igual à do input (a ativação não muda a shape).
        """
        return self.input_shape()

    def parameters(self) -> int:
        """
        Devolve 0: as camadas de ativação não têm parâmetros treináveis.
        """
        return 0

class ReLUActivation(ActivationLayer):
    """
    ReLU (Rectified Linear Unit): f(x) = max(0, x).

    Deixa passar os valores positivos sem mexer e põe os negativos a zero.
    É a ativação mais usada nas camadas escondidas: é barata de calcular e a
    derivada é 1 nos neurónios ativos, por isso não encolhe o erro no backward.
    """

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Aplica a ReLU, elemento a elemento: max(0, x).
        """
        # np.maximum compara cada elemento com 0 (np.max devolvia só o maior valor do array)
        return np.maximum(0, input)

    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Derivada da ReLU: 1 onde x > 0 e 0 onde x <= 0.
        """
        # x > 0: f(x) = x, declive 1; x < 0: f(x) = 0, declive 0
        # em x = 0 a derivada não existe (há um "bico"); por convenção usa-se 0
        return np.where(input > 0, 1, 0)


class SigmoidActivation(ActivationLayer):
    """
    Sigmoide: f(x) = 1 / (1 + e^(-x)).

    Comprime qualquer valor para o intervalo ]0, 1[. Usa-se sobretudo na camada
    de saída em classificação binária, porque o output lê-se como uma probabilidade.
    """

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Aplica a sigmoide, elemento a elemento.
        """
        # reutiliza a função da Aula 3 (regressão logística)
        return sigmoid_function(input)

    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Derivada da sigmoide: f'(x) = f(x) * (1 - f(x)).
        """
        # a derivada escreve-se à custa da própria função: basta calcular f(x) uma vez
        # o valor máximo é 0.25 (em x = 0): com muitas sigmoides seguidas, o erro
        # vai encolhendo de camada para camada (vanishing gradient)
        f = sigmoid_function(input)
        return f * (1 - f)
    