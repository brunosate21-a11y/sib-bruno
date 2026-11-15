from abc import ABCMeta, abstractmethod

import numpy as np


class Optimizer(metaclass=ABCMeta):
    """
    Classe base (abstrata) para os optimizers.

    Um optimizer decide como os parâmetros (pesos ou biases) de uma camada mudam
    a partir do gradiente da loss. Cada DenseLayer recebe duas cópias do optimizer
    no initialize (w_opt para os pesos, b_opt para os biases) e chama o update
    no backward propagation.

    Parameters
    ----------
    learning_rate: float
        A taxa de aprendizagem: o tamanho do passo em cada atualização.
    """

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Atualiza os parâmetros (pesos ou biases) de uma camada.

        Parameters
        ----------
        w: np.ndarray
            Os parâmetros atuais.
        grad_loss_w: np.ndarray
            O gradiente da loss em relação a esses parâmetros (mesma shape de w).

        Returns
        -------
        w: np.ndarray
            Os parâmetros atualizados.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent com momentum.

    Em vez de usar só o gradiente atual, usa uma média dos gradientes anteriores
    (o retained_gradient). Se os gradientes apontam sempre para o mesmo lado, a média
    mantém essa direção; se trocam de sinal a cada passo (zigue-zague), anulam-se em
    parte e o treino fica mais estável.

    Parameters
    ----------
    learning_rate: float
        A taxa de aprendizagem (α).
    momentum: float
        O momentum (β), entre 0 e 1. Com 0 é o gradient descent normal.

    Attributes
    ----------
    retained_gradient: np.ndarray
        A média dos gradientes (V), com a mesma shape dos parâmetros.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        # o __init__ da Optimizer guarda o learning_rate
        super().__init__(learning_rate)
        self.momentum = momentum

        # atributo estimado: a shape só se sabe quando chegar o primeiro gradiente
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Atualiza o retained_gradient e devolve os parâmetros atualizados:
            V_t = β * V_(t-1) + (1 - β) * dL/dW
            W_t = W_(t-1) - α * V_t

        Parameters
        ----------
        w: np.ndarray
            Os parâmetros atuais (pesos ou biases).
        grad_loss_w: np.ndarray
            O gradiente da loss em relação a w.

        Returns
        -------
        w: np.ndarray
            Os parâmetros atualizados.
        """
        # na primeira chamada ainda não há gradientes anteriores: começa a zeros,
        # com a shape dos parâmetros (pesos (n_features, n_units) ou biases (1, n_units))
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))

        # media móvel: uma parte β do que vinha de trás + uma parte (1 - β) do gradiente atual
        # com β = 0 fica só o gradiente atual (gradient descent sem momentum)
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w

        # passo no sentido contrário: o gradiente aponta para onde a loss sobe, por isso subtrai-se
        return w - self.learning_rate * self.retained_gradient


class Adam(Optimizer):
    """
    Adam: junta o momentum do SGD com um passo adaptado a cada parâmetro (como o RMSprop).

    Guarda duas médias móveis do gradiente de cada parâmetro:
    - m (1.º momento): média dos gradientes; dá a direção, como o retained_gradient do SGD;
    - v (2.º momento): média dos gradientes ao quadrado; mede o tamanho típico do gradiente.
    O passo é proporcional a m / sqrt(v): a direção vem de m, mas o tamanho é normalizado,
    por isso cada parâmetro anda ~learning_rate por passo, quer o seu gradiente seja grande ou pequeno.

    Parameters
    ----------
    learning_rate: float
        A taxa de aprendizagem (α).
    beta_1: float
        A taxa de decaimento da média dos gradientes (m).
    beta_2: float
        A taxa de decaimento da média dos gradientes ao quadrado (v).
    epsilon: float
        Uma constante pequena para nunca dividir por 0.

    Attributes
    ----------
    m: np.ndarray
        A média móvel dos gradientes.
    v: np.ndarray
        A média móvel dos gradientes ao quadrado.
    t: int
        O número de atualizações já feitas (cada mini-batch é uma atualização).
    """

    def __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999,
                 epsilon: float = 1e-8):
        # o __init__ da Optimizer guarda o learning_rate
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # atributos estimados: m e v só ganham shape no primeiro update; t começa em 0
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Atualiza t, m e v e devolve os parâmetros atualizados:
            m_t = β1 * m_(t-1) + (1 - β1) * dL/dW
            v_t = β2 * v_(t-1) + (1 - β2) * (dL/dW)^2
            m_hat = m_t / (1 - β1^t)
            v_hat = v_t / (1 - β2^t)
            W_t = W_(t-1) - α * m_hat / (sqrt(v_hat) + ε)

        Parameters
        ----------
        w: np.ndarray
            Os parâmetros atuais (pesos ou biases).
        grad_loss_w: np.ndarray
            O gradiente da loss em relação a w.

        Returns
        -------
        w: np.ndarray
            Os parâmetros atualizados.
        """
        # na primeira chamada, m e v começam a zeros, com a shape dos parâmetros
        if self.m is None:
            self.m = np.zeros(np.shape(w))
            self.v = np.zeros(np.shape(w))

        # mais uma atualização (na NeuralNetwork, t conta mini-batches, e não épocas)
        self.t += 1

        # média dos gradientes: a direção (igual ao retained_gradient do SGD com β = β1)
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w

        # média dos gradientes ao quadrado: o tamanho típico do gradiente (o quadrado tira o sinal)
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * grad_loss_w ** 2

        # correção do viés: m e v começam a 0, por isso nos primeiros passos ficam demasiado pequenos
        # (no 1.º passo, m = 0.1 * gradiente); dividir por (1 - β^t) corrige isso (m_hat = gradiente);
        # à medida que t cresce, β^t vai para 0 e a correção desaparece
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)

        # passo: direção de m_hat, dividida pelo tamanho típico do gradiente (sqrt(v_hat))
        # epsilon evita dividir por 0 se o gradiente for 0
        return w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)