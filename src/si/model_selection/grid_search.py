import copy
import itertools
from typing import Callable, Dict, List

import numpy as np

from si.data.dataset import Dataset


def _build_model(model, hyperparameters: dict):
    """
    Cria um modelo novo, por treinar, com os hiperparâmetros dados.

    Parameters
    ----------
    model: Model or callable
        Um modelo (ex.: LogisticRegression()) ou uma função construtora que devolve um modelo
        novo (ex.: build_network).
    hyperparameters: dict
        Os hiperparâmetros desta configuração, {nome: valor}.

    Returns
    -------
    new_model: Model
        Um modelo novo, por treinar.
    """
    # função construtora: chama-se com os hiperparâmetros e devolve uma rede nova
    # (é o caso da NeuralNetwork: o optimizer é criado no __init__ e copiado para as camadas
    # no add, por isso mudar o learning_rate depois de criada a rede não mudava nada)
    if callable(model):
        return model(**hyperparameters)

    # modelo simples: copia-se o modelo por treinar e mudam-se os atributos
    new_model = copy.deepcopy(model)
    for name, value in hyperparameters.items():
        # um nome errado (ex.: 'l2penalty') criava um atributo novo que o modelo nunca usa,
        # e todas as configurações davam o mesmo score sem nenhum erro
        if not hasattr(new_model, name):
            raise AttributeError(f"O modelo {type(new_model).__name__} não tem o hiperparâmetro '{name}'.")
        setattr(new_model, name, value)
    return new_model


def _make_folds(dataset: Dataset, cv: int, seed: int = None) -> List[tuple]:
    """
    Divide o dataset em cv folds e devolve os pares (treino, validação).

    Os folds são os mesmos da k_fold_cross_validation com a mesma seed.

    Returns
    -------
    folds: List[tuple]
        Lista de cv pares (train, validation), cada um com dois Datasets.
    """
    if seed is not None:
        np.random.seed(seed)

    # baralhar os índices e dividir em cv partes quase iguais (como na k_fold_cross_validation)
    indices = np.random.permutation(dataset.X.shape[0])
    parts = np.array_split(indices, cv)

    folds = []
    for i in range(cv):
        # parte i = validação; as outras partes juntas = treino
        val_idxs = parts[i]
        train_idxs = np.concatenate([parts[j] for j in range(cv) if j != i])
        train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs],
                        features=dataset.features, label=dataset.label)
        validation = Dataset(dataset.X[val_idxs], dataset.y[val_idxs],
                             features=dataset.features, label=dataset.label)
        folds.append((train, validation))
    return folds


def _cross_validate(model, hyperparameters: dict, folds: List[tuple], scoring: Callable = None) -> float:
    """
    Avalia uma configuração: treina um modelo novo em cada fold e devolve a média dos scores.

    Returns
    -------
    score: float
        A média dos cv scores de validação.
    """
    scores = []
    for train, validation in folds:
        # um modelo novo em cada fold: nunca se reaproveitam pesos de outro fold ou de outra configuração
        fold_model = _build_model(model, hyperparameters)
        fold_model.fit(train)

        if scoring is None:
            scores.append(fold_model.score(validation))
        else:
            scores.append(scoring(validation.y, fold_model.predict(validation)))

    return float(np.mean(scores))


def grid_search_cv(model, dataset: Dataset, hyperparameter_grid: Dict[str, list], scoring: Callable = None,
                   cv: int = 5, seed: int = None) -> dict:
    """
    Grid search com k-fold cross validation: experimenta todas as combinações de hiperparâmetros
    e escolhe a que tem a maior média dos scores de validação.

    Parameters
    ----------
    model: Model or callable
        O modelo a afinar, ou uma função construtora que devolve um modelo novo.
    dataset: Dataset
        O dataset de treino (nunca o de teste: o teste só se usa no fim).
    hyperparameter_grid: Dict[str, list]
        Os valores a experimentar para cada hiperparâmetro, {nome: lista de valores}.
    scoring: callable
        A função de avaliação, scoring(y_true, y_pred). Se for None, usa o model.score.
    cv: int
        O número de folds.
    seed: int
        A seed usada para criar os folds.

    Returns
    -------
    results: dict
        hyperparameters: lista com todas as configurações testadas
        scores: a média dos scores de validação de cada configuração
        best_hyperparameters: a configuração com a maior média
        best_score: a média dessa configuração
    """
    # criar os folds uma só vez: todas as configurações são avaliadas nos mesmos folds,
    # por isso as diferenças nos scores vêm dos hiperparâmetros, e não da divisão dos dados
    folds = _make_folds(dataset, cv, seed)

    results = {'hyperparameters': [], 'scores': []}

    # todas as combinações: itertools.product faz o produto cartesiano das listas de valores
    #    ex.: {'a': [1, 2], 'b': [3, 4]} --> (1, 3), (1, 4), (2, 3), (2, 4)
    names = list(hyperparameter_grid.keys())
    for values in itertools.product(*hyperparameter_grid.values()):
        # juntar os nomes aos valores: (0.01, 100) --> {'learning_rate': 0.01, 'epochs': 100}
        configuration = dict(zip(names, values))

        # treinar e validar em cada fold; guardar a configuração e a média dos scores
        score = _cross_validate(model, configuration, folds, scoring)
        results['hyperparameters'].append(configuration)
        results['scores'].append(score)

    # a melhor configuração é a de maior score (accuracy: quanto maior, melhor)
    # em caso de empate, o argmax fica com a primeira
    best = int(np.argmax(results['scores']))
    results['best_hyperparameters'] = results['hyperparameters'][best]
    results['best_score'] = results['scores'][best]

    return results