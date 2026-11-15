import copy
from typing import Callable, List

import numpy as np

from si.data.dataset import Dataset


def k_fold_cross_validation(model, dataset: Dataset, scoring: Callable = None, cv: int = 3,
                            seed: int = None) -> List[float]:
    """
    Avalia um modelo com k-fold cross validation.

    Divide as amostras em cv partes (folds). Em cada iteração, uma parte serve de teste
    e as outras cv - 1 de treino. No fim, cada amostra foi usada para teste exatamente uma vez.

    Parameters
    ----------
    model: Model
        O modelo a avaliar (ainda por treinar).
    dataset: Dataset
        O dataset a usar.
    scoring: callable
        A função de avaliação, scoring(y_true, y_pred). Se for None, usa o model.score.
    cv: int
        O número de folds.
    seed: int
        A seed para baralhar as amostras (resultados reprodutíveis).

    Returns
    -------
    scores: List[float]
        O score de cada fold.
    """
    if seed is not None:
        np.random.seed(seed)

    # baralhar os índices: se os dados viessem ordenados (ex.: por classe),
    # um fold podia ficar só com amostras de uma classe
    n_samples = dataset.X.shape[0]
    indices = np.random.permutation(n_samples)

    # np.array_split divide em cv partes quase iguais (se n não for múltiplo de cv,
    # as primeiras partes ficam com mais uma amostra), por isso nenhuma amostra fica de fora
    folds = np.array_split(indices, cv)

    scores = []
    for i in range(cv):
        # fold i = teste; todos os outros folds juntos = treino
        test_idxs = folds[i]
        train_idxs = np.concatenate([folds[j] for j in range(cv) if j != i])

        train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs],
                        features=dataset.features, label=dataset.label)
        test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs],
                       features=dataset.features, label=dataset.label)

        # cópia do modelo por treinar: cada fold começa do zero, com os mesmos pesos iniciais
        # sem a cópia, a rede neuronal continuava a treinar com os pesos do fold anterior,
        # que já tinham visto as amostras de teste deste fold (fuga de informação)
        fold_model = copy.deepcopy(model)
        fold_model.fit(train)

        if scoring is None:
            scores.append(fold_model.score(test))
        else:
            scores.append(scoring(test.y, fold_model.predict(test)))

    return scores