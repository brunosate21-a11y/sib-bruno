from typing import Callable, Dict

import numpy as np

from si.data.dataset import Dataset
from si.model_selection.grid_search import _make_folds, _cross_validate


def log_uniform(low: float, high: float) -> Callable:
    """
    Distribuição log-uniforme entre low e high, para valores que variam em ordens de grandeza
    (ex.: learning_rate entre 1e-4 e 1e-1).

    Sorteia o expoente de forma uniforme: 10 ** uniform(log10(low), log10(high)).
    Assim, cada ordem de grandeza (1e-4 a 1e-3, 1e-3 a 1e-2, ...) tem a mesma probabilidade.
    Com uma uniforme normal, 90% dos valores caíam entre 0.01 e 0.1.

    Parameters
    ----------
    low: float
        O valor mínimo.
    high: float
        O valor máximo.

    Returns
    -------
    sampler: callable
        Uma função sampler(rng) que devolve um valor sorteado com o gerador rng.
    """
    def sampler(rng: np.random.Generator) -> float:
        # float(): devolve um float do Python, e não um np.float64 (fica mais limpo nos prints)
        return float(10 ** rng.uniform(np.log10(low), np.log10(high)))

    return sampler


def randomized_search_cv(model, dataset: Dataset, hyperparameter_distributions: Dict[str, object],
                         scoring: Callable = None, cv: int = 5, n_iter: int = 10, seed: int = None) -> dict:
    """
    Randomized search com k-fold cross validation: experimenta n_iter configurações sorteadas
    e escolhe a que tem a maior média dos scores de validação.

    Com o mesmo número de treinos, experimenta mais valores diferentes de cada hiperparâmetro
    do que o grid search (Bergstra & Bengio, 2012).

    Parameters
    ----------
    model: Model or callable
        O modelo a afinar, ou uma função construtora que devolve um modelo novo.
    dataset: Dataset
        O dataset de treino (nunca o de teste).
    hyperparameter_distributions: Dict[str, object]
        Para cada hiperparâmetro, uma lista de valores (sorteia-se um, todos com a mesma
        probabilidade) ou uma função sampler(rng) (ex.: log_uniform(1e-4, 1e-1)).
    scoring: callable
        A função de avaliação, scoring(y_true, y_pred). Se for None, usa o model.score.
    cv: int
        O número de folds.
    n_iter: int
        O número de configurações aleatórias a avaliar.
    seed: int
        A seed usada para criar os folds e para sortear as configurações.

    Returns
    -------
    results: dict
        O mesmo dicionário do grid_search_cv: hyperparameters, scores,
        best_hyperparameters e best_score.
    """
    # os folds criam-se uma só vez, como no grid search: todas as configurações usam os mesmos
    folds = _make_folds(dataset, cv, seed)

    # gerador próprio para os sorteios: a função construtora pode fazer np.random.seed(0)
    # (para os pesos iniciais) e isso mexe no gerador global, mas não neste
    rng = np.random.default_rng(seed)

    results = {'hyperparameters': [], 'scores': []}

    for _ in range(n_iter):
        # sortear um valor para cada hiperparâmetro
        configuration = {}
        for name, distribution in hyperparameter_distributions.items():
            if callable(distribution):
                # função (ex.: log_uniform): chama-se com o gerador
                configuration[name] = distribution(rng)
            else:
                # lista: escolhe-se um índice ao acaso (todos com a mesma probabilidade)
                # (indexar a lista devolve o valor original, ex.: um int, e não um np.int64)
                configuration[name] = distribution[rng.integers(len(distribution))]

        # treinar e validar em cada fold; guardar a configuração e a média dos scores
        score = _cross_validate(model, configuration, folds, scoring)
        results['hyperparameters'].append(configuration)
        results['scores'].append(score)

    # a melhor configuração é a de maior score
    best = int(np.argmax(results['scores']))
    results['best_hyperparameters'] = results['hyperparameters'][best]
    results['best_score'] = results['scores'][best]

    return results