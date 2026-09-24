import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectKBest(Transformer):
    """
    Select features according to the k highest scores.
    """

    def __init__(self, score_func=f_classification, k: int = 10, **kwargs):
        """
        Parameters
        ----------
        score_func: callable
            Variance analysis function (e.g. f_classification).
        k: int
            Number of features to select.
        """
        super().__init__(**kwargs)
        self.score_func = score_func
        self.k = k

        # estimated parameters
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        Estimates the F and p values for each feature using the score_func.

        Parameters
        ----------
        dataset: Dataset
            The dataset to estimate the F and p values from.

        Returns
        -------
        self: SelectKBest
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects the top k features with the highest F value.

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform.

        Returns
        -------
        dataset: Dataset
            The transformed dataset, with only the top k features.
        """
        idxs = np.sort(np.argsort(self.F)[-self.k:])
        X = dataset.X[:, idxs]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=X, y=dataset.y, features=list(features), label=dataset.label)