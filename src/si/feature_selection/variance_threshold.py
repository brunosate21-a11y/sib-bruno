import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset


class VarianceThreshold(Transformer):
    """
    Variance Threshold feature selection.
    Selects all features whose variance is greater than a given threshold.
    """

    def __init__(self, threshold: float = 0.0, **kwargs):
        """
        Parameters
        ----------
        threshold: float
            The variance threshold (cut-off value).
        """
        super().__init__(**kwargs)
        self.threshold = threshold

        # estimated parameters
        self.variance = None

    def _fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Estimates the variance of each feature.

        Parameters
        ----------
        dataset: Dataset
            The dataset to estimate the feature variances from.

        Returns
        -------
        self: VarianceThreshold
        """
        self.variance = np.var(dataset.X, axis=0)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects all features with variance greater than the threshold.

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform.

        Returns
        -------
        dataset: Dataset
            The transformed dataset, with only the selected features.
        """
        features_mask = self.variance > self.threshold
        X = dataset.X[:, features_mask]
        features = np.array(dataset.features)[features_mask]
        return Dataset(X=X, y=dataset.y, features=list(features), label=dataset.label)
    