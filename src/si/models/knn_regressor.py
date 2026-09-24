from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
    K-Nearest Neighbors regressor.
    The algorithm is similar to KNNClassifier, but suitable for regression problems:
    it estimates the average value of the k most similar examples instead of the
    most common class.
    """

    def __init__(self, k: int = 5, distance: Callable = euclidean_distance, **kwargs):
        """
        Parameters
        ----------
        k: int
            The number of k nearest examples to consider.
        distance: Callable
            A function that calculates the distance between a sample and the samples
            in the training dataset.
        """
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance

        # estimated parameters
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Stores the training dataset.

        Parameters
        ----------
        dataset: Dataset
            The training dataset.

        Returns
        -------
        self: KNNRegressor
        """
        self.dataset = dataset
        return self

    def _get_average_value(self, sample: np.ndarray) -> float:
        """
        Returns the average value of the k nearest neighbors of a single sample.

        Parameters
        ----------
        sample: numpy.ndarray (n_features,)
            The sample to predict the value for.

        Returns
        -------
        value: float
            The average value among the k nearest neighbors.
        """
        # 1. distance between the sample and every sample in the training dataset
        distances = self.distance(sample, self.dataset.X)

        # 2. indexes of the k nearest examples (shortest distance)
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # 3. values of those k nearest examples
        k_nearest_neighbors_values = self.dataset.y[k_nearest_neighbors]

        # 4. average of those values
        return np.mean(k_nearest_neighbors_values)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the value for each sample in the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The test dataset.

        Returns
        -------
        predictions: numpy.ndarray
            An array of predicted values for the testing dataset (y_pred).
        """
        # 5. apply the above to every sample in the testing dataset
        return np.apply_along_axis(self._get_average_value, axis=1, arr=dataset.X)

    def _score(self, dataset: Dataset) -> float:
        """
        Computes the rmse between the estimated values and the actual ones.

        Parameters
        ----------
        dataset: Dataset
            The test dataset.

        Returns
        -------
        error: float
            The rmse between predictions and actual values.
        """
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)
