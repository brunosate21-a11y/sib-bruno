from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance


class KNNClassifier(Model):
    """
    K-Nearest Neighbors classifier.
    Estimates the class for a sample based on the k most similar examples (shortest distance)
    in the training dataset.
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

    def _fit(self, dataset: Dataset) -> 'KNNClassifier':
        """
        Stores the training dataset.

        Parameters
        ----------
        dataset: Dataset
            The training dataset.

        Returns
        -------
        self: KNNClassifier
        """
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray):
        """
        Returns the most common class among the k nearest neighbors of a single sample.

        Parameters
        ----------
        sample: numpy.ndarray (n_features,)
            The sample to classify.

        Returns
        -------
        label: the most common class among the k nearest neighbors.
        """
        # 1. distance between the sample and every sample in the training dataset
        distances = self.distance(sample, self.dataset.X)

        # 2. indexes of the k nearest examples (shortest distance)
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # 3. classes of those k nearest examples
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        # 4. most common class (highest frequency) among the k examples
        labels, counts = np.unique(k_nearest_neighbors_labels, return_counts=True)
        return labels[np.argmax(counts)]

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the class for each sample in the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The test dataset.

        Returns
        -------
        predictions: numpy.ndarray
            An array of predicted classes for the testing dataset (y_pred).
        """
        # 5. apply the above to every sample in the testing dataset
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)

    def _score(self, dataset: Dataset) -> float:
        """
        Computes the accuracy between the estimated classes and the actual ones.

        Parameters
        ----------
        dataset: Dataset
            The test dataset.

        Returns
        -------
        error: float
            The accuracy between predictions and actual values.
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)
