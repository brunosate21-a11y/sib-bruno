from abc import ABCMeta, ABC, abstractmethod

from si.base.estimator import Estimator


class Model(Estimator, ABC):
    """
    Abstract base class for models.
    A model is an object that can predict the target values of a Dataset object.
    """

    def __init__(self, **kwargs):
        """
        Initialize the model.
        """
        super().__init__(**kwargs)

    def predict(self, dataset):
        """
        Predict the target values of the dataset.
        The model needs to be fitted before calling this method.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the target values of.

        Returns
        -------
        predictions: np.ndarray
            The predicted target values.
        """
        if not self.is_fitted:
            raise ValueError('Model needs to be fitted before calling predict()')
        return self._predict(dataset)

    @abstractmethod
    def _predict(self, dataset):
        """
        Predict the target values of the dataset.
        Abstract method that needs to be implemented by all subclasses.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the target values of.

        Returns
        -------
        predictions: np.ndarray
            The predicted target values.
        """

    def score(self, dataset):
        """
        Compute the error/performance metric between the model's predictions and the true values.
        The model needs to be fitted before calling this method.

        Parameters
        ----------
        dataset: Dataset
            The dataset to score the model on.

        Returns
        -------
        score: float
            The error/performance metric of the model.
        """
        if not self.is_fitted:
            raise ValueError('Model needs to be fitted before calling score()')
        return self._score(dataset)

    @abstractmethod
    def _score(self, dataset):
        """
        Compute the error/performance metric between the model's predictions and the true values.
        Abstract method that needs to be implemented by all subclasses.

        Parameters
        ----------
        dataset: Dataset
            The dataset to score the model on.

        Returns
        -------
        score: float
            The error/performance metric of the model.
        """

    def fit_predict(self, dataset):
        """
        Fit the model to the dataset and predict the target values.
        Equivalent to calling fit(dataset) and then predict(dataset).

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit and predict the target values of.

        Returns
        -------
        predictions: np.ndarray
            The predicted target values.
        """
        self.fit(dataset)
        return self.predict(dataset)
