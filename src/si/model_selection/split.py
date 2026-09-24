from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Splits a Dataset into a training and a testing Dataset.

    Parameters
    ----------
    dataset: Dataset
        The dataset object to split into train and test.
    test_size: float
        The size of the testing Dataset (e.g., 0.2 for 20%).
    random_state: int
        Seed for generating permutations.

    Returns
    -------
    train, test: Tuple[Dataset, Dataset]
        The train and test Dataset objects.
    """
    np.random.seed(random_state)

    n_samples = dataset.shape()[0]

    # generate permutations using np.random.permutation
    permutations = np.random.permutation(n_samples)

    # infer the number of samples in the train and test datasets
    n_test = int(n_samples * test_size)

    # select train and test datasets using the permutations
    test_idxs = permutations[:n_test]
    train_idxs = permutations[n_test:]

    train = Dataset(X=dataset.X[train_idxs], y=dataset.y[train_idxs] if dataset.y is not None else None,
                     features=dataset.features, label=dataset.label)
    test = Dataset(X=dataset.X[test_idxs], y=dataset.y[test_idxs] if dataset.y is not None else None,
                    features=dataset.features, label=dataset.label)

    return train, test


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Splits a Dataset into a training and a testing Dataset, preserving the class proportions
    of the original dataset in both (stratified sampling).

    Parameters
    ----------
    dataset: Dataset
        The dataset object to split into train and test.
    test_size: float
        The size of the testing Dataset (e.g., 0.2 for 20%).
    random_state: int
        Seed for generating permutations.

    Returns
    -------
    train, test: Tuple[Dataset, Dataset]
        The stratified train and test Dataset objects.
    """
    np.random.seed(random_state)

    # get unique class labels and their counts
    labels, counts = np.unique(dataset.y, return_counts=True)

    # initialize empty lists for train and test indices
    train_idxs = []
    test_idxs = []

    # loop through unique labels
    for label, count in zip(labels, counts):
        # calculate the number of test samples for the current class
        n_test = int(count * test_size)
        label_idxs = np.where(dataset.y == label)[0]

        # shuffle and select indices for the current class and add them to the test indices
        permuted_label_idxs = np.random.permutation(label_idxs)
        test_idxs.extend(permuted_label_idxs[:n_test])

        # add the remaining indices to the train indices
        train_idxs.extend(permuted_label_idxs[n_test:])

    # after the loop, create training and testing datasets
    train = Dataset(X=dataset.X[train_idxs], y=dataset.y[train_idxs] if dataset.y is not None else None,
                     features=dataset.features, label=dataset.label)
    test = Dataset(X=dataset.X[test_idxs], y=dataset.y[test_idxs] if dataset.y is not None else None,
                    features=dataset.features, label=dataset.label)

    # return the training and testing datasets
    return train, test
