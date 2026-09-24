from scipy import stats

from si.data.dataset import Dataset


def f_classification(dataset: Dataset) -> tuple:
    """
    Scoring function for classification problems. Computes the one-way ANOVA F-value
    for each feature of the dataset.

    Parameters
    ----------
    dataset: Dataset
        The dataset object.

    Returns
    -------
    F: tuple
        The F value for each feature.
    p: tuple
        The p-value for each feature.
    """
    classes = dataset.get_classes()

    # group the samples by class
    groups = [dataset.X[dataset.y == c] for c in classes]

    F, p = stats.f_oneway(*groups)
    return F, p   