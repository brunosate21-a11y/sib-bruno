import pandas as pd

from si.data.dataset import Dataset


def read_csv(filename: str, sep: str = ',', features: bool = False, label: bool = False) -> Dataset:
    """
    Reads a csv file and returns a Dataset object.

    Parameters
    ----------
    filename: str
        Path/name of the file
    sep: str
        The value separator
    features: bool
        Whether the file has feature columns (a header row is always expected)
    label: bool
        Whether the file has y (assumed to be the last column)

    Returns
    -------
    Dataset
    """
    data = pd.read_csv(filename, sep=sep)

    if features and label:
        features_names = data.columns[:-1].to_list()
        label_name = data.columns[-1]
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()

    elif features and not label:
        features_names = data.columns.to_list()
        X = data.to_numpy()
        y = None
        label_name = None

    elif not features and label:
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        features_names = None
        label_name = "y"

    else:
        X = data.to_numpy()
        y = None
        features_names = None
        label_name = None

    return Dataset(X, y, features=features_names, label=label_name)


def write_csv(filename: str, dataset: Dataset, sep: str = ',', features: bool = False, label: bool = False) -> None:
    """
    Writes a Dataset object to a csv file.

    Parameters
    ----------
    filename: str
        Path/name of the file
    dataset: Dataset
        The dataset object to write
    sep: str
        The value separator
    features: bool
        Whether to use the dataset feature names as column names
    label: bool
        Whether to write y (as the last column)

    Returns
    -------
    None
    """
    df = pd.DataFrame(dataset.X, columns=dataset.features if features else None)

    if label and dataset.y is not None:
        df[dataset.label] = dataset.y

    df.to_csv(filename, sep=sep, index=False)

