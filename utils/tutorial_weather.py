import typing
import urllib
from pathlib import Path

from urllib.request import urlretrieve
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str = '.'):
    """
    Load weather dataset (10.5281/zenodo.4770936.). If it's not on the path specified, it will be downloaded.
    Parameters
    ----------
    path : str
        The local path to the data set folder.

    Returns
    -------
        X_train
        X_test
        y_train
        y_test
    """
    data_path = download_preprocessed_data(path)
    data = pd.read_csv(data_path)
    nr_rows = 365 * 3
    X_data = data.loc[:nr_rows].drop(columns=['DATE', 'MONTH'])

    days_ahead = 1
    y_data = data.loc[days_ahead:(nr_rows + days_ahead)]["MAASTRICHT_sunshine"]
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=0)

    return X_train, X_test, y_train, y_test


def download_preprocessed_data(directory_to_extract_to: typing.Union[str, Path]):
    data_path = Path(directory_to_extract_to) / 'weather'
    data_path.mkdir(exist_ok=True)
    data_set_light_path = data_path / 'weather_prediction_dataset_light.csv'
    if not data_set_light_path.exists():
        _, _ = urllib.request.urlretrieve(
            'https://zenodo.org/record/7053722/files/weather_prediction_dataset_light.csv',
            filename=data_set_light_path)
    return data_set_light_path
