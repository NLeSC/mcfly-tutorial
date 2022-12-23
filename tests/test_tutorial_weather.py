import shutil
import unittest
from pathlib import Path

from utils.tutorial_weather import load_data


class TutorialWeatherSuite(unittest.TestCase):
    """ Weather data set test cases."""
    temp_test_dir = 'temp_weather_test'

    def test_data_downloading_has_correct_shape(self):
        n_features = 89
        n_train_instances = 767
        n_test_instances = 329

        X_train, X_test, y_train, y_test = load_data(self.temp_test_dir)

        assert X_train.shape == (n_train_instances, n_features)
        assert X_test.shape == (n_test_instances, n_features)
        assert y_train.shape == (n_train_instances,)
        assert y_test.shape == (n_test_instances,)

    def setUp(self) -> None:
        Path(self.temp_test_dir).mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(Path(self.temp_test_dir))


if __name__ == '__main__':
    unittest.main()
