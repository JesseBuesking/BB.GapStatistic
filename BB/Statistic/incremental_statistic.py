import numpy as np


class IncrementalStatistic(object):
    number_of_values = 0
    sum_of_values = 0
    sum_of_values_squared = 0

    def reset(self):
        """
        Resets the object so that we can reuse it for another statistic.
        """
        self.number_of_values = 0
        self.sum_of_values = 0
        self.sum_of_values_squared = 0

    def add_value(self, value: float):
        self.number_of_values += 1
        self.sum_of_values += value
        self.sum_of_values_squared += (value * value)

    def get_mean(self):
        if self.number_of_values > 0:
            return np.float64(self.sum_of_values / self.number_of_values)

        return 0.0

    def get_population_variance(self):
        if self.number_of_values > 1:
            return np.float64(
                (self.sum_of_values_squared - self.sum_of_values *
                                              self.get_mean()) / (
                self.number_of_values - 1))

        return 0.0

    def get_population_standard_deviation(self):
        return np.sqrt(self.get_population_variance())

    def get_variance(self):
        if self.number_of_values > 1:
            return np.float64(
                (self.sum_of_values_squared - self.sum_of_values *
                                              self.get_mean()) / (
                self.number_of_values))

        return 0.0

    def get_standard_deviation(self):
        return np.sqrt(self.get_variance())
