import datetime
import unittest
import numpy as np
from BB.Statistic.incremental_statistic import IncrementalStatistic


class Test(unittest.TestCase):
    def test_basic_incremental_statistic(self):
        total = 1000
        inc_stat = IncrementalStatistic()
        storage = np.zeros(shape=(total, 1))

        for index in range(total):
            random_value = np.random.uniform(0.0, 1.0)
            storage[index] = random_value
            inc_stat.add_value(random_value)

        storage_mean = np.mean(storage)
        storage_stddev = storage.std()

        incremental_mean = inc_stat.get_mean()
        incremental_stddev = inc_stat.get_standard_deviation()

        np.testing.assert_almost_equal(storage_mean, incremental_mean)
        np.testing.assert_almost_equal(storage_stddev, incremental_stddev)

    def test_perf(self):
        inc_time = datetime.timedelta(0)
        storage_time = datetime.timedelta(0)
        iterations = 100
        total = 10000

        for iteration in range(iterations):
            inc_stat = IncrementalStatistic()
            start = datetime.datetime.utcnow()
            for index in range(total):
                random_value = np.random.uniform(0.0, 1.0)
                inc_stat.add_value(random_value)

            incremental_mean = inc_stat.get_mean()
            incremental_stddev = inc_stat.get_standard_deviation()
            stop = datetime.datetime.utcnow()
            inc_time += stop - start

            start = datetime.datetime.utcnow()
            storage = np.zeros(shape=(total, 1))
            for index in range(total):
                random_value = np.random.uniform(0.0, 1.0)
                storage[index] = random_value

            storage_mean = np.mean(storage)
            storage_stddev = storage.std()
            stop = datetime.datetime.utcnow()
            storage_time += stop - start

        storage_time_faster = storage_time - inc_time < datetime.timedelta(0)
        self.assertTrue(storage_time_faster)
