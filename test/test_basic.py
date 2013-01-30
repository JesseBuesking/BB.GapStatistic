"""
A couple of simple unit tests.
"""

import unittest

import numpy as np
from BB.GapStatistic import gap_statistic as gs


class Test(unittest.TestCase):
    """
    Unit tests for cluster_base class.
    """

    def test_three_clusters_two_dimensions(self):
        """
        This does a full test, where we generate fake data for 3 clusters
        in two dimensions.
        """
        np.random.seed(1)

        cl1 = np.random.normal(0, 1, (20, 2))
        cl2 = np.random.normal(3, 1, (20, 2))
        cl3 = np.random.normal(0, 1, (20, 2))

        cl2[:, 1] += 2
        cl3[:, 0] += 5

        data = np.concatenate((cl1, cl2, cl3))

        gaps, confidence = gs.gap_statistic(data, 10)

        expected = 3
        actual = gaps.argmax(axis=0)

        self.assertEqual(expected, actual)
