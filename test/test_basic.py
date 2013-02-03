"""
A couple of simple unit tests.
"""

import unittest

import numpy as np
from BB.GapStatistic import gap_statistic as gs
from BB.GapStatistic.gap_statistic import default_clustering
from BB.Plotting import plot
from BB.Plotting.plot import plot_gaps, plot_clusters


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

        gaps, confidence = gs.gap_statistic(data, 10, 10)

        expected = 3
        actual = gaps.argmax(axis=0)

#        plot_gaps(gaps, confidence, data.shape[1])
#        if 3 >= data.shape[1]:S
#            k = gaps.argmax(axis=0)
#            inertia, point_map, centroids = default_clustering(data, k, 10,
#                                                               300)
#            plot_clusters(data, point_map, centroids)

        self.assertEqual(expected, actual)

    def test_four_clusters_three_dimensions(self):
        """
        This does a full test, where we generate fake data for 4 clusters
        in three dimensions.
        """
        np.random.seed(1)

        cl1 = np.random.normal(0, 1, (20, 3))
        cl2 = np.random.normal(3, 1, (20, 3))
        cl3 = np.random.normal(5, 1, (20, 3))
        cl4 = np.random.normal(7, 1, (20, 3))

        data = np.concatenate((cl1, cl2, cl3, cl4))

        gaps, confidence = gs.gap_statistic(data, 10, 10)

        expected = 4
        actual = gaps.argmax(axis=0)

#        plot_gaps(gaps, confidence, data.shape[1])
#        if 3 >= data.shape[1]:
#            k = gaps.argmax(axis=0)
#            inertia, point_map, centroids = default_clustering(data, k, 10,
#                                                               300)
#            plot_clusters(data, point_map, centroids)

        self.assertEqual(expected, actual)

    def test_four_clusters_ten_dimensions(self):
        """
        This does a full test, where we generate fake data for 4 clusters
        in three dimensions.
        """
        np.random.seed(1)

        cl1 = np.random.normal(0, 1, (20, 10))
        cl2 = np.random.normal(3, 1, (20, 10))
        cl3 = np.random.normal(5, 1, (20, 10))
        cl4 = np.random.normal(7, 1, (20, 10))

        data = np.concatenate((cl1, cl2, cl3, cl4))

        gaps, confidence = gs.gap_statistic(data, 10, 10)

        expected = 4
        actual = gaps.argmax(axis=0)

#        plot_gaps(gaps, confidence, data.shape[1])

        self.assertEqual(expected, actual)

    def test_one_cluster_five_dimensions(self):
        """
        This does a full test, where we generate fake data for 1 cluster
        in five dimensions.
        """
        np.random.seed(1)

        cl1 = np.random.normal(0, 1, (20, 5))

        gaps, confidence = gs.gap_statistic(cl1, 10, 10)

        expected = 5
        actual = gaps.argmax(axis=0)

#        plot_gaps(gaps, confidence, cl1.shape[1])

        self.assertEqual(expected, actual)

    def test_plot_multiple(self):

        np.random.seed(1)

        cl1 = np.random.normal(0, 1, (100, 2))
        cl2 = np.random.normal(3, 1, (100, 2))
        cl3 = np.random.normal(5, 1, (100, 2))

        cl2[:, 1] += 1
        cl3[:, 0] += 5

        data = np.concatenate((cl1, cl2, cl3))

        uniform = gs.generate_bounding_box_uniform_points(data)
        pca_uniform = gs.generate_principal_components_box_uniform_points(data)

        plot.plot_data(data, uniform, pca_uniform)
