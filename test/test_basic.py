"""
A couple of simple unit tests.
"""

import unittest

import numpy as np
from sklearn.decomposition import PCA
from BB.GapStatistic import gap_statistic as gs
from BB.GapStatistic.gap_statistic import default_clustering, whiten
from BB.InputOutput import read
from BB.Plotting import plot
from BB.Plotting.plot import plot_gaps, plot_clusters


class Test(unittest.TestCase):
    """
    Unit tests for cluster_base class.
    """

#    def test_three_clusters_two_dimensions(self):
#        """
#        This does a full test, where we generate fake data for 3 clusters
#        in two dimensions.
#        """
#        np.random.seed(1)
#
#        cl1 = np.random.normal(0, 1, (20, 2))
#        cl2 = np.random.normal(3, 1, (20, 2))
#        cl3 = np.random.normal(0, 1, (20, 2))
#
#        cl2[:, 1] += 2
#        cl3[:, 0] += 5
#
#        data = np.concatenate((cl1, cl2, cl3))
#
#        gaps, confidence = gs.gap_statistic(data, 10, 10)
#
#        expected = 3
#        actual = gaps.argmax(axis=0)
#
#        plot_gaps(gaps, confidence, data.shape[1])
#        if 3 >= data.shape[1]:
#            k = gaps.argmax(axis=0)
#            inertia, point_map, centroids = default_clustering(data, k, 10,
#                                                               300)
#            plot_clusters(data, point_map, centroids)
#
#        self.assertEqual(expected, actual)
#
#    def test_four_clusters_three_dimensions(self):
#        """
#        This does a full test, where we generate fake data for 4 clusters
#        in three dimensions.
#        """
#        np.random.seed(1)
#
#        cl1 = np.random.normal(0, 1, (20, 3))
#        cl2 = np.random.normal(3, 1, (20, 3))
#        cl3 = np.random.normal(5, 1, (20, 3))
#        cl4 = np.random.normal(7, 1, (20, 3))
#
#        data = np.concatenate((cl1, cl2, cl3, cl4))
#
#        gaps, confidence = gs.gap_statistic(data, 10, 10)
#
#        expected = 4
#        actual = gaps.argmax(axis=0)
#
##        plot_gaps(gaps, confidence, data.shape[1])
##        if 3 >= data.shape[1]:
##            k = gaps.argmax(axis=0)
##            inertia, point_map, centroids = default_clustering(data, k, 10,
##                                                               300)
##            plot_clusters(data, point_map, centroids)
#
#        self.assertEqual(expected, actual)
#
#    def test_four_clusters_ten_dimensions(self):
#        """
#        This does a full test, where we generate fake data for 4 clusters
#        in three dimensions.
#        """
#        np.random.seed(1)
#
#        cl1 = np.random.normal(0, 1, (20, 10))
#        cl2 = np.random.normal(3, 1, (20, 10))
#        cl3 = np.random.normal(5, 1, (20, 10))
#        cl4 = np.random.normal(7, 1, (20, 10))
#
#        data = np.concatenate((cl1, cl2, cl3, cl4))
#
#        inertia, point_map, centroids = default_clustering(
#            data, 4, 10, 300)
#
#        plot.plot_clusters_pca_reduced(data, point_map, centroids)
#
#        gaps, confidence = gs.gap_statistic(data, 10, 10)
#
#        expected = 4
#        actual = gaps.argmax(axis=0)
#
#        self.assertEqual(expected, actual)
#
#    def test_one_cluster_five_dimensions(self):
#        """
#        This does a full test, where we generate fake data for 1 cluster
#        in five dimensions.
#        """
#        np.random.seed(1)
#
#        cl1 = np.random.normal(0, 1, (20, 5))
#
#        inertia, point_map, centroids = default_clustering(
#            cl1, 1, 10, 300)
#
#        plot.plot_clusters_pca_reduced(cl1, point_map, centroids)
#
#        gaps, confidence = gs.gap_statistic(cl1, 10, 10)
#
#        expected = 1
#        actual = gaps.argmax(axis=0)
#
#        plot_gaps(gaps, confidence, cl1.shape[1])
#
#        self.assertEqual(expected, actual)

#    def test_plot_multiple(self):
#        """
#        Tests plotting multiple plots on one graph.
#        """
#        np.random.seed(1)
#
#        cl1 = np.random.normal(0, 1, (40, 2))
#        cl2 = np.random.normal(3, 1, (40, 2))
#        cl3 = np.random.normal(5, 1, (40, 2))
#
#        cl2[:, 1] += 1
#        cl3[:, 0] += 5
#
#        data = np.concatenate((cl1, cl2, cl3))
#
#        uniform = gs.generate_bounding_box_uniform_points(data)
#        pca_uniform = gs.generate_principal_components_box_uniform_points(data)
#
#        plot.plot_data(data, uniform, pca_uniform)
#
#
#    def test_plot_multiple_3d_data(self):
#        """
#        Tests plotting multiple 3d plots on one graph.
#        """
#        np.random.seed(1)
#
#        cl1 = np.random.normal(0, 1, (40, 3))
#        cl2 = np.random.normal(3, 1, (40, 3))
#        cl3 = np.random.normal(5, 1, (40, 3))
#        cl4 = np.random.normal(7, 1, (40, 3))
#
#        data = np.concatenate((cl1, cl2, cl3, cl4))
#
#        uniform = gs.generate_bounding_box_uniform_points(data)
#        pca_uniform = gs.generate_principal_components_box_uniform_points(data)
#
#        plot.plot_data(data, uniform, pca_uniform)
#
#    def test_four_clusters_three_dimensions_with_pca(self):
#        """
#        This does a full test, where we generate fake data for 4 clusters
#        in three dimensions.
#        """
#        np.random.seed(1)
#
#        cl1 = np.random.normal(0, 0.5, (20, 3))
#        cl2 = np.random.normal(3, 0.5, (20, 3))
#        cl3 = np.random.normal(5, 0.5, (20, 3))
#        cl4 = np.random.normal(7, 0.5, (20, 3))
#
#        data = np.concatenate((cl1, cl2, cl3, cl4))
#        inertia, point_map, centroids = default_clustering(data, 4, 10, 300)
#
#        # Plot the clusters made on the original data, on the original data.
#        plot.plot_clusters(data, point_map, centroids)
#
#        pca = PCA(n_components=2)
#        reduced_data = pca.fit_transform(data)
#
#        # Plot the clusters made on the original data,
#        # on the pca reduced data.
#        plot.plot_clusters(reduced_data, point_map, pca.fit_transform(centroids))
#
#        inertia, point_map, centroids = default_clustering(
#            reduced_data, 4, 10, 300)
#
#        # Plot the clusters made on the pca reduced data,
#        # on the pca reduced data.
#        plot.plot_clusters(reduced_data, point_map, centroids)
#        original_centroids = pca.inverse_transform(centroids)
#
#        # Plot the clusters made on the pca reduced data,
#        # on the original data.
#        plot.plot_clusters(data, point_map, original_centroids)

    def test_real_data(self):
        data = read.read_to_numpy_array('data/real_data.csv', 0)
#        data, _, _ = whiten(data)
#        plot.plot_data(PCA(n_components=2).fit_transform(data))

        gaps, confidence = gs.gap_statistic(data, 100, 10)

        plot_gaps(gaps, confidence, data.shape[1])

#        data_subset = data[:1000,:]
#        print(data_subset.shape[0])
#        data = data_subset
#
#        inertia, point_map, centroids = default_clustering(data, 60, 10, 300)

#        plot.plot_clusters_pca_reduced(data, point_map, centroids)
