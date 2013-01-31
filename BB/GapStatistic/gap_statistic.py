import logging
from sklearn.cluster import KMeans
import numpy as np


def default_clustering(data, k, individual_runs=10, iterations=300):
    estimator = KMeans(init='k-means++',
                       precompute_distances=True,
                       n_clusters=k,
                       n_init=individual_runs,
                       max_iter=iterations)
    estimator.fit(data)

    return estimator.inertia_, estimator.labels_, estimator.cluster_centers_


def generate_bounding_box_uniform_points(data):
    number_of_data_points = data.shape[0]
    number_of_dimensions = data.shape[1]

    minimums = np.zeros(shape=(number_of_dimensions, 1))
    maximums = np.zeros(shape=(number_of_dimensions, 1))

    for dimension in range(number_of_dimensions):
        minimums[dimension] = np.min(data[:, dimension])
        maximums[dimension] = np.max(data[:, dimension])

    uniform_points = np.zeros(data.shape)
    for dimension in range(number_of_dimensions):
        uniform_points[:, dimension] = np.random.uniform(minimums[dimension],
                                                         maximums[dimension],
                                                         number_of_data_points)

    return uniform_points


def find_reference_dispersion(data, k, number_of_bootstraps=10):
    dispersions = np.zeros(shape=(number_of_bootstraps, 1))
    for run_number in range(number_of_bootstraps):
        uniform_points = generate_bounding_box_uniform_points(data)
        dispersion, _, _ = default_clustering(uniform_points, k, 10, 500)

        if 0 == dispersion:
            logging.warning(
                '[Reference Dispersion] Cannot take the log of 0 for run '
                'number = {}.'.format(run_number))
            continue

        dispersions[run_number] = np.log(dispersion)

    mean_dispersions = np.mean(dispersions)
    stddev_dispersions = np.std(dispersions) / np.sqrt(1 + 1 /
                                                       number_of_bootstraps)

    return mean_dispersions, stddev_dispersions


def gap_statistic(data, k_max, number_of_bootstraps):
    actual_dispersions = np.zeros(shape=(k_max, 1))
    mean_ref_dispersions = np.zeros(shape=(k_max, 1))
    stddev_ref_dispersions = np.zeros(shape=(k_max, 1))
    for k in range(1, k_max):
        # add the current k's dispersion
        actual_dispersion, _, _ = default_clustering(data, k, 10, 500)

        if 0 == actual_dispersion:
            logging.warning(
                '[Actual Dispersion] Cannot take the log of 0 for k = {}.'
                .format(k))
            continue

        actual_dispersions[k] = np.log(actual_dispersion)

        # add the mean reference dispersion
        mean_ref_dispersions[k], stddev_ref_dispersions[k] =\
        find_reference_dispersion(data, k, number_of_bootstraps)

    gaps = np.zeros(shape=(k_max, 1))
    for k in range(1, k_max):
        gaps[k] = mean_ref_dispersions[k] - actual_dispersions[k]

    confidence = np.zeros(shape=(k_max, 2))
    for k in range(k_max):
        confidence[k, 0] = stddev_ref_dispersions[k]
        confidence[k, 1] = stddev_ref_dispersions[k]

    return gaps, confidence
