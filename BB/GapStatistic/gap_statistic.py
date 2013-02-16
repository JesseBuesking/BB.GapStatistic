"""
Methods for calculating the gap statistic.
"""
import datetime

import logging
from sklearn.cluster import KMeans
import numpy as np


def default_clustering(data, k, individual_runs=10, iterations=300):
    """
    The default clustering method.
    """
    estimator = KMeans(init='k-means++',
                       precompute_distances=False,
                       n_clusters=k,
                       n_init=individual_runs,
                       max_iter=iterations)
    estimator.fit(data)

    return estimator.inertia_, estimator.labels_, estimator.cluster_centers_


def generate_bounding_box_uniform_points(data):
    """
    Generates a bounding box of uniform points around the data provided.
    """
    logging.info('Generating a bounding box.')
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


def generate_principal_components_box_uniform_points(data):
    """
    Generates a box aligned with the principal components of the data.
    """
    logging.info('Using pca to generate bounding box')
    x = np.asmatrix(data)
    _, _, vt = np.linalg.svd(x)
    x_prime = x * vt.T
    z_prime = generate_bounding_box_uniform_points(x_prime)
    z = z_prime * vt
    return z


def find_reference_dispersion(data, k, number_of_bootstraps=10):
    """
    Finds the reference dispersion (and confidence) for the data supplied.
    """
    dispersions = np.zeros(shape=(number_of_bootstraps, 1))

    logging.info('Finding {} reference dispersions'.format(number_of_bootstraps))
    for run_number in range(number_of_bootstraps):
        start = datetime.datetime.utcnow()
        logging.info('At iteration {}'.format(run_number))
#        uniform_points = generate_principal_components_box_uniform_points(data)
        uniform_points = generate_bounding_box_uniform_points(data)
        dispersion, _, _ = default_clustering(uniform_points, k, 1, 500)

        if 0 == dispersion:
            logging.warning(
                '[Reference Dispersion] Cannot take the log of 0 for run '
                'number = {}.'.format(run_number))
            continue

        dispersions[run_number] = np.log(dispersion)
        end = datetime.datetime.utcnow()
        logging.info('reference\'s duration: {}'.format((end - start)))

    mean_dispersions = np.mean(dispersions)
    stddev_dispersions = np.std(dispersions) / np.sqrt(1 + 1 /
                                                       number_of_bootstraps)

    return mean_dispersions, stddev_dispersions


def gap_statistic(data, min_k, max_k, number_of_bootstraps):
    """
    Calculates the gap statistic for the data supplied.
    """
    actual_dispersions = np.zeros(shape=(max_k + 1, 1))
    mean_ref_dispersions = np.zeros(shape=(max_k + 1, 1))
    stddev_ref_dispersions = np.zeros(shape=(max_k + 1, 1))

    logging.info('Running gap statistic for k between {} and {}'.format(min_k,
                                                                        max_k + 1))
    start = datetime.datetime.utcnow()
    for k in range(min_k, max_k + 1):
        logging.info('At k = {}'.format(k))

        # Add the current k's dispersion.
        actual_dispersion, _, _ = default_clustering(data, k, 10, 500)
        logging.info('Actual dispersion found')

        if 0 == actual_dispersion:
            logging.warning(
                '[Actual Dispersion] Cannot take the log of 0 for k = {}'
                .format(k))
            continue

        actual_dispersions[k] = np.log(actual_dispersion)

        # Add the mean reference dispersion.
        mean_ref_dispersions[k], stddev_ref_dispersions[k] =\
            find_reference_dispersion(data, k, number_of_bootstraps)

    end = datetime.datetime.utcnow()
    logging.info('k\'s duration: {}'.format((end - start)))

    logging.info('Finding actual gaps')
    gaps = np.zeros(shape=(max_k, 1))
    for k in range(min_k, max_k):
        gaps[k] = mean_ref_dispersions[k] - actual_dispersions[k]

    logging.info('Finding confidence values')
    confidence = np.zeros(shape=(max_k, 2))
    for k in range(min_k, max_k):
        confidence[k, 0] = stddev_ref_dispersions[k]
        confidence[k, 1] = stddev_ref_dispersions[k]

    return gaps, confidence


def whiten(data: []):
    """
    Takes the data and `whitens` it. (scales the data)

    http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq
    .whiten.html#scipy.cluster.vq.whiten
    """

    logging.info('Whitening input data')
    if 0 == len(data):
        logging.error('unable to whiten the data supplied since there are 0 '
                      'entries')
        raise Exception('unable to whiten the data supplied since there are 0 '
                      'entries')

    totals = dict()
    counter = 0
    for row in data:
        counter += 1
        for index, col in enumerate(row):
            if index not in totals:
                totals[index] = col
            else:
                totals[index] += col

    averages = dict()
    for key in totals.keys():
        averages[key] = totals[key] / counter

    standard_deviations = dict()
    for row in data:
        for index, col in enumerate(row):
            val = (col - averages[index]) ** 2
            if index not in standard_deviations:
                standard_deviations[index] = val
            else:
                standard_deviations[index] += val

    for key in standard_deviations.keys():
        standard_deviations[key] = (standard_deviations[key] / (counter - 1))\
                                   ** .5

    for row_index, row in enumerate(data):
        for col_index, col in enumerate(row):
            if standard_deviations[col_index] == 0:
                data[row_index][col_index] = 0
            else:
                data[row_index][col_index] = (col - averages[col_index]) / \
                                             standard_deviations[col_index]
    logging.info('Finished whitening input data')

    return data, standard_deviations, averages

