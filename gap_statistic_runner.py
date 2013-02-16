"""
A gap statistic runner.
"""
import logging

from BB.ArgumentParsing.parse import parse_arguments
from BB.GapStatistic.gap_statistic import gap_statistic, default_clustering, whiten
from BB.InputOutput import read
from BB.Number import number
#from BB.Plotting.plot import plot_gaps, plot_clusters_pca_reduced


def main():
    """
    The main method.
    """
    logging.basicConfig(level=logging.INFO)
    logging.StreamHandler()

    # Parse dem arrrrrgs!
    args = parse_arguments()

    # Read the input file in.
    skip_lines = number.to_int(args.input_file_skip_lines)
    data = read.read_to_numpy_array(args.input_file, skip_lines)

    if args.whiten is True:
        data = whiten(data)

    min_k = number.to_int(args.min_k)
    if min_k is None:
        raise Exception('The min_k argument supplied is invalid. "{}"'
                        .format(min_k))

    max_k = number.to_int(args.max_k)
    if max_k is None:
        raise Exception('The max_k argument supplied is invalid. "{}"'
                        .format(max_k))

    bootstraps = number.to_int(args.bootstraps)
    if bootstraps is None:
        raise Exception('The bootstraps argument supplied is invalid. "{}"'
                        .format(bootstraps))

    # Find the gaps and confidence intervals.
    gaps, confidence = gap_statistic(data, min_k, max_k, bootstraps)

#    # Plot the gaps.
#    plot_gaps(gaps, confidence, data.shape[1])
#
#    if args.plot_clusters is True:
#        optimal_k = gaps.argmax(axis=0)
#        _, point_map, centroids = default_clustering(data, optimal_k, 10, 300)
#        plot_clusters_pca_reduced(data, point_map, centroids)

if __name__ == "__main__":
    main()
