import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Gap statistic tool.')

    parser.add_argument('--input_file', dest='input_file', action='store',
                        default=None, help='A csv file containing rows and ' \
                                           'columns of data to cluster')

    parser.add_argument('--input_file_skip_lines',
                        dest='input_file_skip_lines', action='store',
                        default=0, help='The number of lines at the beginning'
                                        ' of the input file that should be '
                                        'skipped over.')

    parser.add_argument('--min_k', dest='min_k', action='store', default=1,
                        help='The minimum value of k to use.')

    parser.add_argument('--max_k', dest='max_k', action='store', default=10,
                        help='The maximum value of k to use.')

    parser.add_argument('--boostraps', dest='bootstraps', action='store',
                        default=10, help='The number of reference '
                                         'distributions to generate and run '
                                         'the clustering against.')

    parser.add_argument('--whiten', dest='whiten', action='store',
                        default=False, help='Whether the data should be '
                                            'whitened before it is clustered '
                                            'or not.')

    parser.add_argument('--plot_clusters', dest='plot_clusters',
                        action='store', default=False,
                        help='Plot the clusters in 2 dimensions when done, '
                             'using the optimal k that\'s found.')

    args = parser.parse_args()

    return args
