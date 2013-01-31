import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def plot_gaps(gaps, confidence, dimensions):
    """
    Plots the gaps with error bars to show confidence.
    """

    # Find the value of k with the maximum gap.
    max_k = gaps.argmax(axis=0)

    # Renaming variables for clarity.
    x = np.arange(0, gaps.shape[0])
    y = gaps

    # Create a new plot.
    fig = plt.figure(1)

    # Not sure what I was doing here...
    ax = fig.gca()

    # Draw some grid lines.
    ax.yaxis.grid(color='0.75', linestyle='dashed')
    ax.xaxis.grid(color='0.75', linestyle='dashed')

    # Plot the gaps plus the confidence intervals.
    plt.errorbar(x, y.flatten(), yerr=confidence.T, ecolor='b', color='k')

    # Circle the k corresponding to the largest gap, and annotate it.
    plt.scatter(max_k, y[max_k], facecolors='none', edgecolors='r', s=200)
    ax.text(max_k, gaps[max_k], "max at k = {}".format(max_k[0]))

    # Show all ticks.
    plt.xticks(x, x)

    # Label it up!
    plt.xlabel("number of clusters")
    plt.ylabel("gap")
    plt.title("Estimating the number of clusters via the gap statistic in {} "
              "dimensions".format(dimensions))

    # Now we can finally render the plot.
    plt.show()


def plot_clusters(data, point_map, centroids):
    """
    Plots the clusters defined by the point_map and centroids supplied,
    coloring the points according to their associated clusters and marking
    their centroids.
    """

    # Verifying the data.
    if 3 < data.shape[1]:
        logging.error('Cannot plot more than 3 dimensions, '
                      'but {} were supplied.'
        .format(data.shape[1]))
        return

    # Creating colors randomly for each cluster.
    color_choices = [[round(np.random.uniform(0, 1), 1),
                      round(np.random.uniform(0, 1), 1),
                      round(np.random.uniform(0, 1),
                            1)] for _ in range(centroids.shape[0])]

    # Assigning colors to the points in the point map.
    colors = ([color_choices[i] for i in point_map])

    if 1 == data.shape[1]:
        plot_1d(data, centroids, colors)
    if 2 == data.shape[1]:
        plot_2d(data, centroids, colors)
    if 3 == data.shape[1]:
        plot_3d(data, centroids, colors)


def plot_1d(data, centroids, colors):
    # Plotting the points.
    plt.scatter(data[:, 0], c=colors)

    # Plotting red circles to mark the cluster centroids.
    plt.scatter(centroids[:, 0], marker='o', s=150, linewidths=2, c='red')

    plt.show()


def plot_2d(data, centroids, colors):
    # Plotting the points.
    plt.scatter(data[:, 0], data[:, 1], c=colors)

    # Plotting red circles to mark the cluster centroids.
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=150, linewidths=2, c='red')

##### plot in 3d`
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    # Plotting the points.
#    ax.scatter(data[:, 0], data[:, 1], c=colors)
#
#    # Plotting red circles to mark the cluster centroids.
#    ax.scatter(centroids[:, 0], centroids[:, 1],
#               marker='o', s=150, linewidths=2, c='red')
    plt.show()


def plot_3d(data, centroids, colors):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plotting the points.
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors)

    # Plotting red circles to mark the cluster centroids.
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                marker='o', s=150, linewidths=2, c='red')

    plt.show()

