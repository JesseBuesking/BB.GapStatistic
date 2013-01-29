from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def gap_statistic(xy, k, iterations):
    xmin = np.min(xy[:,0])
    xmax = np.max(xy[:,0])
    ymin = np.min(xy[:,1])
    ymax = np.max(xy[:,1])

    ind_runs = 10
    max_iter = 500
    #    n_jobs = 4
    gap_estimator = KMeans(init='k-means++', precompute_distances=True,
                           n_clusters=k, n_init=ind_runs, max_iter=max_iter)

    mean = 0
    for i in range(iterations):
        unif = np.random.uniform(0, 100, (xy.size / 2, 2))
        unif[:,0] = ((xmax - xmin) * (unif[:,0] - 0)) / (100 - 0)
        unif[:,1] = ((ymax - ymin) * (unif[:,1] - 0)) / (100 - 0)
    #        plt.plot(unif[:,0], unif[:,1], 'o')
    #        plt.show()
        gap_estimator.fit(unif)
        mean += gap_estimator.inertia_

    mean /= iterations

    return mean

def main():
    k_range = 10
    values = np.zeros(shape=(k_range,2))
    np.random.seed(1)

    cl1 = np.random.normal(0, 1, (20, 2))
    cl2 = np.random.normal(3, 1, (20, 2))
    cl3 = np.random.normal(0, 1, (20, 2))

    cl2[:, 1] += 2
    cl3[:, 0] += 5

    xy = np.concatenate((cl1, cl2, cl3))

    for k in range(1, k_range):
        ind_runs = 10
        max_iter = 500
        #    n_jobs = 4
        estimator = KMeans(init='k-means++', precompute_distances=True,
                           n_clusters=k, n_init=ind_runs, max_iter=max_iter)
        estimator.fit(xy)

        mean = gap_statistic(xy, k, 10)

        values[k, 0] = estimator.inertia_
        values[k, 1] = mean

    gap_results = np.zeros(shape=(k_range,1))
    for i in range(1, k_range):
        gap_results[i] = np.log(values[i, 1]) - np.log(values[i, 0])

    plt.plot([i for i in range(k_range)], gap_results)
#    plt.plot(xy[:,0], xy[:,1], 'o')

#    color_choices = [[round(np.random.uniform(0, 1),1),
#                      round(np.random.uniform(0, 1),1),
#                      round(np.random.uniform(0, 1),1)] for _ in range(3)]
#
#    colors = ([color_choices[i] for i in estimator.labels_])
#
#    plt.scatter(xy[:, 0], xy[:, 1], c=colors)
#
#    plt.scatter(estimator.cluster_centers_[:,0],
#                estimator.cluster_centers_[:,1],
#                marker='o', s = 200, linewidths=2, c='none')
#
#    plt.scatter(estimator.cluster_centers_[:,0],
#                estimator.cluster_centers_[:,1],
#                marker='o', s = 150, linewidths=2, c='red')

    plt.show()

if __name__ == "__main__":
    main()
