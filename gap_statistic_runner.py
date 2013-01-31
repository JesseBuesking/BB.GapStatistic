import numpy as np
from BB.GapStatistic.gap_statistic import gap_statistic, default_clustering
from BB.Plotting.plot import plot_gaps, plot_clusters


def generate_fake_data():
    np.random.seed(1)

    cl1 = np.random.normal(0, 1, (20, 2))
    cl2 = np.random.normal(3, 1, (20, 2))
    cl3 = np.random.normal(0, 1, (20, 2))

    cl2[:, 1] += 2
    cl3[:, 0] += 5

    fake_data = np.concatenate((cl1, cl2, cl3))

    return fake_data


def main():
    k_max = 10 + 1
    number_of_bootstraps = 10
    data = generate_fake_data()
    gaps, confidence = gap_statistic(data, k_max, number_of_bootstraps)

    plot_gaps(gaps, confidence, data.shape[1])
    max_k = gaps.argmax(axis=0)
    if 3 > max_k:
        inertia, point_map, centroids = default_clustering(data, max_k, 10,
                                                           300)
        plot_clusters(data, point_map, centroids)

if __name__ == "__main__":
    main()
