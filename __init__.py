import numpy as np
import matplotlib.pyplot as plt
import pkg_resources
from sklearn.cluster import KMeans
import BB.GapStatistic
from BB.GapStatistic.gap_statistic import gap_statistic


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

    # find the value of k with the maximum gap
    max_k = gaps.argmax(axis=0)

    # renaming variables for clarity
    x = np.arange(0, k_max)
    y = gaps

    # create a new plot
    fig = plt.figure(1)

    # not sure what i was doing here...
    ax = fig.gca()

    # draw some grid lines
    ax.yaxis.grid(color='0.75', linestyle='dashed')
    ax.xaxis.grid(color='0.75', linestyle='dashed')

    # plot the gaps plus the confidence intervals
    plt.errorbar(x, y.flatten(), yerr=confidence.T, ecolor='b', color='k')

    # circle the k corresponding to the largest gap, and annotate it
    plt.scatter(max_k, y[max_k], facecolors='none', edgecolors='r', s=200)
    ax.text(max_k, gaps[max_k], "max at k = {}".format(max_k[0]))

    # show all ticks
    plt.xticks(x, x)

    # label it up!
    plt.xlabel("number of clusters")
    plt.ylabel("gap")
    plt.title("Estimating the number of clusters via the gap statistic")

    # now we can finally render the plot
    plt.show()

if __name__ == "__main__":
    main()


#### old plotting code

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
