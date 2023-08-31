from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def show(labels, cluster_centers, X):
    fig = plt.figure(figsize=(6, 4))
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(labels))))
    ax = fig.add_subplot(1, 1, 1)

    for k, col in zip(range(len(cluster_centers)), colors):
        my_member = (labels == k)
        cluster_center = cluster_centers[k]
        ax.plot(X[my_member, 0], X[my_member, 1], 'w', markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

    ax.set_title('KMeans')
    plt.show()


def main():
    np.random.seed(0)
    X, y = make_blobs(n_samples=5000, centers=[[4, 4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

    print('X', X)

    # plt.scatter(X[:, 0], X[:, 1], marker='.')

    k_means = KMeans(init='k-means++', n_clusters=6, n_init=12)
    k_means.fit(X)

    labels = k_means.labels_
    cluster_centers = k_means.cluster_centers_

    show(labels, cluster_centers, X)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
