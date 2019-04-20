import pandas
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def one_cluster_ds2():
    main(10, 1, 'dbscan_one_cluster_on_dataset2')


def best_fit_ds2():
    main(1.7, 10, 'dbscan_best_fit_on_dataset2')


def best_fit_ds1():
    main(0.18, 10, 'dbscan_best_fit_on_dataset1', dataset=1)


def two_clusters_ds2():
    main(2.8, 10, 'dbscan_two_clusters_on_dataset2', dataset=2)


def two_clusters_ds1():
    main(0.5, 10, 'dbscan_two_clusters_on_dataset1', dataset=1)


def main(eps, min_samples, file_name, dataset=2):
    fig, ax = plt.subplots()
    data_points = None
    if dataset == 2:
        data_points = np.asarray(pandas.read_csv('./data/Dataset2.csv').values.tolist())
    else:
        data_points = np.asarray(pandas.read_csv('./data/Dataset1.csv').values.tolist())
    X = data_points
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    if len(unique_labels) > 1:
        print("Silhouette Coefficient for " + file_name + ": %f" % metrics.silhouette_score(X, labels))
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)
        xy = X[class_member_mask & ~core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)
    ax.set_title('Estimated number of clusters: %d' % n_clusters_)
    fig.savefig(str('./figures/' + file_name + '.png'))


if __name__ == "__main__":
    best_fit_ds2()
    one_cluster_ds2()
    two_clusters_ds1()
    two_clusters_ds2()
    best_fit_ds1()
