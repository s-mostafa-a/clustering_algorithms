import numpy as np
import random
import pandas
import matplotlib.pyplot as plt
from sklearn import metrics


def compute_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))


def assign_label_cluster(distance, data_point, centroids):
    index_of_minimum = min(distance, key=distance.get)
    return [index_of_minimum, data_point, centroids[index_of_minimum]]


def compute_new_centroids(cluster_label, centroids):
    return np.array(cluster_label + centroids) / 2


def iterate_k_means(data_points, centroids, total_iteration):
    cluster_label = []
    total_points = len(data_points)
    k = len(centroids)
    for iteration in range(0, total_iteration):
        for index_point in range(0, total_points):
            distance = {}
            for index_centroid in range(0, k):
                distance[index_centroid] = compute_euclidean_distance(data_points[index_point],
                                                                      centroids[index_centroid])
            label = assign_label_cluster(distance, data_points[index_point], centroids)
            centroids[label[0]] = compute_new_centroids(label[1], centroids[label[0]])

            if iteration == (total_iteration - 1):
                cluster_label.append(label)

    return [cluster_label, centroids]


def create_centroids(X, k):
    rands = []
    centroids = []
    for x in range(k + 1):
        r = random.randint(0, len(X) - 1)
        while r in rands:
            r = random.randint(0, len(X) - 1)
        rands.append(r)
        centroids.append(X[r])
    return np.array(centroids)


def running_k_means():
    data_points = np.asarray(pandas.read_csv('./data/Dataset1.csv').values.tolist())
    for j in range(2, 5):
        fig, ax = plt.subplots()
        number_of_clusters = j
        centroids = create_centroids(data_points, number_of_clusters)
        total_iteration = 10
        [cluster_label, new_centroids] = iterate_k_means(data_points, centroids, total_iteration)
        cluster_list = []
        for i in range(number_of_clusters):
            cluster_list.append([list(x[1]) for x in cluster_label if x[0] == i])
        for i in range(number_of_clusters):
            xs = [x[0] for x in cluster_list[i]]
            ys = [x[1] for x in cluster_list[i]]
            color = str(i + 1)
            ax.plot(xs, ys, color)
        name = './figures/k_means_when_k_is_' + str(number_of_clusters) + '.png'
        fig.savefig(name)


def evaluation():
    data_points = np.asarray(pandas.read_csv('./data/Dataset1.csv').values.tolist())
    mean_errors = []
    for j in range(1, 30):
        number_of_clusters = j
        centroids = create_centroids(data_points, number_of_clusters)
        total_iteration = 10
        [cluster_label, new_centroids] = iterate_k_means(data_points, centroids, total_iteration)
        cluster_list = []
        cluster_arrays = []
        error_from_center = []
        for i in range(number_of_clusters):
            cluster_list.append([list(x[1]) for x in cluster_label if x[0] == i])
            cluster_arrays.append(np.asarray([x[1] for x in cluster_label if x[0] == i]))
            sub_res = np.subtract(cluster_arrays[i][:], new_centroids[i])
            error_from_center.append(np.mean(np.sqrt(np.sum((sub_res) ** 2, axis=1))))
        mean_errors.append(np.mean(np.asarray(error_from_center)))
    plt.plot(list(range(1, len(mean_errors) + 1)), mean_errors, color='g')
    plt.xlabel('K')
    plt.ylabel('Mean error')
    plt.title('Average clustering error of k-means for different values of k')
    plt.savefig('./figures/ks.png')


def weaknesses_and_restrictions_of_k_means():
    data_points = np.asarray(pandas.read_csv('./data/Dataset2.csv').values.tolist())
    number_of_clusters = 3
    centroids = create_centroids(data_points, number_of_clusters)
    total_iteration = 10
    [cluster_label, new_centroids] = iterate_k_means(data_points, centroids, total_iteration)
    cluster_list = []
    for i in range(number_of_clusters):
        cluster_list.append([list(x[1]) for x in cluster_label if x[0] == i])
    for i in range(number_of_clusters):
        xs = [x[0] for x in cluster_list[i]]
        ys = [x[1] for x in cluster_list[i]]
        color = str(i + 1)
        plt.plot(xs, ys, color)
    name = './figures/shows_weakness_of_k_means.png'
    plt.savefig(name)


def silhouette_score_on_ds1_when_k_is(n):
    data_points = np.asarray(pandas.read_csv('./data/Dataset1.csv').values.tolist())
    fig, ax = plt.subplots()
    number_of_clusters = n
    centroids = create_centroids(data_points, number_of_clusters)
    total_iteration = 10
    [cluster_label, new_centroids] = iterate_k_means(data_points, centroids, total_iteration)
    cluster_list = []
    labels_list = []
    all_points = []
    all_labels = []
    for i in range(number_of_clusters):
        cluster_list.append([list(x[1]) for x in cluster_label if x[0] == i])
        labels_list.append(list(np.ones(len(cluster_list[i])) * (i + 1)))
    for i in range(number_of_clusters):
        all_points += cluster_list[i]
        all_labels += labels_list[i]
    unique_labels = set(all_labels)
    if len(unique_labels) > 1:
        print("Silhouette Coefficient for k-means on dataset1 k = " + str(n) + ": %f" % metrics.silhouette_score(all_points, all_labels))
    for i in range(number_of_clusters):
        xs = [x[0] for x in cluster_list[i]]
        ys = [x[1] for x in cluster_list[i]]
        color = str(i + 1)
        ax.plot(xs, ys, color)
    name = './figures/for_silhouette_score_on_ds1_when_k_is_' + str(n) + '.png'
    fig.savefig(name)


def silhouette_score_on_ds2_when_k_is_4():
    data_points = np.asarray(pandas.read_csv('./data/Dataset2.csv').values.tolist())
    fig, ax = plt.subplots()
    number_of_clusters = 4
    centroids = create_centroids(data_points, number_of_clusters)
    total_iteration = 10
    [cluster_label, new_centroids] = iterate_k_means(data_points, centroids, total_iteration)
    cluster_list = []
    labels_list = []
    all_points = []
    all_labels = []
    for i in range(number_of_clusters):
        cluster_list.append([list(x[1]) for x in cluster_label if x[0] == i])
        labels_list.append(list(np.ones(len(cluster_list[i])) * (i + 1)))
    for i in range(number_of_clusters):
        all_points += cluster_list[i]
        all_labels += labels_list[i]
    unique_labels = set(all_labels)
    if len(unique_labels) > 1:
        print("Silhouette Coefficient for k-means on dataset2 k = 4: %f" % metrics.silhouette_score(all_points, all_labels))
    for i in range(number_of_clusters):
        xs = [x[0] for x in cluster_list[i]]
        ys = [x[1] for x in cluster_list[i]]
        color = str(i + 1)
        ax.plot(xs, ys, color)
    name = './figures/for_silhouette_score_on_ds2_when_k_is_4.png'
    fig.savefig(name)

if __name__ == "__main__":
    # running_k_means()
    # evaluation()
    # weaknesses_and_restrictions_of_k_means()
    silhouette_score_on_ds1_when_k_is(2)
    silhouette_score_on_ds1_when_k_is(4)
    silhouette_score_on_ds2_when_k_is_4()
