from skimage import io
import numpy as np
import random

from sklearn.cluster import KMeans


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
            # injaaaa label shod label[2]
            if iteration == (total_iteration - 1):
                cluster_label.append(label[2])

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


def compress(n_color):
    """
    image = io.imread('./data/imageSmall.png')
    number_of_clusters = 16
    rows = image.shape[0]
    cols = image.shape[1]
    # image = image / 255
    X = image.reshape(image.shape[0] * image.shape[1], 3)
    centroids = create_centroids(X, number_of_clusters)
    total_iteration = 20
    [cluster_label, new_centroids] = iterate_k_means(X, centroids, total_iteration)
    labels = np.asarray(cluster_label)  # , dtype=np.uint8)
    np.save('codebook_img.npy', labels)
    # labels = np.load('./codebook_img.npy')
    print(labels[0])
    labels = np.reshape(labels, (rows, cols, 3))
    print(labels[0][0])
    io.imsave('compressed_img_256.png', labels)
    # X_recovered = new_centroids[cluster_label]
    # print(np.shape(X_recovered))
    # X_recovered = np.reshape(X_recovered, (rows, cols, 3))
    # print(np.shape(X_recovered))
    """
    image = io.imread('./data/imageSmall.png')
    io.imshow(image)
    rows = image.shape[0]
    cols = image.shape[1]
    image = image.reshape(image.shape[0] * image.shape[1], 3)
    kmeans = KMeans(n_clusters=n_color, max_iter=20)
    kmeans.fit(image)
    labels = list(np.asarray(kmeans.labels_, dtype=np.uint8))
    length = len(labels)
    for i in range(length):
        labels[i] = kmeans.cluster_centers_[kmeans.labels_[i]]
    labels = np.reshape(labels, (rows, cols, 3))
    io.imsave('./figures/compressed_img' + str(n_color) + '.png', labels)


if __name__ == '__main__':
    compress(16)
    # compress(256)
