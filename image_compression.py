from skimage import io
import numpy as np

from sklearn.cluster import KMeans


def compress(n_color):
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
    compress(256)
