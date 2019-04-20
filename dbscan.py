import pandas
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt


def main():
    data_points = np.asarray(pandas.read_csv('./data/Dataset2.csv').values.tolist())
    clustering = DBSCAN(eps=2, min_samples=10).fit(data_points)
    plt.plot(clustering[0], clustering[1], 'g')
    '''
    print(clustering.labels_)
    xs = [float(x[0]) for x in data_points]
    ys = [float(x[1]) for x in data_points]
    for i in range(len(data_points)):
        if clustering.labels_[i] == -1:
            plt.plot(xs[i], ys[i], 'g')
        else:
            plt.plot(xs[i], ys[i], str(clustering.labels_[i]+1))
    '''
    plt.savefig('./figures/dbscan.png')

if __name__ == "__main__":
    main()
