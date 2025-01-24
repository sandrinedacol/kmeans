from sklearn.datasets import make_blobs

from kmeans import Kmeans
from k_search import Ksearch


if __name__ == '__main__':

    X, y = make_blobs(n_samples=100)
    kmeans = Kmeans(n_clusters=3)

    # kmeans.fit(X)
    # kmeans.plot_clustering(y)
    # kmeans.plot_silhouettes()
    # kmeans.plot_learning()

    kmeans.shoot_learning_video(X)

    # k_search = Ksearch(X)
    # k_search.find_best_k()
    # k_search.plot_elbow()