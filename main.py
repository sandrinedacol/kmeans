from sklearn.datasets import make_blobs

from kmeans import Kmeans
from k_search import Ksearch


if __name__ == '__main__':
    samples = [100,50,200,100]
    X, y = make_blobs(n_samples=samples)
    kmeans = Kmeans(n_clusters=len(samples))

    # kmeans.fit(X)
    # kmeans.plot_clustering(y)
    # kmeans.plot_silhouettes()
    # kmeans.plot_learning()

    kmeans.shoot_learning_video(X, n_iter = 3)

    # k_search = Ksearch(X)
    # k_search.find_best_k()
    # k_search.plot_elbow() 