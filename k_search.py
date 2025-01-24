import numpy as np
import random
import matplotlib.pyplot as plt
import itertools

from kmeans import Kmeans

class Ksearch():

    def __init__(self, data, dist=2, n_iter=5):
        self.data = data
        self.dist = dist
        self.n_iter = n_iter

    def find_best_k(self, k_range=None):
        if not k_range:
            k_min = 2
            k_max = int(len(self.data)/10)
        else:
            k_min, k_max = k_range
        self.k_range = k_min, k_max
        self.sil_scores = []
        for n_clusters in range(k_min, k_max):
            kmeans = Kmeans(n_clusters)
            kmeans.fit(self.data, self.dist, self.n_iter)
            self.sil_scores.append(kmeans.silhouette_score)
    
    def plot_elbow(self):
        k_min, k_max = self.k_range
        plt.plot(range(k_min, k_max), self.sil_scores)
        plt.xlabel('n_clusters')
        plt.ylabel('silhouette score')
        plt.show()
