import numpy as np
import random
import matplotlib.pyplot as plt

import matplotlib.animation as animation
import itertools


def compute_distance(x, y, power):
    """Compute Minkowski distance between x and y of finite dimensions"""
    distance = 0
    for i in range(len(x)):
        distance += (x[i] - y[i])**power
    distance = distance**(1/power)
    return float(distance)

def compute_barycentre(X):
    """Compute the coordinates of the center of a set of points of finite dimensions"""
    barycentre = [float(X[:,j].sum()/len(X[:,j])) for j in range(X.shape[1])]
    return barycentre


class Kmeans():

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.plot_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        
    def fit(self, data, dist=2, n_iter=5):
        """data is a list of points, each of which being a list of coordinates (float)"""
        self.data = np.array(data)
        self.dist = dist
        # run clustering several times
        centroids_list, labels_list, sil_score_list, sil_coeffs_list, sil_scores_during_learning_list, n_iter_list  = [], [], [], [], [], []
        for _ in range(n_iter):
            self.fit_once()
            centroids_list.append(self.centroids.copy())
            labels_list.append(self.labels.copy())
            sil_score_list.append(self.silhouette_score)
            sil_coeffs_list.append(self.silhouettes.copy())
            sil_scores_during_learning_list.append(self.sil_scores_during_learning.copy())
            n_iter_list.append(self.n_iter)
        # choose best partition
        best_clustering = sil_score_list.index(max(sil_score_list))
        self.centroids = centroids_list[best_clustering]
        self.labels = labels_list[best_clustering]
        self.silhouette_score = sil_score_list[best_clustering]
        self.silhouettes = sil_coeffs_list[best_clustering]
        self.sil_scores_during_learning = sil_scores_during_learning_list[best_clustering]
        self.n_iter = n_iter_list[best_clustering]
        
    def fit_once(self):
        # initialization
        self.centroids = [self.choose_random_point() for _ in range(self.n_clusters)]
        self.labels = np.array([0] * len(self.data))
        # learning
        self.__stop = False
        self.sil_scores_during_learning = []
        self.n_iter = 0
        while not self.__stop:
            self.__stop = True
            self.perform_iteration()
            self.sil_scores_during_learning.append(self.silhouette_score)

    def perform_iteration(self):
        self.n_iter += 1
        self.reassign_points_to_cluster()
        self.compute_new_centroids()
        self.compute_sil_score()
        
        
    def reassign_points_to_cluster(self):
        for i, observation in enumerate(self.data):
            new_cluster = self.assign_point_to_cluster(observation)
            if new_cluster != self.labels[i]:
                self.__stop = False
                self.labels[i] = new_cluster

    def compute_new_centroids(self):
        for k in range(self.n_clusters):
            mask = np.where(self.labels == k)[0]
            if not len(mask):
                self.centroids[k] = self.choose_random_point()
            else:
                self.centroids[k] = compute_barycentre(self.data[mask])


    def compute_sil_score(self):
        self.silhouettes = np.array([None] * len(self.data))
        for i in range(len(self.data)):
            self.compute_sil_coeff(i)
        silhouette_score = 0
        for k in range(self.n_clusters):
            mask = np.where(self.labels == k)[0]
            if len(mask):
                sil_k = self.silhouettes[mask].sum() / len(mask)
            else:
                sil_k = 0
            silhouette_score += sil_k
        self.silhouette_score = silhouette_score/self.n_clusters

    def compute_sil_coeff(self, i):
        # intra
        label = self.labels[i]
        a = self.compute_avg_distance_to_cluster(i, label)
        # neighbor cluster
        other_clusters = list(range(self.n_clusters))
        other_clusters.remove(label)
        avg_dists = [self.compute_avg_distance_to_cluster(i, k) for k in other_clusters]
        avg_dists = [val for val in avg_dists if val != 0]
        b = min(avg_dists)
        # silhouette coeff
        try:
            self.silhouettes[i] = (b-a)/max(a,b)
        except ZeroDivisionError:
            pass

    def compute_avg_distance_to_cluster(self, i, k):
        mask = np.where(self.labels == k)[0]
        if self.labels[i] == k:
            mask = np.setdiff1d(mask,i)
        if not len(mask):
            avg_dist = 0
        else:
            avg_dist = sum([compute_distance(self.data[i], self.data[j], self.dist) for j in mask])/len(mask)
        return avg_dist
        
    def predict(self, x):
        one_point = False
        if type(x) in [float, np.float32, np.float64]:
            one_point = True
            x = [x]
        np.append(self.data, np.array(x), axis=0)
        predicted_label = [self.assign_point_to_cluster(x_i) for x_i in x]
        np.append(self.data, x, axis=0)
        np.append(self.labels, predicted_label, axis=0)
        if one_point:
            predicted_label = predicted_label[0]
        return predicted_label
    
    def plot_clustering(self, y=[]):
        centroids = np.array(self.centroids)
        plt.scatter(self.data[:,0], self.data[:,1], c=[self.plot_colors[label] for label in self.labels])
        plt.scatter(centroids[:,0], centroids[:,1], marker = "*", c=[self.plot_colors[label] for label in range(self.n_clusters)], s=200, edgecolors='black')
        if len(y):
            wrongs = self.find_points_in_the_wrong_cluster(y)
            plt.scatter(self.data[wrongs,0], self.data[wrongs,1], s=80, facecolors='none', edgecolors='r')
        plt.show()

    def plot_silhouettes(self):
        ordered_silhouettes = dict()
        for k in range(self.n_clusters):
            mask = np.where(self.labels == k)[0]
            ordered_silhouettes[k] = sorted(list(self.silhouettes[mask]))

        fig, axs = plt.subplots(ncols=1, nrows=self.n_clusters, figsize=(7, 7), layout="constrained")
        for k in range(self.n_clusters):
            row = k
            axs[row].barh(range(len(ordered_silhouettes[k])), ordered_silhouettes[k], color=self.plot_colors[k])
            axs[row].title.set_text(f'cluster {k}')
        plt.show()



    def plot_learning(self):
        plt.plot(range(len(self.sil_scores_during_learning)), self.sil_scores_during_learning)
        plt.xlabel('learning iterations')
        plt.ylabel('silhouette score')
        plt.show()

    def find_points_in_the_wrong_cluster(self, y):
        self.change_label_numbers(y)
        wrongs = []
        for i in range(len(self.y)):
            if self.labels[i] != self.y[i]:
                wrongs.append(i)
        return wrongs

    def change_label_numbers(self, y):
        self.accuracy = 0
        for labels_order in itertools.permutations(range(self.n_clusters)):
            ord_y = [labels_order[old_label] for old_label in y]
            acc = self.compute_accuracy(ord_y)
            if acc > self.accuracy:
                self.accuracy  = acc
                self.y = ord_y
                
    def compute_accuracy(self, y):
        score = 0
        for i in range(len(y)):
            if y[i] == self.labels[i]:
                score += 1
        return score/len(y)

    def assign_point_to_cluster(self, x):
        dists = [compute_distance(x, centroid, power=self.dist) for centroid in self.centroids]
        cluster = dists.index(min(dists))
        return cluster

    def choose_random_point(self):
        point = []
        for j in range(self.data.shape[1]):
            space_min, space_max = self.data[:,j].min(), self.data[:,j].max()
            point.append(float(random.uniform(space_min, space_max)))
        return point

    def shoot_learning_video(self, data, dist=2, n_iter=5):
        self.data = data
        self.dist = dist
        plt.rcParams['figure.figsize'] = [8, 8]
        fig, ax = plt.subplots()
        ax.set_axis_off()
        ims = []
        
        for i in range(n_iter):
            # initialization
            self.centroids = [self.choose_random_point() for _ in range(self.n_clusters)]
            self.labels = np.array([0] * len(self.data))
            im1 = ax.scatter(self.data[:,0], self.data[:,1], c='grey', alpha=0.3, s=100, edgecolors='none', animated=True)
            im2 = ax.annotate(f'random\ninitialization\n#{i+1}', (0.5, 0.5), fontsize=18, ha='center', va='center')
            for _ in range(5):
                ims.append([im1, im2])
            self.centroids = [self.choose_random_point() for _ in range(self.n_clusters)]
            ims, ax = self.take_data_photo(ims, ax)
            # learning
            self.__stop = False
            self.sil_scores_during_learning = []
            self.n_iter = 0
            while not self.__stop:
                self.__stop = True
                self.n_iter += 1
                self.reassign_points_to_cluster()
                ims, ax = self.take_data_photo(ims, ax)
                self.compute_new_centroids()
                ims, ax = self.take_data_photo(ims, ax)
            for _ in range(5):
                ims, ax = self.take_data_photo(ims, ax)
        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
        writergif = animation.PillowWriter()
        ani.save('learning_video.gif',writer=writergif)
        plt.show()

    def take_data_photo(self, ims, ax):
        centroids = np.array(self.centroids)
        im1 = ax.scatter(self.data[:,0], self.data[:,1], c=[self.plot_colors[label] for label in self.labels], alpha=0.3, s=100, edgecolors='none', animated=True)
        im2 = ax.scatter(centroids[:,0], centroids[:,1], marker = "*", c=[self.plot_colors[label] for label in range(self.n_clusters)], s=500, edgecolors='black', animated=True)
        ims.append([im1, im2])
        return ims, ax
    