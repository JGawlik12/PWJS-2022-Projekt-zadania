from random import randint
import numpy as np


class MyKMeans:
    def __init__(self, n_clusters=5, max_iters=300):
        self.k = n_clusters
        self.max_iters = max_iters

    def fit_predict(self, X):
        X = self.__min_max_normalization(X)
        centroids = self.__create_centroids(X)
        labels = []
        for _ in range(self.max_iters):
            distance = self.__calc_distance_to_centroids(X, centroids)
            labels = self.__create_labels(distance)
            new_centroids = self.__calc_new_centroids(X, labels)
            if self.__compare_centroids(centroids, new_centroids):
                break
            else:
                centroids = new_centroids
        return labels

    def __create_centroids(self, X):
        indexes = []
        for _ in range(self.k):
            while True:
                index = randint(0, np.shape(X)[0]-1)
                if index not in indexes:
                    indexes.append(index)
                    break

        return [X[index, :] for index in indexes]

    @staticmethod
    def __euclidean_distance(x1, x2):
        distance = 0.0

        for i in range(np.shape(x1)[0]):
            distance += (x1[i] - x2[i])**2
        distance **= 0.5
        return distance

    def __calc_distance_to_centroids(self, X, centroids):
        distance = np.ndarray(shape=(np.shape(X)[0], len(centroids)), dtype=float)
        for cent_index in range(len(centroids)):
            for i in range(np.shape(X)[0]):
                point = X[i, :]
                distance[i, cent_index] = self.__euclidean_distance(centroids[cent_index], point)
        return distance

    def __min_max_normalization(self, X):
        normalized_data = np.ndarray(shape=np.shape(X), dtype=float)
        for i in range(np.shape(X)[1]):
            mx = float(max(X[:, i]))
            mn = float(min(X[:, i]))
            normalized_data[:, i] = (X[:, i] - mn) / (mx - mn)
        return normalized_data

    def __create_labels(self, distance_table):
        labels = np.ndarray(shape=(np.shape(distance_table)[0],), dtype=int)
        for point in range(np.shape(distance_table)[0]):
            labels[point] = self.__find_index_of_min(distance_table[point, :])

        return labels

    def __find_index_of_min(self, row):
        if np.shape(row)[0] == 1:
            return 0
        index = 0
        for i in range(1, np.shape(row)[0]):
            if row[index] > row[i]:
                index = i

        return index

    def __calc_new_centroids(self, X, labels):
        new_centroids = []

        for i in range(self.k):
            x = X[np.where(labels == i)]
            new_centroids.append(self.__find_central_point(x))
        return new_centroids

    def __find_central_point(self, points):
        centroid = np.array([0 for _ in range(np.shape(points)[1])], dtype=float)
        for dimension in range(np.shape(points)[1]):
            centroid[dimension] = float(np.sum(points[:, dimension])) / float(np.shape(points)[0])
        return centroid

    def __compare_centroids(self, c1, c2):
        for centroid_index in range(len(c1)):
            for dimension in range(np.shape(c1)[1]):
                if c1[centroid_index][dimension] != c2[centroid_index][dimension]:
                    return False
        return True