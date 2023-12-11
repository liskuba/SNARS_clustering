import numpy as np
import networkx as nx
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


def determine_number_of_clusters(X):
    silhouette_results = []
    for i in range(2, 15):
        kmeans = KMeans(n_clusters=i, random_state=0, n_init="auto").fit(X[:, 1:i].real)
        labels = kmeans.labels_
        silhouette_results.append(silhouette_score(X[:, 1:i].real, labels, random_state=0))

    return np.argmax(silhouette_results) + 2


def spectral_clustering(G, num_of_clusters):
    L = nx.laplacian_matrix(G).todense()

    eigenvals, eigenvecs = np.linalg.eig(L)
    eigenvecs = eigenvecs[:, np.argsort(eigenvals)]

    if num_of_clusters is None:
        num_of_clusters = determine_number_of_clusters(eigenvecs)

    kmeans = KMeans(n_clusters=num_of_clusters, random_state=0, n_init="auto").fit(eigenvecs[:, 1:num_of_clusters].real)
    labels = kmeans.labels_

    return labels, num_of_clusters
