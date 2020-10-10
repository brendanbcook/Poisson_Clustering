import sklearn.datasets as datasets
from graphcluster import update_centroids, gaussian_mixture, build_graph, poisson_cluster_kmeans
from graphlearning import graph_laplacian
import numpy as np
import pytest


X, y = datasets.load_wine(return_X_y=True)
indices = np.arange(X.shape[0])
k = 3
W = build_graph(X, k=10)
L = graph_laplacian(W)
indices = np.arange(X.shape[0])

def test_poisson_kmeans():
    centroid_inits = [[129, 147, 127]]
    methods = ['dijkstra', 'laplacian_eigenvector', 'poisson_solve']
    A = np.load('tests/graphcluster_test_labels.npz')
    expected = {method: A[method] for method in methods}
    for method in methods:
        message = f"Poisson k-means for method = {method} was not equal to expected value."
        y = poisson_cluster_kmeans(W, L, k, centroid_method=method, centroid_inits=centroid_inits, num_iter=3, num_starts=1)
        assert expected[method] == pytest.approx(y), message