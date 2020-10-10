'''
Graph Clustering Algorithms
Implemented by Brendan Cook
'''

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.cluster import KMeans
import sklearn
import seaborn as sns
import scipy
import ipdb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import graphlearning as gl
# To-do: Finish poisson_topdown
# Test poisson_kmeans

# Test Datasets
def gaussian_mixture(n, separation):
    #Make up some mixture of Gaussian data with 3 classes
    n = 500
    separation = 1
    X1 = np.random.randn(n, 2)
    L1 = np.zeros((n,))
    X2 = np.random.randn(n, 2) + separation*np.array([4,2])
    L2 = np.ones((n,))
    X3 = np.random.randn(n,2) + separation*np.array([0,4])
    L3 = 2*np.ones((n,))
    n = 3*n
    X = np.vstack((X1,X2,X3))
    L = np.hstack((L1,L2,L3))
    return X, L
def build_graph(X, k=10):
    #Build graph
    I,J,D = gl.knnsearch(X,k)
    W = gl.weight_matrix(I,J,D,k)
    W = gl.diag_multiply(W,0)
    if not gl.isconnected(W):
        print('Warning: Graph not connected')
    return W  

### Poisson K-means ###

def label_matrix(y):
    U, _ = gl.LabelsToVec(np.asarray(y))
    return U.transpose()

def update_centroids(W, L, y_pred, indices, n, k, centroid_method, centroid_samples=None):
    if centroid_method == 'laplacian_eigenvector':
        #eigenvectors = np.vstack((gl.dirichlet_eigenvectors(L, indices[y_pred != i] , 1)[0] for i in range(k)))
        # u_vals, _ = gl.dirichlet_eigenvectors(L, I=indices[y_pred != 0] , k=1)
        # for i in range(1, k):
        #     new_eigenvector, _ = gl.dirichlet_eigenvectors(L, I=indices[y_pred != i] , k=1)
        #     u_vals = np.vstack((u_vals, new_eigenvector))
        u_vals = np.vstack([gl.dirichlet_eigenvectors(L, I=indices[y_pred != i] , k=1)[0] for i in range(k)])
    elif centroid_method == 'poisson_solve':
        u_vals = np.vstack([gl.constrained_solve(L, I=indices[y_pred != i], g=np.zeros(len(indices[y_pred != i])), f=np.ones(n)) for i in range(k)])
    elif centroid_method == 'dijkstra':
        u_vals = np.vstack([gl.cDijkstra(W, I=indices[y_pred != i], g=np.zeros(len(indices[y_pred != i]))) for i in range(k)])
    else:
        raise ValueError('Invalid Centroid Method')
    u_vals = np.abs(u_vals)
    if centroid_samples is None:
        centroids = np.argmax(u_vals, axis=1)
        labels = np.arange(k)
    elif type(centroid_samples) != int:
        raise ValueError('centroid_samples must be an integer')
    else:
        probs = u_vals**1 / np.sum(u_vals**1, axis=1, keepdims=True)
        centroids, labels = [], []
        for i in range(k):
            centroids.append(np.random.choice(indices, size=centroid_samples, replace=False, p=probs[i, :]))
            labels.append([i]*centroid_samples)
        centroids, labels = np.array(centroids), np.array(labels)
    return centroids, labels
def poisson_cluster_kmeans(W, L, k, centroid_method='dijkstra', centroid_inits=None, num_iter=10, num_starts=1, centroid_samples=None, show_plot=False, y=None):
    #Randomly choose k labeled points
    n = W.shape[0]
    L = gl.graph_laplacian(W)
    indices = np.arange(n)
    run_metrics = []
    y_preds = []
    centroid_labels = np.arange(k)
    if not centroid_inits:
        centroid_inits = [np.random.choice(indices,size=k,replace=False) for _ in range(num_starts)]
    for centroids in centroid_inits:
        for _ in range(num_iter):
            #Semi-supervised learning 
            y_pred = gl.graph_ssl(W, centroids, centroid_labels,method='poisson')
            centroids, centroid_labels = update_centroids(W, L, y_pred, indices, n, k, centroid_method)
        y_preds.append(y_pred)
        run_metrics.append(graph_cut(L, label_matrix(y_pred)))
    best_index = np.argmin(run_metrics)
    return y_preds[best_index]       
# Bottom up approach.
def withinss(x, n):
    x = np.sort(x)
    v = np.zeros(n-1,)
    for i in range(n-1):
        x1 = x[:(i+1)]
        x2 = x[(i+1):]
        m1 = np.mean(x1)
        m2 = np.mean(x2)
        v[i] = (np.sum((x1-m1)**2) + np.sum((x2-m2)**2))
    ind = np.argmin(v)
    threshold = x[ind] # Clustering threshold
    loss = v[ind] # Clustering loss function (goodness of fit)
    return threshold, loss     
def find_split(L, nodes, trials=1):
    nodes = np.asarray(nodes)
    n = nodes.shape[0]
    sources = np.random.choice(nodes, replace=False, size=trials)
    best_loss = np.Inf
    best_threshold = None
    best_u = None
    for i in range(trials):
        f = np.asarray([-1.0/n if j != sources[i] else 1.0 - 1.0/n for j in range(n)])
        # Solve the Poisson equation
        u = gl.pcg_solve(L,f,x0=None,tol=1e-10)
        assert np.sum(f) < 1e-9 # Ensure u is mean 0
        threshold, loss = withinss(u, n)
        if loss < best_loss:
            best_loss = loss
            best_threshold = threshold
            best_u = u
    nodes_left = np.asarray([nodes[i] for i in range(n) if best_u[i] <= best_threshold])
    nodes_right = np.asarray([nodes[i] for i in range(n) if best_u[i] > best_threshold])
    return nodes_left, nodes_right, best_loss     
# Bottom-up hierarchical clustering approach. Each point starts in its own cluster, and clusters are progressively merged to minimize the withinSS criterion.
def poisson_cluster_bottom(W, L, n_clusters=2, split_trials = 10):
    n = W.shape[0]
    nodes = np.arange(n)
    losses = {}
    splits = {} # Can make more efficient if we make splits a Binary tree?
    clusters = [nodes] # List of clusters we will consider splitting
    one_pt_clusters = []
    k = 1
    y = np.zeros(n)
    while k < n_clusters:
        if clusters == []:
            raise Exception('There are insufficient data points to obtain the desired number of clusters')
        chosen_cluster = None
        best_loss = np.Inf
        for cluster in clusters:
            if tuple(cluster) not in losses:
                L_cluster = L[cluster, :]
                L_cluster = L_cluster[:, cluster]
                assert L_cluster.shape[0] == L_cluster.shape[1]
                left_cluster, right_cluster, loss = find_split(L_cluster, cluster, trials=split_trials)
                losses[tuple(cluster)] = loss
                splits[tuple(cluster)] = left_cluster, right_cluster
            if losses[tuple(cluster)] < best_loss:
                best_loss = losses[tuple(cluster)]
                chosen_cluster = cluster
        left_split, right_split = splits[tuple(chosen_cluster)]
        clusters.remove(chosen_cluster)
        for item in [left_split, right_split]:
            if len(item) > 1:
                clusters.append(item)
            else:
                one_pt_clusters.append(item)
        k += 1
    for i, cluster in enumerate(one_pt_clusters+clusters):
        y[cluster] = i
    return y

# Clustering purity measures (unsupervised)

# sklearn.metrics.silhouette_score(X, labels, *, metric='euclidean', sample_size=None, random_state=None, **kwds)
# X is the matrix of pairwise distances

# Supervised clustering metrics
# sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)
# sklearn.metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
# sklearn.metrics.normalized_mutual_info_score(labels_true, labels_pred, *, average_method='arithmetic')
# gl.clustering_accuracy(y_true,y_pred)

# Return the graph-cut energy. U is an nxk matrix whose rows are the one-hot encoded cluster labels, and L is the graph Laplacian.
def graph_cut(L, U):
    k = U.shape[1]
    assert type(U) == np.ndarray
    assert type(L) == scipy.sparse.csr.csr_matrix
    return np.sum([U[:, i].transpose()*L*U[:, i] for i in range(k)])

# Threshold the matrix U. The label decision is l(x_i) = argmax_j s_j^{-1} u(x_i)_j
def threshold(U, class_sizes):
    D = scipy.sparse.diags(class_sizes)
    U_scaled = D * scipy.sparse.csr_matrix(u)
    max_locations = np.argmax(U_scaled.todense(),axis=0)
    max_locations = np.squeeze(np.asarray(max_locations)) # Convert from matrix of shape (n, 1) to array
    k = u.shape[0]
    I = np.eye(k,n)
    U_threshold = I[:, max_locations]
    return U_threshold
def merge_matrix(k, i, j):
    assert i < j
    A = np.eye(k, k)
    A[:, j] = A[:, i] # Merge jth cluster into ith cluster, leaving row j empty
    A = np.delete(A, j, axis=0)
    return A
# Top-down approach for clustering. We begin with one cluster containing all points, and clusters progressively split such that a criterion (graph cut, withinSS) is minimized.
def poisson_cluster_top(W, n_clusters=2, use_cuda=False):
    n = W.shape[0]
    N = int(0.1*n)
    L = gl.graph_laplacian(W)
    L = L.tolil()
    class_sizes = np.ones(N)
    degrees = W.sum(axis=1)
    # Select N > N_c random vertices
    indices = np.random.choice(range(n), replace=False, size=N)
    # Initialize labels
    F = np.zeros((n,n))
    F[indices, range(len(indices))] = 1
    F = scipy.sparse.csr_matrix(F)
    # Poisson Learning
    u, _ = gl.poisson(W,indices,range(N)) # Return k x n array of unthresholded poisson values
    assert (u*degrees > 1.0e-07).sum() == 0 # Ensure the sum of d_i u(x_i) is 0
    # Initialize Cluster labeling
    U = threshold(u, class_sizes)
    k = N # current number of clusters
    cluster_min = 1
    print(f'U.shape = {U.shape}')
    print(f'L.shape = {U.shape}')
    while k > n_clusters:
        C = np.diag([np.Inf]*k)
        for i in range(k):
            for j in range(i+1,k):
                U_ij = merge_matrix(k,i,j).dot(U)
                print(f'shape of U_ij is {U_ij.shape}')
                C[i,j] = graph_cut(L, threshold(U_ij, class_sizes))
        i, j = numpy.unravel_index(C.argmin(), C.shape) # Merge clusters i and j
        # Update the label matrix U
        U = merge_matrix(k,i,j).dot(U)
        k -= 1
        break
    return U

### Experiments ###
def experiment(X, y, W, L, centroid_methods = ['dijkstra', 'poisson_solve', 'laplacian_eigenvector'], centroid_inits=None, show_plot=False, num_starts=3, centroid_samples=None):
    k=len(np.unique(y))
    n = len(X)
    indices = np.arange(n)
    if not centroid_inits:
        centroid_inits = [np.random.choice(indices,size=k,replace=False) for _ in range(num_starts)]
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y).set(title='Ground Truth')
    plt.show()
    # Poisson k-means
    try:
        y_kmeans = {}
        for method in centroid_methods:
            y_kmeans[method] = poisson_cluster_kmeans(W, L, k, centroid_method=method, num_starts=num_starts, centroid_inits=centroid_inits, centroid_samples=centroid_samples)
    except:
        print('Encountered an error in running poisson k-means')
        print(f'centroid inits: {centroid_inits}')
        raise Exception('Poisson k-means encountered an error')
    #Spectral clustering
    y_spec = gl.spectral_cluster(W,k)
    # Generate plots
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_spec).set(title='Spectral Clustering')
    plt.show()
    for method in centroid_methods:
        g = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_kmeans[method])
        g.set(title = f'Poisson Clustering ({method})')
        plt.show()
    metric_names = ['Accuracy', 'Adjusted Rand Score (0-1)', 'Normalized Mutual Information (0-1)']
    metrics = [gl.clustering_accuracy, sklearn.metrics.adjusted_rand_score, sklearn.metrics.normalized_mutual_info_score]
    for metric, name in zip(metrics, metric_names):
        print(f"\n ### {name} ### \n")
        print(f"Spectral Clustering {name}={metric(y, y_spec)}")
        for method, y_pred in y_kmeans.items():
            print(f"Poisson Clustering ({method}) {name}={metric(y, y_pred)}")

if __name__ == "__main__":
    X, y = datasets.load_wine(return_X_y=True)
    indices = np.arange(X.shape[0])
    # Standardize and apply PCA as a preprocessing step
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)
    pca = PCA(n_components=X.shape[1])
    X = pca.fit_transform(X_scale)
    W = build_graph(X)
    L = gl.graph_laplacian(W)
    #poisson_cluster_bottom(W, L, len(np.unique(y)))