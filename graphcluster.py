'''
Graph Clustering methods
Implemented by Brendan Cook
'''

import numpy as np
import GraphLearning.graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.cluster import KMeans
import seaborn as sns
import kmeans1d
import scipy

# To-do: Finish poisson_topdown
# Test poisson_kmeans

# Test Datasets
def gaussian_mixture():
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
def moons_dataset(n=500, noise=0.1):
    X, L = datasets.make_moons(n_samples=n, noise=noise)
    return X, L
def wines_dataset():
    X, L = load_wine(return_X_y=True)
    return X, L
def build_graph(X, k=10):
    #Build graph
    I,J,D = gl.knnsearch(X,k)
    W = gl.weight_matrix(I,J,D,k)
    W = gl.diag_multiply(W,0)
    return W  
def poisson_cluster_kmeans(X, W, labels, k, centroid_method='closest', max_iter=20, num_starts=1, show_plot=True):
    #Randomly choose k labeled points
    n = W.shape[0]
    L = gl.graph_laplacian(W)
    indices = np.arange(n)
    centroids = np.random.choice(range(n),size=k,replace=False)
    if show_plot: plt.ion()
    last_acc = 100 
    for i in range(max_iter):
        #Semi-supervised learning 
        l = gl.graph_ssl(W,centroids,range(k),method='poisson')
        #Compute accuracy
        acc = gl.clustering_accuracy(l,labels)   
        if show_plot:
            print("Accuracy=%f"%acc)
            #Plot result (red points are labels)
            scat1=plt.scatter(X[:,0],X[:,1], c=l)
            scat2=plt.scatter(X[centroids,0],X[centroids,1], c='r')
            plt.pause(0.1) #This makes matplotlib draw to the window within the loop, instead of at end
            scat1.remove()
            scat2.remove()

        #Update means
        if centroid_method == 'closest': 
            m = [np.mean(X[l==j, :] ,axis=0) for j in range(k)]
            centroids = [np.argmin(np.linalg.norm(X - m[j],axis=1)) for j in range(k)]
        elif centroid_method == 'laplacian_eigenvector':
            eigenvectors = np.vstack((gl.dirichlet_eigenvectors(L,indices[l != i],1)[0] for i in range(k)))
            centroids = np.argmax(eigenvectors, axis=1)
        elif centroid_method == 'poisson_solve':
            centroids = [np.argmax(np.abs(gl.constrained_solve(L,indices[l != j],0, np.ones(n)))) for j in range(k)]
        elif centroid_method == 'dijkstra':
            centroids = [np.argmax(gl.cDijkstra(L,indices[l != j],0)) for j in range(k)]
    if show_plot:
            print("Accuracy=%f"%acc)
            #Plot result (red points are labels)
            scat1=plt.scatter(X[:,0],X[:,1], c=l)
            scat2=plt.scatter(X[centroids,0],X[centroids,1], c='r')            
    return l
# Bottom up approach.
def withinss(x):
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
def withinss_cluster(x, n_clusters):
    n = len(x)
    x = np.sort(x)
    k = n_clusters
    D = np.zeros((n+1, k+1))
    B = np.zeros((n, k))
    mu = np.zeros(n)
    mu[0] = x[0]
    withinss_vals = np.zeros(n)  # withinss_vals[i] = withinss([x_1, ... , x_i])
    for i in range(1,n):
        mu[i] = (x[i] + (i-1)*mu[i-1])/i
        withinss_vals[i] = withinss_vals[i-1] + ((i-1)/i)* (x[i] - mu[i-1])**2
    for i in range(1,n+1):
        
        withinss_vals
        for m in range(1, k+1):
            vals = [D[j-1, m-1] + withinss_vals[j,i] for j in range(m,i+1)]
            j = np.argmin(vals)
        
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
        threshold, loss = withinss(u)
        if loss < best_loss:
            best_loss = loss
            best_threshold = threshold
            best_u = u
    nodes_left = np.asarray([node for node in nodes if best_u[node] <= best_threshold])
    nodes_right = np.asarray([node for node in nodes if best_u[node] > best_threshold])
    return nodes_left, nodes_right, best_loss     

def poisson_cluster_bottom(W, n_clusters=2, split_trials = 100):
    n = W.shape[0]
    L = gl.graph_laplacian(W)
    L = L.tolil()
    nodes = range(n)
    losses = {}
    splits = {} # Can make more efficient if we make splits a Binary tree?
    clusters = [nodes] # List of clusters we will consider splitting
    one_pt_clusters = []
    k = 1
    while k < n_clusters:
        if not clusters:
            raise Exception('There are insufficient data points to obtain the desired number of clusters')
        chosen_cluster = None
        best_loss = np.Inf
        for cluster in clusters:
            if cluster not in losses:
                L_cluster = L[cluster, :]
                L_cluster = L[:, cluster]
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
    return one_pt_clusters+clusters, splits, losses

# Return the graph-cut energy. U is an nxk matrix whose rows are the one-hot encoded labels, and L is the graph Laplacian.
def graph_cut(L, U):
    k = U.shape[1]
    assert type(L) == scipy.sparse.csr.csr_matrix
    assert type(U) == scipy.sparse.csr.csr_matrix
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

if __name__ == "__main__":
    n = 500
    nodes = range(n)
    
    L = gl.graph_laplacian(W)
    N = 3
    indices = np.random.choice(nodes, replace=False, size=N)
    u, _ = gl.poisson(W,indices,range(N))
    degrees = W.sum(axis=1)
    class_sizes = np.diag([1,1,1])
    #clusters, splits, losses = poisson_cluster_bottom(W, 2, split_trials=1)
    #left, right, loss = find_split(L, nodes)
    poisson_cluster_top(W, 2)