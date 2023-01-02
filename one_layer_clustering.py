import numpy as np
from utils.py import create_graph, Degree_Matrix
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



def Normalized_spectral_clustering(W,k):
  """
    This function enables to 
    
    Arguments:
        W : weighted adjancency matrix of sizen n x n.
        k : target number of clusters 
    
    Returns: the prediction of the clustering, 
    i.e. an array with the label of each cluster for each vertices.
  """
  n= W.shape[0]
  #compute the degree matrix
  D=Degree_Matrix(W)
  #compute the random walk graph Laplacian
  L=np.dot(np.linalg.pinv(D),D-W)
  #w:eigenvalues, v:eigenvectors of L
  w,v=np.linalg.eig(L)
  w_arg=np.argsort(w)
  #we take the k smallest values 

  wk_arg = w_arg[:k]
  #U matrix for the k first eigenvectors
  U = np.zeros((n,k))
  for i, j in enumerate(wk_arg):
    U[:,i] = v[:,j]
  kmeans = KMeans(n_clusters=k, random_state=0).fit(U)
  return kmeans.labels_ 