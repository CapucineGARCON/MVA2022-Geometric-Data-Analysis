import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, Normalized_spectral_clustering
from sklearn.metrics import normalized_mutual_info_score



def create_graph(W):
    """
    This function enables to display a one layer graph from its weighted adjacency matrix.
    
    Arguments:
        W : weighted adjacency matrix of size  n x n.  
    
    Returns: an image of the graph.
    """
    # Create Graph from A
    G = nx.from_numpy_matrix(W, create_using=nx.Graph)
    # Use spring_layout to handle positioning of graph
    layout = nx.spring_layout(G)
    # Draw the graph using the layout - with_labels=True if you want node labels.
    nx.draw(G, layout, with_labels=True)
    # Get weights of each edge and assign to labels
    labels = nx.get_edge_attributes(G, "weight")
    # Draw edge labels using layout and list of labels
    nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=labels)
    # Show plot
    plt.show()
  
  
def Degree_Matrix(W):
  """
    This function enables to extract the degree matrix from the weighted 
    adjacency matrix W of a graph with n vertices.
    
    Arguments:
        W : weighted adjacency matrix of size  n x n.  
    
    Returns: a matrix that is the same size as W. 
  """
  n = W.shape[0]
  D=[]
  for i in range(n):
    d=0
    for j in range(n):
      if W[i,j]>0:
        d+=1
    D.append(d)
  D=np.diag(D)
  return D


def informative_layers(W, k):
  """
    This function enables to know the impotance of layers in multi-graphs. 
    
    Arguments:
        W : weighted adjancency matrix of sizen n x n.
        k : the target number of clusters.
    
    Returns: a ranking list of the layers by indexes. 
  """
  M = W.shape[0]
  n = W.shape[1]
  C = np.zeros((M,n))
  NMI = np.zeros((M,M))
  means = []
  rank = []
  for m in range(M):
    C[m,:] = Normalized_spectral_clustering(W[m,:,:], k)
  for i in range(M):
    for j in range(i+1,M):
      NMI[i,j] = normalized_mutual_info_score(C[i,:], C[j,:])
      NMI[j,i] = normalized_mutual_info_score(C[i,:], C[j,:])
  means = [np.mean(NMI[i,:]) for i in range(M)]
  print(means)
  most_informative = np.argmax(means)
  L = list(NMI[most_informative,:])
  LL = zip(L, np.arange(0,M))
  LL = sorted(LL)
  indexes = [y for (x,y) in LL][::-1]
  d = indexes.index(most_informative)
  del indexes[d]
  rank = [most_informative] + indexes
  return rank
  
  
def Purity(Omega,C,k,N):
  """
  This function evaluates a multi-layer graphs clustering.  
    
  Arguments:
      Omega : clusters computed.
      C : ground truth clusters.
      k : the target number of clusters.
      N : number of nodes.
    
  Returns: returns a float between 0 and 1.  
  """
  #intersections is the matrix |Omega_i inter C_j|
  intersections=np.zeros((k,k))
  for i in range(k):
    for j in range(k):
      intersections[i,j]=np.count_nonzero([Omega[i][m] in C[j] for m in range(len(Omega[i]))])
  #compute the max of intersections for each cluster computed
  maximums=np.zeros(k)
  for i in range(k):
    maximums[i]=np.max(intersections[i,:])
  return (1/N)*np.sum(maximums)
