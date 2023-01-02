
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
import pylab as plt
import networkx as nx
import pandas as pd
from datetime import datetime, date, time
import cvxpy as cp
from scipy.optimize import minimize
from numpy.linalg import norm, multi_dot
from sklearn.metrics import normalized_mutual_info_score
import copy
import scipy.io
import time
import sys, os
import itertools
from collections import deque

from utils import informative_layers, create_graph, Degree_Matrix, Normalized_spectral_clustering

alpha = 10
beta =100

def objective_function(L, P, Q, Lambda, alpha, beta):
  """
    This function calculates the objective function of the optimization problem to solve. 
    
    Arguments:
        L : a list containing the Laplacians of each layer.
        P : the joint eigenvectors matrix of the graph.
        Q : the inverse of P.
        alpha, beta : some regularisation parameters.
    
    Returns: the value of the objective function. 
  """
  sum = 0
  M = len(L)
  n = L[0].shape[0]
  P = P.reshape(n,n)
  Q = Q.reshape(n,n)
  for m in range(M):
    sum += norm(L[m] - np.dot(np.dot(P,Lambda[m,:,:]), Q))**2
  return 0.5*sum + (alpha/2)*(norm(P)**2 + norm(Q)**2) + (beta/2)*norm(np.dot(P,Q)-np.eye(n))**2



def solve_optimization(Laplacians,Degrees,M,n,f,N_iter):
  """
    This function enables to optimize the matrix P and Q with the L-BFGS-B method.
    P represents the joint eigenvectors of the graph and Q its inverse.
    
    Arguments:
        Laplacians : weighted adjancency matrix of sizen n x n.
        M : the number of layers
        n : the number of vertices per layer
        f : the indices of the most informative layer in the multi-layer graph.
        N_iter : the number of iterations to perform in the optimization loop.
    
    Returns: the optimal value of the matrix P and Q. 
  """
  #matrices diagonales des valeurs propres
  Lambda=np.ones((M,n,n))
  w, P =np.linalg.eig(Laplacians[f])
  Q = np.linalg.inv(P)

  for m in range(M):
    w,v=np.linalg.eig(Laplacians[m])
    w=np.sort(w)
    Lambda[m, :, :]=np.diag(w)

  res_P = [objective_function(Laplacians, P, Q, Lambda, alpha, beta)]
  res_Q = [objective_function(Laplacians, P, Q, Lambda, alpha, beta)]
  for i in range(N_iter):
    if i%10==0:
      print(i)
    function_P=lambda P: objective_function(Laplacians, P, Q, Lambda, alpha, beta)
    sol_P=minimize(function_P,P,method='L-BFGS-B')
    P=sol_P.x
    res_P.append(objective_function(Laplacians, P, Q, Lambda, alpha, beta))
    function_Q=lambda Q: objective_function(Laplacians, P, Q, Lambda, alpha, beta)
    sol_Q=minimize(function_Q,Q,method='L-BFGS-B')
    Q=sol_Q.x
    res_Q.append(objective_function(Laplacians, P, Q, Lambda, alpha, beta))
  return P,Q, res_P, res_Q


def SC_GED(W,k,N_iter):
  """
    This function performs clustering on multi-layer graphs with the 
    Clustering with generalized eigen-decomposition method. 
    
    Arguments:
        W : weighted adjancency matrix of sizen n x n.
        k : the target number of clusters.
        N_iter : the number of iterations in the optimlization loop.
    
    Returns: the predictions of the clustering,
    i.e. the cluster for each vertices. 
  """
  # we compute the informative rank in order to initialize the optimization loop with the most informative W.
  rank = informative_layers(W, k)
  M, n = W.shape[0], W.shape[1]
  #Compute the degree matrix for each graph
  Degrees=[]
  for m in range(M):
    V=W[m,:,:]
    D= Degree_Matrix(W[m,:,:])
    Degrees.append(D)
  #Compute the random walk Laplacian for each graph
  Laplacians=[]
  for m in range(M):
    Laplacians.append(np.dot(np.linalg.pinv(Degrees[m]),Degrees[m]-W[m,:,:]))
  #Solve the optimization problem
  P,Q, res_P, res_Q =solve_optimization(Laplacians,Degrees,M,n,rank[0], N_iter)
  P = P.reshape(n,n)
  U_prime=P[:,0:k]
  kmeans = KMeans(n_clusters=k, random_state=0).fit(U_prime)
  return kmeans.labels_, res_P, res_Q


def SC_SR(W,k,lambd):
  """
  This function performs clustering on multi-layer graphs with the 
  Clustering with spectral regularization method. 
    
  Arguments:
      W : weighted adjancency matrix of sizen n x n.
      k : the target number of clusters.
      lambd : a regularization parameter.
    
  Returns: the predictions of the clustering,
      i.e. the cluster for each vertices. 
  """
  M, n = W.shape[0], W.shape[1]
  rank = informative_layers(W, k)
  W_1 = W[rank[0],:,:]
  
  #Etape 2 : Compute the degree matrix for the Graph 1
  D = Degree_Matrix(W_1)
  
  # Etape 3 : Compute the random walk Laplacian for the Graph 1
  L = np.dot(np.linalg.pinv(D), D - W_1)
  
  # Etape 4 : first k eigen vectors of the Laplacian matrix
  w,v=np.linalg.eig(L)
  w_arg=np.argsort(w)
    #we take the k smallest values 
  wk_arg = w_arg[:k]
  
  # Etape 5 : U matrix for the k first eigenvectors
  U = np.zeros((n,k))
  for i, j in enumerate(wk_arg):
    U[:,i] = v[:,j]
  
  # Etape 6 : Solve the spectral regularization problem
  for m in rank[1:]:
    W_ = W[m,:,:]
    D = Degree_Matrix(W_)
    L_sym  = np.dot(np.dot(np.linalg.pinv(D**(1/2)), (D-W_)), np.linalg.pinv(D**(1/2)))
    lam = lambd[m-1]
    mu = 1/lam
    for i in range(k):
      f_i = mu * np.dot(np.linalg.inv(L_sym + mu*np.eye(n)), U[:,i])
      U[:,i] = f_i
  
  kmeans = KMeans(n_clusters=k, random_state=0).fit(U)
  return kmeans.labels_


def K_Kmeans(W, k, d):
  """
  This function performs clustering on multi-layer graphs with the 
  Kernel K-means method
    
  Arguments:
      W : weighted adjancency matrix of sizen n x n.
      k : the target number of clusters.
      lambd : a regularization parameter.
      d : number of eigen vectors choosen in the Laplacian matrix to build 
      the Kernels. Must have d << n and d inferior or equal to k. 
    
  Returns: the predictions of the clustering,
      i.e. the cluster for each vertices. 
  """
  M, n = W.shape[0], W.shape[1]
  D_matrix = np.zeros((M,n,n)) # array to store the degree matrix
  L_matrix = np.zeros((M,n,n)) # array to store the Laplacians
  Kernel_matrix = np.zeros((M,n,n))
  sum_K = 0
  for m in range(M):
    D = Degree_Matrix(W[m,:,:])
    L_sym = np.dot(np.dot(np.linalg.pinv(D**(1/2)), (D-W[m,:,:])), np.linalg.pinv(D**(1/2)))
    w,v=np.linalg.eig(L_sym)
    w_arg=np.argsort(w)
    #we take the k smallest values 
    wk_arg = w_arg[:k]
    K_m = 0
    for i in range(d):
      j = wk_arg[i]
      u = v[:,i].reshape(n,1)
      u = np.asarray(u, dtype = np.float64)
      K_m += np.dot(u, u.T)
    sum_K += K_m 
  kmeans = KMeans(n_clusters=k, random_state=0).fit(sum_K)
  return kmeans.labels_