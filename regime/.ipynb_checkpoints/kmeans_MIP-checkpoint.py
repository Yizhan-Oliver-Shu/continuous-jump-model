import numpy as np
import pandas as pd
# gurobi import
import gurobipy as gp
from gurobipy import *


# import the following helpers for any clustering related tasks
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import _euclidean_distances
from itertools import product


def compute_big_M(X):
    """
    for each data point, its big M is the longest squared distance to any other data point.
    """
    return euclidean_distances(X, squared=True).max(axis=1)

class KMeans_direct_MIP():
    def __init__(self, n_clusters, timeLimit = None):
        """
        k-means algo fitted by MIP.
        """
        self.n_clusters = n_clusters
        self.timeLimit = timeLimit
        
        
    def fit(self, X):
        """
        
        """
        n_clusters = self.n_clusters
        n_samples, n_features = X.shape
        M = compute_big_M(X)
        
        model = Model("k-means-MIP")
        assign = model.addMVar((n_samples, n_clusters), vtype=GRB.BINARY, name="assign")
        
        constrs = []
        constrs.append(
            model.addConstr(assign.sum(axis=1)==1, name="unique assignemnt")
        ) 
        
        centers = model.addMVar((n_clusters, n_features), lb=-GRB.INFINITY, name="centers")
        ub_dist = model.addMVar(n_samples, obj=1., name="upper bound on sq distance")      
        constrs.append(
            model.addConstrs(((X[t] - centers[k]) @ (X[t] - centers[k]) <= ub_dist[t] + M[t] * (1-assign[t, k]) \
                              for t, k in product(range(n_samples), range(n_clusters))), name="quad constrs")
        )
        
        self.model = model
        model.update()
        model.optimize()
        
        if model.Status == GRB.OPTIMAL:
            labels_ = assign.X.argmax(axis=1)
            centers_ = centers.X
            inertia_ = model.ObjVal
            runtime = model.Runtime
        self.labels_, self.centers_, self.inertia_, self.runtime = labels_, centers_, inertia_, runtime    
        return self



###############
## archive
###############

class KMeans_gurobi_global():
    def __init__(self, n_clusters=8, random_state=None, timeLimit = None):
        """
        Use gurobi to find the global solution to k-means problem through its MIP formulation. APIs emulate those from `sklearn.cluster.KMeans`.
        """
        self.n_clusters = n_clusters
        self.gurobi_model = None
        self.timeLimit = timeLimit
        # no randomness is involved here.

    def _compute_big_M(self, X):
        """
        compute big-M, to be used in the MIP formulation
        """
        return _euclidean_distances(X, X, squared=True).max(axis=1)
   
    def _extract_optimal_solution_as_array(self, x):
        """
        for a dictionary of gurobi (scalar) variables, extract its values, and concat into an array.
        """
        return np.array(list(map(lambda s:s.X, x.values())))
    
    def _extract_optimal_labels_(self, assign):
        return self._extract_optimal_solution_as_array(assign).reshape((self.n_samples, self.n_clusters)).argmax(axis=1)
        
    def _extract_optimal_cluster_centers_(self, means):
        return self._extract_optimal_solution_as_array(means).reshape((self.n_clusters, self.n_features))
    
    def fit(self, X, y=None):
        # model
        self.gurobi_model = gp.Model(name = 'kmeans')
        m = self.gurobi_model
        if np.isscalar(self.timeLimit):
            m.Params.timelimit = self.timeLimit
        # shape
        self.n_samples, self.n_features = X.shape
        # big-M 
        M = self._compute_big_M(X) #* 2

        # index set
        datas = tuplelist(range(self.n_samples))
        clusters = tuplelist(range(self.n_clusters))
        dimensions = tuplelist(range(self.n_features))

        datas_prod_clusters = tuplelist(product(datas, clusters))
        clusters_prod_dimensions = tuplelist(product(clusters, dimensions))        

        # decision vars
        means = m.addVars(clusters_prod_dimensions, lb=-GRB.INFINITY, name = "cluster centers")
        assign = m.addVars(datas_prod_clusters, vtype = GRB.BINARY, name = "assignment")
        dist_sq = m.addVars(datas, name = 'min_dist_sq')     
    
        # obj
        m.setObjective(quicksum(dist_sq), GRB.MINIMIZE)
        
        # assignment constr
        assign_constrs = m.addConstrs((quicksum(assign[data, cluster] for cluster in clusters)==1 for data in datas))

        # quad constr
        quad_constrs = {}
        for data, cluster in datas_prod_clusters:
            quad_constrs[data, cluster] = m.addQConstr(
                quicksum((X[data, dim]-means[cluster, dim])**2 for dim in dimensions) <= dist_sq[data] + M[data] * (1-assign[data, cluster])
            )
        # optimize!
        m.optimize()

        # if m.Status != GRB.OPTIMAL:
        #     raise Exception("optimal solution found is not found!")
            
        # save results
        # runtime
        self.Runtime = m.Runtime
        # optimal value
        self.inertia_ = m.ObjVal
        # optimal solution
        self.labels_ = self._extract_optimal_labels_(assign)
        self.cluster_centers_ = self._extract_optimal_cluster_centers_(means)
        #dist_sq_opt = extract_optimal_solution_as_array(dist_sq)   
        return self