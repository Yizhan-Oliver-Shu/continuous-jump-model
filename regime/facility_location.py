import numpy as np
from gurobipy import *
from itertools import product
from .gurobi_utils import *

def extract_labels_from_assign_arr(assign_arr):
    """
    extract the labeling and assignments, from a clustering results.
    """
    labels_ = assign_arr.argmax(axis=1)
    indices = np.unique(labels_)
    for i, ind in enumerate(indices):
        labels_[labels_==ind] = i  
    return indices, labels_ 


class facility_location(object):
    """
    base class for the facility location problem.
    """
    def fit(self, dist_mx, n_locations = 2):
        # initiate the model
        self.model = Model("facility-location")
        # save parameters
        self.n_samples, self.n_candidates = dist_mx.shape
        self.n_locations = n_locations
        
class facility_location_matrix(facility_location):
    """
    solve the plain facility location problem, in matrix-form in gubori.
    """
    def fit(self, dist_mx, n_locations = 2):
        # self.model = Model("facility-matrix")
        # self.n_samples, self.n_candidates = dist_mx.shape
        # self.n_locations = n_locations
        super().fit(dist_mx, n_locations)
        model, n_samples, n_candidates = self.model, self.n_samples, self.n_candidates

        period_to_candidate = model.addMVar((n_samples, n_candidates), obj = dist_mx, vtype=GRB.BINARY, name="period_to_candidate")
        select = model.addMVar(n_candidates, vtype=GRB.BINARY, name = "select")

        constrs = []

        constrs.append(
            model.addConstr(period_to_candidate.sum(axis=1) == 1, name = "assign to one candidate")
        )
        
        constrs.append(
            model.addConstr(select.sum() == n_locations, name="select K candidates")
        )

        constrs.append(
            model.addConstrs((period_to_candidate[:, j] <= select[j]  for j in range(n_candidates)), name = "active")
        )
        
        model.update()
        model.optimize()  
        
        if model.Status == GRB.OPTIMAL:
            indices, labels_ = extract_labels_from_assign_arr(period_to_candidate.X)
        else:
            indices, labels_ = None, None
            
        self.indices, self.labels_ = indices, labels_
        return self
    
def inverse_dict(candidate_dict, n_candidates):
    arr = -np.ones(n_candidates, dtype=int)
    for regime in candidate_dict:
        arr[candidate_dict[regime]] = regime
    return arr

class facility_location_regime(facility_location):
    """
    solve the plain facility location problem, in matrix-form in gubori.
    """
    def fit(self, dist_mx, n_locations = 2, candidate_dict = None):
        # self.model = Model("facility-matrix")
        # self.n_samples, self.n_candidates = dist_mx.shape
        # self.n_locations = n_locations
        super().fit(dist_mx, n_locations)
        model, n_samples, n_candidates = self.model, self.n_samples, self.n_candidates
        regime_list = inverse_dict(candidate_dict, n_candidates)

        period_to_regime = model.addMVar((n_samples, n_locations), vtype=GRB.BINARY, name="period_to_regime")
        period_to_candidate = model.addMVar((n_samples, n_candidates), obj = dist_mx, vtype=GRB.BINARY, name="period_to_candidate")
        select = model.addMVar(n_candidates, vtype=GRB.BINARY, name = "select")

        constrs = []
        
        constrs.append(
            model.addConstr(period_to_regime.sum(axis=1) == 1, name = "assign to one regime")
        )
        
        constrs.append(
            model.addConstr(period_to_candidate.sum(axis=1) == 1, name = "assign to one candidate")
        )
        
        constrs.append(
            model.addConstrs((select[candidate_dict[k]].sum() == 1 for k in range(n_locations)), name="select a unique candidate for each regime")
        )

        constrs.append(
            model.addConstrs((period_to_candidate[:, j] <= select[j]  for j in range(n_candidates)), name = "active1")
        )
        
        constrs.append(
            model.addConstrs((period_to_candidate[:, j] <= period_to_regime[:, regime_list[j]]  for j in range(n_candidates)), name = "active2")
        )        
        
        model.update()
        model.optimize()  
        
        if model.Status == GRB.OPTIMAL:
            labels_ = period_to_regime.X.argmax(axis=1)
            indices_unordered = np.where(select.X)[0]
            indices = -np.ones(n_locations, dtype=int)
            for index in indices_unordered:
                indices[regime_list[index]] = index
        else:
            indices, labels_ = None, None
            
        self.indices, self.labels_ = indices, labels_
        return self
    

    
class facility_location_jump_penalty(facility_location):
    """
    solve the original facility location problem, in matrix-form in gubori.
    """
    def fit(self, dist_mx, n_locations = 2, lambd = .005, min_capacity = .0):
        # self.model = Model("facility-matrix")
        # self.n_samples, self.n_candidates = dist_mx.shape
        # self.n_locations = n_locations
        super().fit(dist_mx, n_locations)
        self.lambd = lambd
        model, n_samples, n_candidates = self.model, self.n_samples, self.n_candidates

        period_to_candidate = model.addMVar((n_samples, n_candidates), obj = dist_mx, vtype=GRB.BINARY, name="period_to_candidate")
        select = model.addMVar(n_candidates, obj = 0., vtype=GRB.BINARY, name = "select")
        abs_ub = model.addMVar((n_samples-1, n_candidates), vtype=GRB.BINARY, obj = lambd, name="abs_ub")

        constrs = []

        constrs.append(
            model.addConstr(period_to_candidate.sum(axis=1) == 1, name = "assign to one candidate")
        )
        
        constrs.append(
            model.addConstr(select.sum() == n_locations, name="select K candidates")
        )

        constrs.append(
            model.addConstrs((period_to_candidate[:, j] <= select[j]  for j in range(n_candidates)), name = "active") # <= 0
        )
        
        constrs.append(
            model.addConstr(period_to_candidate[:-1] - period_to_candidate[1:] <= abs_ub, name = "ub_1") 
        )
        
        constrs.append(
            model.addConstr(period_to_candidate[1:] - period_to_candidate[:-1] <= abs_ub, name = "ub_2") 
        )
        
        # if min_capacity > 0:
        #     constrs.append(
        #         model.addConstr(period_to_candidate.sum(axis=0) >= int(min_capacity * n_samples), name = "min_capacity") 
        #     )            
        
        # constrs.append(
        #     model.addConstrs((period_to_candidate[:, j] - select[j] <= 0 for j in range(n_candidates)), name = "active") # <= 0
        # )        
        # constrs.append(
        #     model.addConstr(period_to_candidate - select <= 0, name = "active")
        # )

        model.update()
        model.optimize()  
        
        if model.Status == GRB.OPTIMAL:
            indices, labels_ = extract_labels_from_assign_arr(period_to_candidate.X)
        else:
            indices, labels_ = None, None
            
        self.indices, self.labels_ = indices, labels_
        return self    

    
    
    
    
    
    
    
def process_impossible_indices(impossible_candidates):
    if isinstance(impossible_candidates, dict):
        return {key: list(impossible_candidates[key]) for key in impossible_candidates}
    return None

class facility_location_explicit_regime(facility_location):        
    def fit(self, dist_mx, n_locations = 2, impossible_candidates=None):
        super().fit(dist_mx, n_locations)
        # self.model = Model("facility-matrix")
        # self.n_samples, self.n_candidates = dist_mx.shape
        # self.n_locations = n_locations
        
        period_to_regime = self.model.addMVar((self.n_samples, self.n_locations), vtype=GRB.BINARY, name = "period_to_regime")
        regime_to_candidate = self.model.addMVar((self.n_locations, self.n_candidates), vtype=GRB.BINARY, name = "regime_to_candidate")
        
        dist_mx_repeat = np.repeat(dist_mx[:, np.newaxis, :], self.n_locations, axis=1)
        period_to_regime_to_candidate = self.model.addMVar((self.n_samples, self.n_locations, self.n_candidates), 
                                                           obj = dist_mx_repeat,
                                                           vtype = GRB.BINARY, 
                                                           name = "period_to_regime_to_candidate")

        constraints = []
        constraints.append(
            self.model.addConstrs((period_to_regime[t, :].sum() == 1 for t in range(self.n_samples)), name="period assigned to one regime")
        )

        constraints.append(
            self.model.addConstrs((regime_to_candidate[k, :].sum() == 1 for k in range(self.n_locations)), name="one candidate assigned to one regime")
        )

        constraints.append(
            self.model.addConstrs((regime_to_candidate[:, j].sum() <= 1 for j in range(self.n_candidates)), name="candidate assigned to at most one regime")
        )
        
        impossible_candidates = process_impossible_indices(impossible_candidates)
        if impossible_candidates:
            for k in range(n_locations):
                if impossible_candidates[k]:
                    constraints.append(
                        self.model.addConstr(regime_to_candidate[k, impossible_candidates[k]] == 0, name=f"Impossible candidates for regime {k}")
                    )

        constraints.append(
            self.model.addConstrs((period_to_regime_to_candidate[t, :, :].sum() == 1 for t in range(self.n_samples)), name="one path")
        )

        constraints.append(
            self.model.addConstrs((period_to_regime_to_candidate[:, k, j] - period_to_regime[:, k] <= 0 for k, j in product(range(self.n_locations), range(self.n_candidates))), name="active 1")
        )

        constraints.append(
            self.model.addConstrs((period_to_regime_to_candidate[:, k, j] - regime_to_candidate[k, j] <= 0 for k, j in product(range(self.n_locations), range(self.n_candidates))), name="active 2")
        )

        self.model.update()
        self.model.optimize()  
        
        if self.model.Status == GRB.OPTIMAL:
            self.indices = regime_to_candidate.X.argmax(axis=1)
            self.labels_ = period_to_regime.X.argmax(axis=1)
        return self
    
    
class facility_location_nonmatrix(facility_location):        
    def fit(self, dist_mx, n_locations = 2):
        super().fit(dist_mx, n_locations)

        samples, candidates = range(self.n_samples), range(self.n_candidates)
        samples_prod_candidates = tuplelist(product(samples, candidates))

        dist_dict = {index: dist_mx[index] for index in samples_prod_candidates}

        assign = self.model.addVars(samples_prod_candidates, vtype=GRB.BINARY)
        select = self.model.addVars(candidates, vtype=GRB.BINARY)

        self.model.setObjective(assign.prod(dist_dict), GRB.MINIMIZE)

        constraints = []

        constraints.append(
            self.model.addConstrs((assign.sum(t, "*") == 1 for t in samples), name = "assign to only one")
        )

        constraints.append(
            self.model.addConstr(select.sum()==n_locations, 'select_total')
        )

        constraints.append(
            self.model.addConstrs((assign[t, j] <= select[j] for t, j in samples_prod_candidates), name = "activation")
        )

        self.model.update()
        self.model.optimize()
        
        if self.model.Status == GRB.OPTIMAL:
            self.indices, self.labels_ = self._extract_labels_from_assign_arr(extract_optimal_solution_as_array(assign, (self.n_samples, self.n_candidates), return_as_df=False))
        return self
