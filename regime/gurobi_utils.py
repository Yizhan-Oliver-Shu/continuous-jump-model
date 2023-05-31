from gurobipy import *
import numpy as np
import pandas as pd

def extract_optimal_solution_as_array(x, shape=None, return_as_df=True, index=None, columns=None):
    """
    for a tupledict of gurobi Vars, extract its values into an array, with an optional shape (only 1d/2d data are ideally supported).
    
    Parameters:
    ----------------------------------
    x: tupledict
    
    shape: tuple of length 2 or None. default None
        None implies that the data is 1-dimension.
    
    """
    if not isinstance(x, tupledict):
        raise Exception("Can only accept gurobi tupledict type as input!")
    # all values into 1d array    
    arr = np.array(list(map(lambda s:s.X, x.values())))
    
    if not shape:  # 1d data
        if not return_as_df:  # return array
            return arr
        # return series
        return pd.Series(arr, index=x.keys())
    # 2d data
    arr = arr.reshape(shape)
    if not return_as_df: # return array
        return arr
    # return df
    return pd.DataFrame(arr, index=index, columns = columns)


def print_value_runtime(m):
    """
    print the optimal value and runtime of a solved gurobi model.
    """
    print(f"Optimal Value: {m.ObjVal}")
    print(f"Runtime: {m.Runtime}s.")    
    return 