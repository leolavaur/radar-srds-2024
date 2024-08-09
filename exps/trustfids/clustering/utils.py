"""Utilities for x-evals manipulation

Used by clustering.
"""
from typing import Dict, List, Tuple

import numpy as np
import scipy as sp
from numpy import dot
from numpy.linalg import norm
from numpy.typing import NDArray


def l2_norm(vec: Dict[str, float]) -> float:
    """return the l2 norm of an evaluation vector."""
    total = 0
    tab = list(vec.values())
    return np.sqrt(np.sum(np.square(tab)))

def l2_norm_models(model:NDArray) -> float:
    """return the l2 norm of an NDarray"""
    return  float(norm(model)) 

def zip_evals(
    eval1: Dict[str, float], eval2: Dict[str, float]
) -> List[Tuple[float, float]]:
    """Zip two evals into an ordered list of tuple

    Args:
        eval1 (Dict[str, float]):  where str is an identifier and float a scalar.
        eval2 (Dict[str, float]):  where str is an identifier and float a scalar. vec2 must have value for all keys from vec1.

    Returns:
        List[Tuple[float,float]]: _description_
    """
    return [(eval1[key], eval2[key]) for key in eval1]

def cosin_similarity_models(m1:NDArray,m2:NDArray)-> float:
    """return the cosin similarity of two NDArray
    """
    return 1-dot(m1, m2)/(norm(m1)*norm(m2))

def cosin_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """return the cosin distance of two vectors
    args:
        vec1 (Dict[str, float]):  where str is an identifier and float a scalar.
        vec2 (Dict[str, float]):  where str is an identifier and float a scalar. vec2 must have value for all keys from vec1.
    return:
        cosin distance of vec 1 and 2.
    """
    x, y = zip(*zip_evals(vec1, vec2))
    return sp.spatial.distance.cosine(x, y)


def vector_substraction(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    """difference of two vector"""
    return {key: abs(b[key] - a[key]) for key in b.keys()}

def distance_xevals_matrice(xevals : Dict[str, Dict[str, float]],distance_type:str)->np.ndarray[float]:
    keys = list(xevals.keys())
    size = len(keys)
    square_ndarray = np.empty((size, size), dtype=float)
    
    for i, outer_key in enumerate(keys):
        inner_dict = xevals[outer_key]
        for j, inner_key in enumerate(keys):
            square_ndarray[i, j] = distance_xevals(xevals[outer_key],xevals[inner_key],distance_type=distance_type)
    return square_ndarray    
    
def distance_xevals(
    vec1: Dict[str, float], vec2: Dict[str, float], distance_type: str
) -> float:
    """_summary_

    Args:
        vec1 (Dict[str, float]): _description_
        vec2 (Dict[str, float]): _description_
        distance_type (str): Methodology used to compute distance, must be in ["euclidean","cosin_sim"]

    Returns:
        float: _description_
    """
    if distance_type == "euclidean":
        vec_dif = vector_substraction(vec1, vec2)
        return l2_norm(vec_dif)
    elif distance_type == "cosin_sim":
        return cosin_similarity(vec1, vec2)
    else:
        raise ValueError(
            f"Distance_type must be in ['euclidean','cosin_sim'] but it's {distance_type}"
        )
        
def distance_models(m1 : NDArray,m2 : NDArray, distance_type:str)->float:
    if distance_type == "euclidean":
        return l2_norm_models(m1-m2)
    elif distance_type == "cosin_sim":
        return cosin_similarity_models(m1, m2)
    else:
        raise ValueError(
            f"Distance_type must be in ['euclidean','cosin_sim'] but it's {distance_type}"
        )
    

def mean_xevals(xevals: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Mean values from an evaluation vector"""
    mean = {}
    for client in xevals.keys():
        client_evals = []
        for eval in xevals.values():
            client_evals.append(eval[client])
        mean[client] = np.mean(client_evals, dtype=np.float32)
    return mean


def mat_inversion(xevals: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """xevals inversion
    xevals have the format [A][X]=eval(A->X)  .
    this function return [A][X]= eval(X->A)
    """
    inversed = {}
    keys = xevals.keys()
    for key in keys:
        inversed[key] = {}
        for subkey in keys:
            inversed[key][subkey] = xevals[subkey][key]
    return inversed
