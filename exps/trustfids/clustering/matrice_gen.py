"""Foo x-evals matrice used for testing purposes 

"""
import random
from typing import Dict

import scipy.stats as stats


def init_foo_evals(mat_size: int)-> Dict[str, Dict[str, float]]:
    """
    Create a random dict of evaluation to test clustering/reputation 
    before FL cross evaluation results are avalaible. 
    """
    evaluations = {}
    id="cid"
    for i in range(mat_size): 
        participant_id=id+str(i+1)
        row = {}
        row[participant_id] = {}
        for j in range(mat_size):
            evaluation_id = id+str(j+1)
            row[evaluation_id]=random.uniform(0.0,1.0)
        evaluations[participant_id]=row
    return evaluations

def init_foo_evals_multi_cluster()->Dict[str,Dict[str,float]]: 
    """
    Create a dict of evaluation to test clustering/reputation 
    before FL cross evaluation results are avalaible.
    Result are ordered so that cluster might emerege     
    """


    # https://stackoverflow.com/questions/18441779/how-to-specify-upper-and-lower-limits-when-using-numpy-random-normal
    evaluations = {}
    id="cid"
    mat_size = 9 
    cluster_nb = 3
    cluster_size = mat_size//cluster_nb
    lower, upper = 0.0, 1.0
    
    mat = []
    mu, sigma = 0.1, 0.1
    X3 = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

    mu = 0.5
    X2 = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

    mu = 0.9
    X1 = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    
    for i in range(3):
        mat.append(list(X1.rvs(cluster_size))+list(X2.rvs(cluster_size))+list(X3.rvs(cluster_size)))
    for i in range(3):
        mat.append(list(X2.rvs(cluster_size))+list(X1.rvs(cluster_size))+list(X3.rvs(cluster_size)))
    for i in range(3):
        mat.append(list(X3.rvs(cluster_size))+list(X2.rvs(cluster_size))+list(X1.rvs(cluster_size)))
    
    for i in range(len(mat)):
        participant_id=id+str(i+1)
        evaluations[participant_id] = {}
        for j in range(len(mat)):
            evaluations[participant_id][id+str(j+1)]=mat[i][j]
    return evaluations


def init_foo_evals_multi_variable_cluster()->Dict[str,Dict[str,float]]: 
    """
    Create a dict of evaluation to test clustering/reputation 
    before FL cross evaluation results are avalaible.
    Result are ordered so that cluster might emerege     
    """


    # https://stackoverflow.com/questions/18441779/how-to-specify-upper-and-lower-limits-when-using-numpy-random-normal
    evaluations = {}
    id="cid"
    mat_size = 9 
    cluster_nb = 3
    cluster_size = mat_size//cluster_nb
    lower, upper = 0.0, 1.0
    good_sigma = 0.1
    bad_sigma = 0.3
    mat = []
    mu=0.1
    X3_good = stats.truncnorm(
        (lower - mu) / good_sigma, (upper - mu) / good_sigma, loc=mu, scale=good_sigma)
    X3_bad = stats.truncnorm(
        (lower - mu) / bad_sigma, (upper - mu) / bad_sigma, loc=mu, scale=bad_sigma)

    mu = 0.5
    X2_good = stats.truncnorm(
        (lower - mu) / good_sigma, (upper - mu) / good_sigma, loc=mu, scale=good_sigma)
    X2_bad = stats.truncnorm(
        (lower - mu) / bad_sigma, (upper - mu) / bad_sigma, loc=mu, scale=bad_sigma)

    mu = 0.9
    X1_good = stats.truncnorm(
        (lower - mu) / good_sigma, (upper - mu) / good_sigma, loc=mu, scale=good_sigma)
    X1_bad = stats.truncnorm(
        (lower - mu) / bad_sigma, (upper - mu) / bad_sigma, loc=mu, scale=bad_sigma)
    
    for i in range(2):
        mat.append(list(X1_good.rvs(cluster_size))+list(X2_good.rvs(cluster_size))+list(X3_good.rvs(cluster_size)))
    sigma = bad_sigma
    for i in range(1):
        mat.append(list(X1_bad.rvs(cluster_size))+list(X2_bad.rvs(cluster_size))+list(X3_bad.rvs(cluster_size)))
   
    sigma = good_sigma
    for i in range(2):
        mat.append(list(X2_good.rvs(cluster_size))+list(X1_good.rvs(cluster_size))+list(X3_good.rvs(cluster_size)))
    sigma = bad_sigma
    for i in range(1):
        mat.append(list(X2_bad.rvs(cluster_size))+list(X1_bad.rvs(cluster_size))+list(X3_bad.rvs(cluster_size)))
    
    sigma = good_sigma
    for i in range(2):
        mat.append(list(X3_good.rvs(cluster_size))+list(X2_good.rvs(cluster_size))+list(X1_good.rvs(cluster_size)))
    sigma = bad_sigma
    for i in range(1):
        mat.append(list(X3_bad.rvs(cluster_size))+list(X2_bad.rvs(cluster_size))+list(X1_bad.rvs(cluster_size)))
    
    for i in range(len(mat)):
        participant_id=id+str(i+1)
        evaluations[participant_id] = {}
        for j in range(len(mat)):
            evaluations[participant_id][id+str(j+1)]=mat[i][j]
    return evaluations


def init_foo_evals_simple_cluster()->Dict[str,Dict[str,float]]: 
    """
    Create a dict of evaluation to test clustering/reputation 
    before FL cross evaluation results are avalaible.
    Result are ordered so that cluster might emerege  
    """
    # https://stackoverflow.com/questions/18441779/how-to-specify-upper-and-lower-limits-when-using-numpy-random-normal
    evaluations = {}
    id="cid"
    lower, upper = 0.0, 1.0

    mat_size = 9 
    mat = []
    mu, sigma = 0.1, 0.1
    X1 = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

    mu, sigma = 0.5, 0.1
    X2 = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

    mu, sigma = 0.9, 0.1
    X3 = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

    for i in range(3):
        mat.append(list(X1.rvs(mat_size)))
    for i in range(3):
        mat.append(list(X2.rvs(mat_size)))
    for i in range(3):
        mat.append(list(X3.rvs(mat_size)))
        
    for i in range(len(mat)):
        participant_id=id+str(i+1)
        evaluations[participant_id] = {}
        for j in range(len(mat)):
            evaluations[participant_id][id+str(j+1)]=mat[i][j]
    return evaluations


