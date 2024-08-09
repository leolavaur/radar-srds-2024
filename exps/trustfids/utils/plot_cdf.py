from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import MarkerStyle
from scipy.stats import median_abs_deviation, norm
from trustfids.utils.plot_utils import (
    check_paths,
    load_distribution,
    load_weights,
    markers,
)


def participants_over_cdf(dir: str, dataset:str, sigma:float=0, r:str="r10", title:str=""):
    """Plot the exploded weights of participants. 

    Args:
        dir (str): Run results folder uppon which weights should be found.
        dataset (str): Dataset whose client should be evaluated. 
        sigma (float, optional): If left to zero the standard deviation of the distribution is used. Otherwise the specified value is used. Defaults to 0.
        r (str, optional): Round weights that should be used. Defaults to "r10".
    """
     
    # Faire varier les méthodes pour fixer sigma ?  
    # Intéresser de mettre en titre l'attaque.      
    p = Path(dir)
    check_paths(p)
    w = load_weights(p)
    d = load_distribution(p)
    participants = d[dataset]
    valeurs = [w[r][p] for p in participants]
    attackants = [w[r][p] for p in participants if "attacker" in p]
    benign = [w[r][p] for p in participants if "client" in p]
    x = np.linspace(-0.01, 0.01, 10000)
    if not sigma : 
        std = np.std(valeurs)
    else :
        std = sigma
  
    sample_size = 1000

    # STD
    mean = 0
    cdf = norm.cdf(x, mean, std)
    cdf_sigma_1_6 = norm.cdf(x, mean, std * 1.6)
    cdf_sigma_2 = norm.cdf(x, mean, std * 2)

    # Plot the CDF of the normal distribution
    plt.figure(figsize=(10, 6))
    if not sigma : 
        plt.plot(x, cdf, "b", label="CDF 1.0 sigma")
        plt.plot(x, cdf_sigma_1_6, "r", label="CDF 1.6 sigma")
        plt.plot(x, cdf_sigma_2, "g", label="CDF 2 sigma")
    else : 
        plt.plot(x, cdf, "b", label=f"CDF σ={sigma}")
        
    if attackants : 
        a_ecart_moy = [a - 0.2 for a in attackants]
        a_cdf = norm.cdf(a_ecart_moy, mean, std)
        plt.scatter(
            a_ecart_moy, a_cdf, color="r", marker=MarkerStyle("d"), label="Attacker weight"
        )
    
    b_ecart_moy = [b - 0.2 for b in benign]
    b_cdf = norm.cdf(b_ecart_moy, mean, std)
    plt.scatter(
        b_ecart_moy, b_cdf, color="g", marker=MarkerStyle("o"), label="Benign participants weight"
    )
    plt.title(f"{dataset} {title} /  standard devation of weights")
    plt.xlabel("Value")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(True)