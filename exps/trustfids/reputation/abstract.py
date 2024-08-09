"""
Abstract reputation class
"""
from typing import Dict, List
from abc import ABC, abstractmethod

class ReputationEnvironment(ABC):
    """
    Reputation environment, 
    Interact with the history
    """
    
    @abstractmethod
    def __init__(self, class_nb:int=10): 
        """
        Args: 
            class_nb : number of class used for evaluation discretization. 
        """
        pass

    
    @abstractmethod
    def new_round(self, clusters: List[List[str]], xevals: Dict[str, Dict[str, float]])->None: 
        """
        Add a new round to the reputation environment. 
         
        Args:
            cluster:
                List[string] : every participants id from cluster members:
                    [
                        "cid1","cid3","cid7"
                    ]
            xevals:
                cross evaluation results for the current round
        """
        
    @abstractmethod
    def compute_cluster_weights(self, cluster: List[str] ,round_nb:int=None): 
        """Compute weights for a cluster at a specific round
        Args: 
            cluster: cluster to compute the weights on.
            round_nb: weights for this round are computed, default to current round. 
        Return: 
            Dict[str, float] : aggregtion weights for cluster members:
            {
                "cid1" : 0.2,
                "cid3" : 0.5,
                "cid7" : 0.3,
            }
        """