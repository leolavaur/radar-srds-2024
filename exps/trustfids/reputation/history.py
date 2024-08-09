"""
Store cross-evaluation and reputation information.
"""
from typing import Dict, List, Tuple


class Round : 
    """
    All elements that are stored for a given round.
    Dict: with key "xevals" and "clusters" for the cross evaluation
                and cluster list from this round. Structure :  
                {
                    "xevals" : {
                            "cid1": {
                                "cid1": 0.7,
                                "cid2": 0.6,
                                ...
                            },...
                    }, 
                    "d_xevals" : {
                            "cid1": {
                                "cid1": 1,
                                "cid2": 6,
                                ...
                            },...
                    },
                    "clusters": {
                            ["cid1","cid2","cid3"],
                            ...    
                    },
                    "id": "r1",
                    "round_number": 1
                }

    """
    def __init__(self,xevals:Dict[str, Dict[str, float]], clusters:List[List[str]], discrete_xevals:Dict[str, Dict[str, int]] ,id:str, round_nb:int): 
        self.xevals:Dict[str, Dict[str, float]] = xevals
        self.clusters:List[List[str]] = clusters
        self.normed_xevals:Dict[str, Dict[str, int]] = discrete_xevals
        self.id:str  = id 
        self.round_number:int = round_nb
        
class Hist:
    """    
            hist: A dictionary that store evaluations and cluster for each 
            roundd, with the following structure:
            {
                "r1": {
                    "xevals": {
                            "cid1": {
                                "cid1": 0.7,
                                "cid2": 0.6,
                                ...
                            },
                            "cid2": {
                                "cid1": 0.8,
                                "cid2": 0.5,
                                ...
                            },
                            ...
                    },
                    "clusters":[
                                ["cid1","cid2","cid3"],
                                ["cid4"],
                                ["cid5","cid6"]
                    ]
                },
                "r2": {
                    "xevals": {
                            "cid1": {
                                "cid1": 0.7,
                                "cid2": 0.6,
                                ...
                            },
                            "cid2": {
                                "cid1": 0.8,
                                "cid2": 0.5,
                                ...
                            },
                            ...
                    },
                    "clusters":[
                                ["cid1","cid2","cid3"],
                                ["cid4,cid5"],
                                ["cid6"]
                    ]
                },
                ...
            }

    """ 
    def __init__(self): 
        self.rounds: Dict[str, Round] = {}
        self.round_counter:int = 1 
        self.next_id:str = "r1"
        self.current_id:str = ""

    def add_round(self, xevals:Dict[str, Dict[str, float]], discrete_xevals:Dict[str, Dict[str, int]],clusters:List[List[str]]): 
        """
        Store the list of clusters and the cross evaluation for the current round. 
        Auto-increment round id.
        """
        # Control on clusters and xevals dimmensions ? 
        self.rounds[self.next_id] = Round(xevals,clusters, discrete_xevals,self.next_id,self.round_counter)
        self.round_counter+=1
        self.current_id = self.next_id
        self.next_id="r"+str(self.round_counter)
    
    def get_round(self, round_id:str)->Round: 
        """
        Return a round object containing both the cross evaluation and the list of clusters for specified
        round. 
            Args: 
                round_id: string identifying the round example : "r1"
            Returns: 
                Round : the round with the requested id.
    
        """
        return self.rounds[round_id]
    
    def get_round_id(self)->str:
        """Return the id of the current round. 

        Returns:
            str: str at the format "r1" where 1 is replaced by the round number.
        """
        return self.current_id
  
    def get_last_round(self)-> Round: 
        """
        Return the last round 
        Returns: 
                Round : last round
        """ 
        return self.rounds[self.current_id]

    def get_previous_round(self, id)-> Round: 
        """
        Return the round preceeding the submitted id 
        Args: 
                id : id for the current round
        Returns: 
                Round : last round
        """ 
        nb = int(id[1])-1
        if nb == 0 :
            return None
        elif nb >= self.round_counter : 
            return None
        prev_id = "r"+str(nb) 
        return self.rounds[prev_id]

    def get_round_xevals(self, round_id:str)->Dict[str, Dict[str, float]]: 
        return self.rounds[round_id].xevals
    
    def get_round_clusters(self, round_id:str)->List[List[str]]: 
        return self.rounds[round_id].clusters

    def get_round_normed_xevals(self, round_id:str)->Dict[str, Dict[str, float]]: 
        return self.rounds[round_id].normed_xevals
