"""
Tool for extracting relevant clustering results in json file. 
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

@dataclass
class Merge: 
    threshold:float
    inter:float
    c1:List[str]
    c2:List[str]
class ClusterLog:
    def __init__(self, threshold_type:str, initial_inter:Dict[str,Dict[str,float]]):
        """If log is True print clustering log in current directory

        Args:
            threshold_type (str): type of clustering used
        """
        self.threshold_type = threshold_type
        self.merge_nb=0
        self.fusions: Dict[int,Merge]={}
        self.initial_inter = initial_inter
    
    def log_merge(self, threshold:float, c1_c2:float, c1_p:List[str], c2_p:List[str]):
        """_summary_

        Args:
            threshold (float): thresold used as the limit.
            c1_c2 (float): inter distance.
            c1_p (List[str]): receiving cluster participants.
            c2_p (List[str]): merged cluster participants.
        """
        self.fusions[self.merge_nb]=Merge(threshold=threshold,inter=c1_c2,c1=c1_p,c2=c2_p)
        self.merge_nb+=1
        
    def print_results(self,clusters:List[List[str]],final_inter_distance :Dict[str,Dict[str,float]]):
        """Check for the existence of clustering_logs file in the current directory
        create it if necessary. Otherwise append the necessary results to this file.
        """
        merging_log = {k:asdict(fus) for k,fus in self.fusions.items()}
        p  = Path() / "clusters_merging.json"
        if p.exists(): #Round > 1
            full_log:Dict[str,Any] = json.load(open(p))
            # Assert current round
            max:int = 0
            for k in full_log.keys():
                if k[0] == "r":
                   if int(k[1]) >= max:
                    max = int(k[1])
            r = f"r{max+1}"
            
            # Add merging log for the current round
            full_log[r]=merging_log 
            p.write_text(json.dumps(full_log)) 
        else: # First round
            p.write_text(
                json.dumps(
                    {"threshold_type": self.threshold_type, "initial_interdistance":self.initial_inter,"final_interdistance":final_inter_distance ,"end_clusters":clusters,"r1":merging_log}
                )
            )   