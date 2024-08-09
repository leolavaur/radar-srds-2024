"""
Tool for extracting relevant reputation results in json file. 
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from trustfids.reputation.history import Hist


def log_weights(weights:Dict[str,float], history:Hist,exploded:bool=True): 
    """_summary_

    Args:
        weights (Dict[str,float]): Weights of the participants at a specific round that should be saved. 
        history (Hist): history of the run.
        exploded (bool, optional): True if the score have been further exploded False if it's the normalized weight out of the reputation system. Defaults to True.
    """
    p  = Path() / "client_weights_exploded.json" if exploded else Path() / "client_weights.json" 
    r:str = history.get_round_id()
    if p.exists():
        full_log:Dict[str,Any] = json.load(open(p))
        if r not in full_log : 
            full_log[r] = {}     
        full_log[r].update(weights) 
        p.write_text(json.dumps(full_log,indent=2)) 
    else: # First round
        p.write_text(
            json.dumps(
                {r: weights}
            )
        )   