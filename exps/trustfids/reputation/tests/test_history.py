"""unit test for history 
"""
from ...clustering.cluster import build_clusters
from ...clustering.matrice_gen import (
    init_foo_evals_multi_cluster,
    init_foo_evals_simple_cluster,
)
from ..history import Hist
from ..utils import discretize_matrice


class TestHistory:
    """
    Hist class testing 
    """
    def setup_class(self): 
        self.hist = Hist()    
        self.test_add_round(self)
    
   
    
    def test_get_round(self):
        self.hist.get_round("r1")
        self.hist.get_round("r2")

    def test_get_last_round(self):
        assert self.hist.get_last_round().id == "r2"

    def test_get_previous_round(self): 
        assert self.hist.get_previous_round("r4") == None
        assert self.hist.get_previous_round("r1") == None
        assert self.hist.get_previous_round("r2").id == "r1"
        
    def test_get_xevals(self):
        self.hist.get_round_xevals("r2")

    def test_get_clusters(self):
        self.hist.get_round_clusters("r2")
        
    # placed last to keep results preserved.
    def test_add_round(self):     
            evals = init_foo_evals_multi_cluster()
            clusters = build_clusters(evals,alpha=0.8)
            discrete_xevals = discretize_matrice(evals,10)
            self.hist.add_round(evals, discrete_xevals, clusters)

            evals = init_foo_evals_simple_cluster()
            clusters = build_clusters(evals,alpha=0.8)
            self.hist.add_round(evals, discrete_xevals, clusters)