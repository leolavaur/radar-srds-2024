"""unit test for reputation 
"""
import json

import pytest
from numpy import isclose

from ...clustering.cluster import build_clusters
from ...utils.plot_utils import extract_xevals
from ..reput import *

path_from_exec = "./trustfids/reputation/tests/"
class TestReputation:
    """
    Hist class testing 
    """
    # No class in reput for now 
    # def setup_class(self): 

    def test_similarity(self): 
        pass
        
    
    def test_weight_votes(self):
        pass

class TestDirichletReputationEnvironment:
    """
    Hist class testing 
    """
    # No class in reput for now 
    # def setup_class(self): 

    def setup_class(self):
        """
        Init Dirichlet reput env.
        """
        self.dirich = DirichletReputationEnvironment(log=False, class_nb = 10000)
        path_test_data = f"{path_from_exec}xevals_test.json"
        self.r1_xevals= json.load(open(path_test_data))
        self.r1_clusters = build_clusters(self.r1_xevals, threshold_type="mean")
        
    def test_similarity(self): 
        pass
        
    def test_exponential_decay(self): 
        decayed = self.dirich.exponential_decay(1,0)
        assert decayed == 1
        assert 0.0 <= decayed <=1.0
        
        self.dirich.lmbd = 0.0
        decayed = self.dirich.exponential_decay(1,0)
        assert decayed == 1
        assert 0.0 <= decayed <=1.0
        self.dirich.lmbd = 0.3
        
        decayed = self.dirich.exponential_decay(0,0)
        assert 0.0 <= decayed <=1.0
        assert decayed == 0

        decayed = self.dirich.exponential_decay(1,100)
        assert 0.0 <= decayed <=1.0

    def test_compute_cluster_weights(self):
        self.dirich.new_round(self.r1_clusters,self.r1_xevals)
        # r1_clusters[3] = ['client_7', 'client_6', 'client_4', 'client_5']
        # cluster = self.r1_clusters[3]
        for cluster in self.r1_clusters :
            c_weights = self.dirich.compute_cluster_weights(cluster)
            assert isclose(sum(c_weights.values()),1.0)
        
        #Single element in cluster 
        c_weights = self.dirich.compute_cluster_weights(self.r1_clusters[0])
        
        #Empty cluster
        with pytest.raises(ValueError):
            self.dirich.compute_cluster_weights([])

    
    def test_compute_cluster_weights_2_rounds(self):
        self.dirich.new_round(self.r1_clusters,self.r1_xevals)
        for cluster in self.r1_clusters :
            c_weights = self.dirich.compute_cluster_weights(cluster)
            assert isclose(sum(c_weights.values()),1.0)
            # assert sum(c_weights.values()) == 1.0
        
    
    # At the end of testing to prevent impact on others tests. 
    def test_new_round(self):
        # Note : the round added for this test isn't accessible by other unitest.
        self.dirich.new_round(self.r1_clusters,self.r1_xevals)

    def test_lambda_0(self): 
        dirich0 = DirichletReputationEnvironment(lmbd=0.0, log=False) 
        lambda_0 = f"{path_from_exec}/lambda_0"   
        xevals,_ = extract_xevals(lambda_0 )
        clusters:List[List[str]] = json.load(open(lambda_0 +"/clusters.json"))
        for xeval in xevals.values():
            dirich0.new_round(clusters,xeval)
        
        for i in range(10): 
            for clust in clusters :
                c_weights = dirich0.compute_cluster_weights(clust,i+1)
                assert math.isclose(math.fsum(list(c_weights.values())), 1.0)