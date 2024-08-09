"""Unit test for similarity 
"""
import json

from ..similarity import *
from ..utils import *

path_from_exec = "./trustfids/reputation/tests/"
class TestSimilarity:

    def setup_class(self) :
        path_test_data = f"{path_from_exec}discretization_matrice_test.json"
        self.xevals = json.load(open(path_test_data))
        self.cluster = ["client_5","client_4","client_6","client_7"]
        self.centroid = centroid_from_cluster(self.cluster,self.xevals)

    def test_similarity_to_centroid(self): 
        sim = similarity_to_centroid(self.xevals[self.cluster[0]],self.centroid)
        assert 0 <= sim <= 1.0 

    def test_similarity_participant(self): 
        sim = similarity_participants("client_4","client_5",self.xevals)
        assert 0 <= sim <= 1.0 

    def test_similarity_for_cluster_members(self): 
        sim = similarity_for_cluster_members(self.cluster,self.xevals)
        assert len(sim.keys()) ==4
        for key in self.cluster: 
            assert key in sim
        for value in sim.values(): 
            assert 0 <= value <= 1.0 