"""unit test for clustering 
"""
import json
from copy import deepcopy
from pathlib import Path
from unittest import TestCase

import pytest
from numpy import float64
from trustfids.clustering.cluster import *
from trustfids.clustering.matrice_gen import (
    init_foo_evals_multi_cluster,
    init_foo_evals_simple_cluster,
)
from trustfids.reputation.history import Hist

path_from_exec = "./trustfids/clustering/tests/"

class TestClusters(TestCase):
    """
    Test the class Clusters
    """ 
    def setup_class(self):
        self.xevals_r1 = json.load(open(Path(f"{path_from_exec}/data/xeval_accuracy_1.json")))
        self.clusters = Clusters(self.xevals_r1,"xevals","cosin_sim")  
        
        # Unique identifier 
        for c in self.clusters.clusters: 
            id_count = 0
            for c2 in self.clusters.clusters:
                if c.id == c2.id:
                    id_count +=1 
            assert id_count == 1 
        
        # Extract participants from each cluster.
        participants = [c.participants[0] for c in self.clusters.clusters]
        
        # Validate that xevals elems have been creted 
        assert len(list(self.clusters.data.keys())) == len(participants)
        for p in self.clusters.data.keys(): 
            assert p in participants 
        
        
    def test_initial_inter_distance(self):
        # Test that inter values make sense. 
        for c1 in self.clusters.clusters: 
            for c2 in self.clusters.clusters: 
                if c1.id == c2.id : 
                    continue
                assert type(self.clusters.inter[c1.id][c2.id]) is float64
                assert self.clusters.inter[c1.id][c2.id] >=0
                # assert self.clusters.inter[c1.id][c2.id] <=1
                assert self.clusters.inter[c1.id][c2.id] == self.clusters.inter[c2.id][c1.id]

    def test_mean_intradistance(self):
        # At init all elements are single and equal to the centroid of their cluster
        assert self.clusters.mean_intradistance==0
    
    def test_mean_interdistance(self):
        assert self.clusters.mean_interdistance > 0
    
    def test_min_interdistance(self):
        c1,c2,inter = self.clusters.min_interdistance()
        
        assert type(c1) is ClusterXevals
        assert type(c2) is ClusterXevals
        assert type(inter) is float64
        
        assert inter == min(min(d.values()) for d in self.clusters.inter.values())

    def test_get_cluster(self): 
        c1 = self.clusters.get_cluster("c1")
        assert type(c1) is ClusterXevals
        assert c1.id == 'c1'
        
        with pytest.raises(IndexError):
            self.clusters.get_cluster("c99")
    
    def test_get_printable_interdistance(self): 
        a= self.clusters.get_printable_interdistance()
        k,v = next(iter(a.items()))
        assert isinstance(k,str)
        assert isinstance(v, Dict)
        assert 2.0 > next(iter(v.values())) > 0.0
        
    def test_erase_cluster(self): 
        new_clusters = Clusters(self.xevals_r1,"xevals","cosin_sim")
        c1 = new_clusters.clusters[0]
        c1_id = c1.id        
        new_clusters.erase_cluster(c1)
        self.assertNotIn(c1,new_clusters.clusters)
        self.assertNotIn(c1_id,new_clusters.inter)
        for c_inter in new_clusters.inter.values(): 
            self.assertNotIn(c1_id,c_inter) 
    
    def test_merge_cluster(self): 
        new_clusters = Clusters(self.xevals_r1,"xevals","cosin_sim")
        c2_participants = new_clusters.clusters[1].participants[0]
        c1_intra = new_clusters.clusters[0].intra
        c1_inter = deepcopy(new_clusters.inter['c1'])
        new_clusters.merge_cluster(new_clusters.clusters[0],new_clusters.clusters[1])
        
        # Check that participants from c2 have been merged into c1
        self.assertIn(c2_participants, new_clusters.clusters[0].participants)
        
        # Check that c1 intra distance have been updated
        assert c1_intra != new_clusters.clusters[0].intra
        
        # Check that inter distance have been updated 
        for c in new_clusters.clusters:
            if c.id == 'c1':
                continue
            assert c1_inter[c.id] != new_clusters.inter['c1'][c.id]
        
        # Verify numerical values the check the correctness of the update ? 
        
class TestClusterXevals(TestCase):
    """
    Testing the Cluster
    """
    def setup_class(self):
        self.xevals_r1 = json.load(open(Path(f"{path_from_exec}/data/xeval_accuracy_1.json")))
        self.clusters = Clusters(self.xevals_r1,"xevals","euclidean")
        self.cluster = ClusterXevals(id="1",participants=["client_0"],clusters = self.clusters,xevals=self.xevals_r1)
        self.cluster_two_elems = ClusterXevals(id="2",participants=["client_0","client_1"],clusters = self.clusters, xevals=self.xevals_r1)

    def test_cluster_centroid(self):
        centroid = self.cluster.cluster_centroid()
        
        # Coherent typing
        assert isinstance(centroid,dict), "centroid is not a dictionnary"
        
        key, value = next(iter(centroid.items()))
        assert isinstance(key,str)
        assert isinstance(value,float) 

        # Exact results
        assert value == 0.00364 #Client0-> Client0 
        self.cluster_two_elems = ClusterXevals(id="2",participants=["client_0","client_1"],clusters = self.clusters, xevals=self.xevals_r1)
        centroid = self.cluster_two_elems.cluster_centroid()
        key, value = next(iter(centroid.items()))
        assert value == 0.003585 #Mean Client0-> Client0 Client1-> Client0         

        
    def test_intra_cluster_distance(self):
        c1_center = self.cluster.intra_cluster_distance()
        self.cluster_two_elems = ClusterXevals(id="2",participants=["client_0","client_1"],clusters = self.clusters, xevals=self.xevals_r1)
        c2_center = self.cluster_two_elems.intra_cluster_distance()
        
        assert c1_center == 0
        assert c2_center >= 0.034747026851 
        assert c2_center <= 0.034747026852 
        
    def test_inter_cluster_distance(self):
        # Cant comput intercluster distance with the same cluster.
        # Refactor the function to use inter in Clusters not cluster.
        with self.assertRaises(ValueError):
            self.cluster.inter_cluster_distance(self.cluster)
    
        # Create a new Clusters 
        self.c_test = ClusterXevals(id="c_test",participants=["client_0","client_1"],clusters = self.clusters, xevals=self.xevals_r1)
        assert self.cluster.inter_cluster_distance(self.c_test) <= 0.0347471
        assert self.cluster.inter_cluster_distance(self.c_test) >= 0.0347470
        
    def test_add_participant(self): 
        old_intra = self.cluster_two_elems.intra 
        self.cluster_two_elems.add_participants(["client_3"])
        assert old_intra != self.cluster_two_elems.intra
        
        self.cluster_two_elems.add_participants(["client_4","client_5"])
        assert "client_4" in self.cluster_two_elems.participants
        assert len(self.cluster_two_elems.participants) == 5
    
        
  
class TestClusterModels(TestCase):
    """
    Testing the cluste class on models.
    TODO_testing : complement the test with numerical comparisons once Models files will be extacted.
    """
    def setup_class(self):
        self.data = {f"client_{i}": np.array(np.random.rand(10)) for i in range(10) }  
        self.clusters = Clusters(self.data,"models","cosin_sim")
        self.cluster = ClusterModel(id="1",participants=["client_0"], models=self.data, clusters = self.clusters)
        
        # self.cluster_two_elems = ClusterXevals(id="2",participants=["client_0","client_1"],clusters = self.clusters, )

    def test_cluster_centroid(self):
        centroid = self.cluster.cluster_centroid()
        # TODO_type_assert? 
        # assert isinstance(centroid, NDArray)
        
                
    def test_intra_cluster_distance(self): 
        self.cluster.intra_cluster_distance()
        self.cluster_two = ClusterModel(id="2",participants=["client_0","client_1"],clusters = self.clusters,models=self.data)
        self.cluster_two.intra_cluster_distance()
        #  TODO_testing numerical values with extracted cluster
    
    def test_inter_cluster_distance(self): 
        with self.assertRaises(ValueError):
            self.cluster.inter_cluster_distance(self.cluster)
        self.cluster.inter_cluster_distance(cast(ClusterModel,self.clusters.clusters[-1]))

    def test_add_participant(self):
        self.cluster_two = ClusterModel(id="2",participants=["client_0","client_1"],clusters = self.clusters,models=self.data)
        old_intra = self.cluster_two.intra 
        self.cluster_two.add_participants(["client_3"])
        assert old_intra != self.cluster_two.intra
        
        self.cluster_two.add_participants(["client_4","client_5"])
        assert "client_4" in self.cluster_two.participants
        assert len(self.cluster_two.participants) == 5
    
class TestClustering():
    """
    Testing the clustering that are not in a class
    """
    def setup_class(self): 
        self.xevals = json.load(open(Path(f"{path_from_exec}/data/xeval_accuracy_1.json")))
        self.models = {f"client_{i}": np.array(np.random.rand(10)) for i in range(10) }            
        self.fixed_evals = {"p1":{"p1":0.8,"p2":0.8,"p3":0.8},"p2":{"p1":0.5,"p2":0.5,"p3":0.5}, "p3":{"p1":0.1,"p2":0.1,"p3":0.1}}
        
        
    def test_build_cluster(self): 
        #Xevals  
        # Fixed threshold
        clusters = build_clusters(self.xevals,alpha=0.8)
        assert type(clusters) is list
        assert type(clusters[0]) is list
        assert type(clusters[0][0]) is str
        
        # Dynamic threshold
        build_clusters(self.xevals, threshold_type="dynamic",alpha=1.5)
        # Mean threshold
        build_clusters(self.xevals, threshold_type="mean")
        # Mean iterative threshold
        build_clusters(self.xevals, threshold_type="mean_iterative")
        # Euclidean distance 
        build_clusters(self.xevals, threshold_type="mean", distance_type="cosin_sim")
        
        #Models  
        # Fixed threshold
        clusters = build_clusters(self.models, input_type="models", alpha=0.8)
        assert type(clusters) is list
        assert type(clusters[0]) is list
        assert type(clusters[0][0]) is str
        
        # Dynamic threshold
        build_clusters(self.models, input_type="models",  threshold_type="dynamic",alpha=1.5)
        # Mean threshold
        build_clusters(self.models, input_type="models",  threshold_type="mean")
        # Mean iterative threshold
        build_clusters(self.models, input_type="models",  threshold_type="mean_iterative")
        # Euclidean distance 
        build_clusters(self.models, input_type="models",  threshold_type="mean", distance_type="cosin_sim")
        
    def test_mean_xevals(self):
        """
        testing mean_xevals on hardcoded values
        """
        fixed_xevals = {"p1":{"p1":np.float32(0.6),"p2":np.float32(0.3),"p3":np.float32(0.2)},"p2":{"p1":np.float32(0.7),"p2":np.float32(0.5),"p3":np.float32(0.2)}, "p3":{"p1":np.float32(0.8),"p2":np.float32(0.7),"p3":np.float32(0.2)}}
        mean = mean_xevals(fixed_xevals)
        assert mean == {'p1':np.float32(0.7), 'p2': np.float32(0.5), 'p3': np.float32(0.2)}
 