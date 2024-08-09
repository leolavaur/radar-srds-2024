"""unit test for utils.py
"""
import json
from unittest import TestCase

from trustfids.utils.tests.data_examples import distribution_attacker, example_xeval

from ..utils import *


class TestUtils(TestCase) : 
    """
    Test suite for utils.py functions
    """

    def setup_class(self):
        self.a = {"z":0.5, "q":0.8,"s":0.3,"d":0.1}
        self.b = {"z":0.2, "q":0.7,"s":0.5,"d":0.3}
        self.c = {"z":0.3, "q":0.1,"s":0.2,"d":0.2}

        ################################
        # Used for matrice inversion.
        ################################
        in_dir = "./trustfids/xeval/saved_matrices/"
        file = "matrix_1"
        f = open(f"{in_dir}{file}.json")
        self.xevals = json.load(f)    
    
    def test_mat_inversion(self): 
        inversed = mat_inversion(self.xevals)
        keys = self.xevals.keys()
        for key in keys : 
            for subkey in keys: 
                assert inversed[key][subkey] == self.xevals[subkey][key] 

    def test_validate_symetry(self):
        assert vector_substraction(self.a,self.b) == vector_substraction(self.b,self.a)
    
    def test_zip_evals(self):
        ziped = zip_evals(self.a,self.b)
        assert ziped.pop() == (0.1,0.3)
        
    
    def test_l2_norm(self): 
        v1 = {str(i):1.0 for i in range(10)}
        assert l2_norm(v1) == np.sqrt(10)

        v2 =  {str(i):0.0 for i in range(10)}
        assert l2_norm(v2) == 0

        v3 =  {str(i):np.random.rand() for i in range(10)}
        assert l2_norm(v3) == np.sqrt(np.sum(np.square(list(v3.values()))))
    
    def test_cosin_similarity(self):
        # test like l2_norm ? 
        assert cosin_similarity(self.a,self.b) >= 0
        
    def test_distance(self):
        x = vector_substraction(self.a,self.b)
        assert distance_xevals(self.a,self.b,distance_type="euclidean") == l2_norm(x)
        assert distance_xevals(self.a,self.b,distance_type="cosin_sim")>=0.0
        with self.assertRaises(ValueError):
            distance_xevals(self.a,self.b,"ward")
    
    def test_distance_xevals_matrice(self):
        nd_distance_xeval = distance_xevals_matrice(example_xeval,"cosin_sim")
        assert nd_distance_xeval[1,0] == nd_distance_xeval[0,1]
        assert distance_xevals(example_xeval["client_9"],example_xeval["client_8"],distance_type="cosin_sim") == nd_distance_xeval[9,8]
        

class TestUtilsModels(TestCase) : 
    """
    Test suite for utils.py models related functions
    """
    def setup_class(self):
        self.models = {f"client_{i}": np.array(np.random.rand(10)) for i in range(10)}            
    
    def test_cosin_similarity_models(self):
        cosin_sim = cosin_similarity_models(self.models["client_1"],self.models["client_2"])
        assert cosin_sim >= 0
    
    def test_l2_norm_models(self):
        L2_norm = l2_norm_models(self.models["client_1"]-self.models["client_2"])
        assert L2_norm >= 0 
                
    def test_distance_models(self): 
        m1 = self.models["client_1"]
        m2 = self.models["client_2"]
        assert distance_models(m1,m2, "euclidean") == l2_norm_models(m1-m2)
        assert distance_models(m1,m2, "cosin_sim") == cosin_similarity_models(m1,m2)
        with self.assertRaises(ValueError):
            distance_models(m1,m2,"ward")