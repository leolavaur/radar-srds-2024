"""unit test for matrice generation
"""
from ..matrice_gen import init_foo_evals
import random 
class TestMatriceGen: 
    """
    Test for the matric generated for testing purposes. 
    """
    def setup_class(self):
        self.test_matrice_4 = init_foo_evals(4)

    def test_matrice_square(self): 
        # make sure the matrix is square
        assert len(self.test_matrice_4) == 4
        key,evaluations = random.choice(list(self.test_matrice_4.items()))
        assert len(evaluations) == 4

    def test_evaluations_range(self):
        # make sure that evaluations results are within expected range
        key,evaluations = random.choice(list(self.test_matrice_4.items()))
        key, single_evaluation = random.choice(list(evaluations.items()))
        assert single_evaluation <= 1
        assert single_evaluation >= 0.0