"""unit test for plot_cluster.py
"""
from pathlib import Path
from typing import OrderedDict

from trustfids.utils.plot_cluster import (
    attacker_as_a_dataset,
    create_square_ndarray,
    extract_xevals,
    get_rand_over_round,
    load_distribution,
    plot_evals_2D,
)
from trustfids.utils.tests.data_examples import distribution_attacker, example_xeval


class TestUtils:
    """ """

    def setup_class(self):
        self.distrib_path = "./trustfids/utils/tests/"
        self.f1_f1_path = "./trustfids/utils/tests/f1-f1/"
    def test_load_distribution(self):
        distrib = load_distribution(Path(self.distrib_path))
        assert ["client_0", "client_1"] in distrib.values()
        assert len(distrib) == 5

    def test_create_square_ndarray(self):
        data = {
        'A': {'A': 1.0, 'B': 2.0, 'C': 3.0},
        'B': {'A': 4.0, 'B': 5.0, 'C': 6.0},
        'C': {'A': 7.0, 'B': 8.0, 'C': 9.0}
        }
        testnd = create_square_ndarray(data)
        assert testnd[1,1] == 5.0
        testnd_xeval = create_square_ndarray(example_xeval)
        assert example_xeval["client_0"]["client_0"] == testnd_xeval[0,0]
        assert example_xeval["client_9"]["client_9"] == testnd_xeval[9,9]
        assert example_xeval["client_8"]["client_9"] == testnd_xeval[8,9]

    def test_extract_xevals(self):
        extract_xevals(self.f1_f1_path)
    
    def test_attacker_as_a_dataset(self):
        n_distrib = attacker_as_a_dataset(OrderedDict(distribution_attacker))
        assert len(list(n_distrib.keys())) == 5
        assert list(n_distrib.keys())[0] == "cicids" 
        assert list(n_distrib.keys())[3] == "botiot_attacker"
        assert len(n_distrib["botiot_attacker"]) == 3
        
    # def test_plot_evals_2D(self):
    #     plot_evals_2D(self.f1_f1_path)
    # def test_get_rand_over_round(self):
    #     in_dir1 = "./trustfids/utils/tests/acc_acc/"
    #     in_dir2 = "./trustfids/utils/tests/loss_acc/"

    #     get_rand_over_round(in_dir1)
    #     get_rand_over_round(in_dir1, in_dir2)
