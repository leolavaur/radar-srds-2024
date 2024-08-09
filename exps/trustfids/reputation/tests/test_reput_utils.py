"""
unit test for reput_utils 
"""
import json

from ..utils import *

path_from_exec = "./trustfids/reputation/tests/"


class TestReputUtils:
    def test_discretize(self):
        inputs = [0.0, 0.2, 0.25, 0.35, 0.75, 0.9, 1.0]
        results = [1, 2, 2, 3, 7, 9, 9]
        tests = zip(inputs, results)
        for test in tests:
            assert test[1] == discretize(test[0], 10)

    def test_discretize_matrice(self):
        path_test_data = f"{path_from_exec}discretization_matrice_test.json"
        test_data = json.load(open(path_test_data))

        path_expected_results = (
            f"{path_from_exec}discretization_matrice_result_class_10.json"
        )
        expected_results = json.load(open(path_expected_results))

        c = 10
        d_mat = discretize_matrice(test_data, c)

        # Does the results looks correct ?
        for participant in d_mat.values():
            for eval in participant.values():
                assert 0 <= eval <= 9
                assert eval % 1 == 0

        # Is the result correct ?
        for participant in d_mat.keys():
            assert (
                d_mat["client_0"][participant]
                == expected_results["client_0"][participant]
            )

    def test_xeval_max(self):
        path_test_data = f"{path_from_exec}discretization_matrice_test.json"
        test_data = json.load(open(path_test_data))
        a = xeval_max(test_data)
        assert a >= 0

    def test_explode_reput(self):
        w:Dict[str,float] = {
            "a":0.19634522072191,
            "b":0.2009111432699602,
            "c":0.20091529942578645,
            "d":0.2009118763563134,
            "e":0.20091646022602985
        }
        w2 = {
             "attacker_000": 0.1963069469877501,
        "client_011": 0.20090924125994036,
        "client_013": 0.2009310700407662,
        "client_012": 0.2009310700407662,
        "client_010": 0.20092167167077707
        }
        w2:Dict[str,float] = explode_reput(w)
        assert math.isclose(1,sum(list(w2.values())))
    def test_normalized_matrice(self):
        path_test_data = f"{path_from_exec}normalize_matrice_test.json"
        test_data = json.load(open(path_test_data))
        normalized = normalize_matrice(test_data)
        assert xeval_max(normalized) <= 1.0

    def test_reverse_matrice(self):
        path_test_data = f"{path_from_exec}normalize_matrice_test.json"
        test_data = json.load(open(path_test_data))
        test_data = reverse_matrice(test_data)
        # 1
        assert test_data["client_1"]["client_1"] == 0
        # others tests in mind ?

    def test_(self):
        path_test_data = f"{path_from_exec}normalize_matrice_test.json"
        test_data = json.load(open(path_test_data))
        test_data = normalize_loss_matrice(test_data)

        # 9999999999999
        assert test_data["client_0"]["client_1"] < 1
        assert test_data["client_0"]["client_1"] > 0

        # 0
        assert test_data["client_0"]["client_0"] == 1

    def test_centroid_from_cluster(self):
        path_test_data = f"{path_from_exec}discretization_matrice_test.json"
        test_data = json.load(open(path_test_data))

        cluster = ["client_1", "client_2"]
        centroid = centroid_from_cluster(cluster, test_data)
        # Check that all participants have a value in the centroid
        assert len(centroid.values()) == 10
        # Test value if time ?
