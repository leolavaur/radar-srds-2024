"""unit test for plot_utils
"""
from numpy import extract
from pathlib import Path
from trustfids.utils.plot_utils import (
    Attack_Scenario,
    exclude_attackers,
    extract_attacks_stats,
    extract_attacker_dataset,
    load_poisoning_selector,
    load_change_in_poisoning,
    load_participants_from_dataset,
    load_attack_scenario,
    load_participants,
    load_metric,
)

path_from_exec = "./trustfids/utils/tests/"


class TestPlotUtils:
    """ """

    def setup_class(self):
        self.p = f"{path_from_exec}/acc-f1/"
        self.targeted = f"{path_from_exec}/targeted"
        self.untargeted = f"{path_from_exec}/untargeted"
        self.benign = f"{path_from_exec}/benign"
        self.oscillatory = f"{path_from_exec}/oscillatory"

    def test_extract_attacks_stats(self):
        attack_stats = extract_attacks_stats(self.p)
        assert "client_2" in attack_stats

    # def test_load_participants_from_dataset(self):
    #     ps = load_participants_from_dataset(self.p, "botiot_sampled")
    #     assert ps == ["client_0", "client_1"]

    # def test_extract_attacker_dataset(self):
    #     d = extract_attacker_dataset(self.targeted)
    #     assert d == "botiot"
    #     d_benign = extract_attacker_dataset(self.p)
    #     assert d_benign == ""

    def test_load_attack_scenario(self):
        attack_type, target, dt = load_attack_scenario(Path(self.targeted))
        assert attack_type == "targeted"
        assert target == "Reconnaissance"
        assert dt == "botiot"

        attack_type, target, dt = load_attack_scenario(Path(self.untargeted))
        assert attack_type == "untargeted"
        assert target == ""
        assert dt == "botiot"

        attack_type, target, dt = load_attack_scenario(Path(self.benign))
        assert attack_type == ""
        assert target == ""
        assert dt == ""

        ats = load_attack_scenario(Path(self.benign))
        assert isinstance(ats, Attack_Scenario)
        assert ats.attack_type == ""

    def test_load_participants(self):
        assert len(load_participants(Path(self.benign), attacker=False)) == 20
        assert len(load_participants(Path(self.targeted), attacker=False)) == 19
        assert len(load_participants(Path(self.targeted), attacker=True)) == 20

    def test_load_metric(self):
        assert len(load_metric((Path(self.targeted)))) == 20
        assert (
            len(load_metric((Path(self.targeted)), attacker=False, metric="missrate"))
            == 19
        )

    def test_load_poisoning_selector(self):
        # TODO investigate why benign have a poisoning ratio.
        assert load_poisoning_selector(Path(self.oscillatory)) == "1.0-1.0{4}+1.0{10}"
        assert load_poisoning_selector(Path(self.benign)) == "1.0"
        assert load_poisoning_selector(Path(self.targeted)) == "1.0"

    def test_load_change_in_poisoning(self):
        a = load_change_in_poisoning(Path(self.oscillatory))
        print("bb")

    def test_exclude_attackers(self):
        partition1 = [
            ["client_000", "client_001"],
            ["attacker_000", "attacker_001", "attacker_002"],
            ["client_002"],
        ]
        partition2 = [
            ["client_000", "client_001"],
            ["client_002", "attacker_000", "attacker_001", "attacker_002"],
        ]
        partition_target = [["client_000", "client_001"], ["client_002"]]
        assert exclude_attackers(partition1) == partition_target
        assert exclude_attackers(partition2) == partition_target
