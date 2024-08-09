"""unit test for stats_utils
"""
from trustfids.utils.stats_utils import get_missrate

path_from_exec = "./trustfids/utils/tests/"

class TestStatsUtils:
    """ """
    def setup_class(self):
        self.p= f"{path_from_exec}/acc-f1/"
        
        
    def test_get_missrate(self):
        # TODO test with attacker in the dataset. 
        missrate,detection_rate = get_missrate(self.p,"botiot_sampled","Reconnaissance")
        assert missrate == 2 / (7058+6736)
        assert detection_rate == (7058+6736-2) / (7058+6736)