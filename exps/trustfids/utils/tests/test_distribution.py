"""File containing unit test for distribution.py
"""

from trustfids.utils.distribution import *


def test_build_distribution_baseline():
    """ """
    a = [2, 2, 4, 2]
    b = build_distribution_baseline(a)
    assert len(b) == 4
    assert b[0] == ["client_0", "client_1"]


def test_build_merged_distribution_baseline():
    """ """
    a = [2, 2, 4, 2]
    b = build_merged_distribution_baseline(a)
    assert b[0] == ["client_0", "client_1", "client_8", "client_9"]
    assert len(b) == 3


def test_distribute_clients() -> None:
    """Test distribute_clients."""

    # Assert
    assert sum(distribute_clients(10, [100, 200, 300])) == 10
    assert distribute_clients(10, [200, 200, 300, 300]) == [2, 2, 3, 3]
    assert distribute_clients(10, [100, 200, 400, 300], min=2) == [2, 2, 3, 3]
    assert distribute_clients(10, [3569, 880623, 960078, 359618], min=2) == [2, 3, 3, 2]
